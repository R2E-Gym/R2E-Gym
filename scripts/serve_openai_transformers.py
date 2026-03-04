import argparse
import json
import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from flask import Flask, Response, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d in {"cpu", "cuda", "mps"}:
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    dt = (dtype or "auto").lower()
    if dt == "fp16":
        return torch.float16
    if dt == "bf16":
        return torch.bfloat16
    if dt == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _extract_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs = payload.get("messages")
    if isinstance(msgs, list) and msgs and all(isinstance(x, dict) for x in msgs):
        out = []
        for m in msgs:
            role = str(m.get("role") or "")
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(str(it.get("text") or ""))
                content = "\n".join([p for p in parts if p])
            out.append({"role": role, "content": str(content or "")})
        return out
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": ""}]


def _build_input(
    tokenizer,
    messages: List[Dict[str, str]],
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    s = ""
    for m in messages:
        s += f"{m.get('role','user')}: {m.get('content','')}\n"
    s += "assistant: "
    return s


def _num(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true", default=False)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens-default", type=int, default=160)
    ap.add_argument("--use-fast-tokenizer", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    model_dir = str(args.model)
    if not os.path.isdir(model_dir) and not ("/" in model_dir and not model_dir.startswith("/")):
        raise SystemExit("model_path_not_found_or_not_a_dir: " + model_dir)

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=bool(args.use_fast_tokenizer),
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    logger.info(f"Loading model from {model_dir} with device={device} and dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.eval()
    model.to(device)
    logger.info("Model loaded successfully")
    if tokenizer.vocab_size != getattr(model.get_input_embeddings(), "num_embeddings", tokenizer.vocab_size):
        model.resize_token_embeddings(len(tokenizer))

    gen_lock = Lock()
    app = Flask(__name__)

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    @app.get("/")
    def root():
        return jsonify({"status": "ok"})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/v1/models")
    def v1_models():
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": "local-transformers",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    def v1_chat_completions():
        payload = request.get_json(silent=True) or {}
        logger.info(f"Received completion request: {payload}")
        messages = _extract_messages(payload)
        prompt = _build_input(tokenizer, messages)
        logger.info(f"Processed prompt: {prompt[:100]}...")

        max_new_tokens = _int(payload.get("max_tokens"), None)
        if max_new_tokens is None:
            max_new_tokens = _int(payload.get("max_completion_tokens"), None)
        if max_new_tokens is None:
            max_new_tokens = int(args.max_new_tokens_default)
        max_new_tokens = max(1, int(max_new_tokens))

        temperature = _num(payload.get("temperature"), 0.0)
        top_p = _num(payload.get("top_p"), None)
        top_k = _int(payload.get("top_k"), None)
        do_sample = bool(temperature is not None and float(temperature) > 0.0)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_input_tokens),
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with gen_lock, torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=float(temperature or 0.0),
                top_p=float(top_p) if top_p is not None else None,
                top_k=int(top_k) if top_k is not None else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen = out[0, enc["input_ids"].shape[1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True)

        prompt_tokens = int(enc["input_ids"].shape[1])
        completion_tokens = int(gen.shape[0])
        total_tokens = prompt_tokens + completion_tokens

        resp = {
            "id": "chatcmpl-" + str(int(time.time() * 1000)),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model") or "local-transformers"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        logger.info(f"Generated response: {resp}")
        return Response(json.dumps(resp, ensure_ascii=False), mimetype="application/json")

    @app.errorhandler(Exception)
    def handle_error(e):
        return jsonify({"error": str(e), "status": "error"}), 500

    app.run(host=str(args.host), port=int(args.port), threaded=True, processes=int(args.workers))


if __name__ == "__main__":
    main()
