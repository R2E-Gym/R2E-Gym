import argparse
import json
import os
import random
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_RE_FUNC = re.compile(r"<function\s*=\s*([a-zA-Z0-9_]+)>")
_RE_PARAM = re.compile(r"(?s)<parameter\s*=\s*([a-zA-Z0-9_]+)>(.*?)</parameter>")
_RE_TOOL_BLOCK = re.compile(r"(?s)<function\s*=\s*[a-zA-Z0-9_]+>.*?</function>")


def _loads_jsonl(path: Path):
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_action_xml(action: str) -> Tuple[str, Dict[str, str]]:
    if not isinstance(action, str):
        return "", {}
    m = _RE_FUNC.search(action)
    fn = m.group(1).strip() if m else ""
    params = {}
    for k, v in _RE_PARAM.findall(action):
        params[k.strip()] = v.strip()
    return fn, params


def _clip_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.7)]
    tail = s[-int(max_chars * 0.2) :]
    return head + "\n...\n<response clipped>\n...\n" + tail


def _get_ds(obj: dict) -> dict:
    ds = obj.get("ds")
    if isinstance(ds, dict):
        return ds
    env_args = obj.get("env_args") or {}
    if isinstance(env_args, dict):
        ds2 = env_args.get("ds")
        if isinstance(ds2, dict):
            return ds2
    return {}


def _label_info(obj: dict) -> dict:
    cto = obj.get("custom_test_outputs") or {}
    if isinstance(cto, dict):
        lab = cto.get("labels_0228")
        if isinstance(lab, dict):
            return lab
    return {}


def _build_prompt(
    problem_statement: str,
    repo_language: str,
    selected_tests: Any,
    history: List[Dict[str, str]],
    steps_remaining: int,
) -> str:
    env = {
        "repo_root": "/testbed",
        "repo_language": repo_language,
        "selected_test_files_to_run": selected_tests,
    }
    h = ""
    for it in history:
        a = (it.get("action") or "").strip()
        o = (it.get("observation") or "").strip()
        if a:
            h += f"\n[ACTION]\n{a}\n"
        if o:
            h += f"\n[OBSERVATION]\n{o}\n"
        h += "\n---\n"
    prompt = (
        "You are a programming agent operating in a repository mounted at /testbed.\n"
        "Return exactly one tool call block in the XML format:\n"
        "<function=TOOL_NAME>\n"
        "  <parameter=KEY>VALUE</parameter>\n"
        "  ...\n"
        "</function>\n"
        "Allowed TOOL_NAME: file_editor, search, execute_bash, finish\n"
        "Only call one tool.\n\n"
        "<github_issue>\n"
        + (problem_statement or "").strip()
        + "\n</github_issue>\n\n"
        + "[ENV]\n"
        + json.dumps(env, ensure_ascii=False)
        + "\n\n"
        + f"Steps Remaining: {steps_remaining}\n"
        + (("\n[HISTORY]\n" + h) if h.strip() else "\n[HISTORY]\n<empty>\n")
        + "\nNow output the next tool call.\n"
    )
    return prompt


@dataclass
class Sample:
    prompt: str
    response: str


def _make_samples(
    path: Path,
    context_steps: int,
    obs_max_chars: int,
    require_evaluable: bool,
    require_positive_terminal: bool,
    keep_tools: Optional[str],
    seed: int,
) -> List[Sample]:
    rng = random.Random(seed)
    keep_set = None
    if keep_tools:
        keep_set = {x.strip() for x in keep_tools.split(",") if x.strip()}

    samples: List[Sample] = []
    for obj in _loads_jsonl(path):
        lab = _label_info(obj)
        if require_evaluable and not bool(lab.get("evaluable", True)):
            continue
        reward_gated = lab.get("reward_gated")
        if require_positive_terminal and not (isinstance(reward_gated, (int, float)) and float(reward_gated) == 1.0):
            continue

        ds = _get_ds(obj)
        repo_language = str(ds.get("repo_language") or "")
        selected_tests = ds.get("selected_test_files_to_run")
        problem = str(obj.get("problem_statement") or "")
        steps = obj.get("trajectory_steps") or []
        if not isinstance(steps, list) or not steps:
            continue

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").strip()
            fn, _ = _parse_action_xml(action)
            if keep_set is not None and fn not in keep_set:
                continue
            start = max(0, i - context_steps)
            history = []
            for j in range(start, i):
                prev = steps[j]
                if not isinstance(prev, dict):
                    continue
                history.append(
                    {
                        "action": str(prev.get("action") or ""),
                        "observation": _clip_text(prev.get("observation") or "", obs_max_chars),
                    }
                )
            steps_remaining = max(int(obj.get("max_steps") or 0) - int(step.get("step_count") or (i + 1)), 0)
            prompt = _build_prompt(
                problem_statement=problem,
                repo_language=repo_language,
                selected_tests=selected_tests,
                history=history,
                steps_remaining=steps_remaining,
            )
            samples.append(Sample(prompt=prompt, response=action))

    rng.shuffle(samples)
    return samples


def _pick_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d in {"cpu", "cuda", "mps"}:
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_tool_block(text: str) -> str:
    if not text:
        return ""
    m = _RE_TOOL_BLOCK.search(text)
    return m.group(0).strip() if m else ""


def _is_valid_tool_call(text: str) -> bool:
    block = _extract_tool_block(text or "")
    fn, _ = _parse_action_xml(block)
    return fn in {"file_editor", "search", "execute_bash", "finish"}


def _has_usable_model_config(dir_path: str) -> bool:
    cfg = os.path.join(dir_path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mt = obj.get("model_type")
        return isinstance(mt, str) and bool(mt.strip())
    except Exception:
        return False


def _resolve_model_dir(model_id: str) -> str:
    if not os.path.isdir(model_id):
        return model_id

    if _has_usable_model_config(model_id):
        return model_id

    final_dir = os.path.join(model_id, "final")
    if _has_usable_model_config(final_dir):
        return final_dir

    ckpts = []
    for p in glob.glob(os.path.join(model_id, "checkpoint-*")):
        name = os.path.basename(p)
        m = re.match(r"checkpoint-(\d+)$", name)
        if not m:
            continue
        if _has_usable_model_config(p):
            ckpts.append((int(m.group(1)), p))
    if ckpts:
        ckpts.sort()
        return ckpts[-1][1]

    raise SystemExit(
        "model_dir_missing_usable_config_json: "
        + model_id
        + " (expected config.json with a non-empty model_type in the directory itself, or in ./final/, or in ./checkpoint-*/)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--context-steps", type=int, default=6)
    ap.add_argument("--obs-max-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-evaluable", action="store_true", default=True)
    ap.add_argument("--require-positive-terminal", action="store_true", default=False)
    ap.add_argument("--keep-tools", default=None)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true", default=False)
    args = ap.parse_args()

    data_path = Path(args.data)
    model_id = _resolve_model_dir(str(args.model_name_or_path))
    is_local_dir = os.path.isdir(model_id)

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=is_local_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

    samples = _make_samples(
        data_path,
        context_steps=int(args.context_steps),
        obs_max_chars=int(args.obs_max_chars),
        require_evaluable=bool(args.require_evaluable),
        require_positive_terminal=bool(args.require_positive_terminal),
        keep_tools=args.keep_tools,
        seed=int(args.seed),
    )
    if not samples:
        raise SystemExit("no_samples_built_from_data")

    n = min(int(args.max_samples), len(samples))
    samples = samples[:n]

    device = _pick_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=is_local_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.eval()
    model.to(device)
    if tok.vocab_size != getattr(model.get_input_embeddings(), "num_embeddings", tok.vocab_size):
        model.resize_token_embeddings(len(tok))

    bs = max(1, int(args.batch_size))
    max_inp = int(args.max_input_tokens)
    max_new = int(args.max_new_tokens)

    exact = 0
    valid = 0
    tool_match = 0
    total = 0
    examples = []

    for start in range(0, len(samples), bs):
        batch = samples[start : start + bs]
        prompts = [s.prompt for s in batch]
        refs = [s.response.strip() for s in batch]
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_inp,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        preds = tok.batch_decode(gen, skip_special_tokens=True)

        for p, r in zip(preds, refs):
            pred_raw = (p or "").strip()
            pred = _extract_tool_block(pred_raw) or pred_raw
            total += 1
            if _is_valid_tool_call(pred):
                valid += 1
            if pred == r:
                exact += 1
            pf, _ = _parse_action_xml(pred)
            rf, _ = _parse_action_xml(r)
            if pf and rf and pf == rf:
                tool_match += 1
            if len(examples) < 5:
                examples.append({"pred": pred[:500], "ref": r[:500]})

    print(
        json.dumps(
            {
                "samples": total,
                "valid_tool_call_rate": valid / total if total else 0.0,
                "exact_match_rate": exact / total if total else 0.0,
                "tool_name_match_rate": tool_match / total if total else 0.0,
                "model": model_id,
                "device": str(device),
                "batch_size": bs,
                "max_input_tokens": max_inp,
                "max_new_tokens": max_new,
                "examples": examples,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
