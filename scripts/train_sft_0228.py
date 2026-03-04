import argparse
import json
import math
import os
import random
import re
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


_RE_FUNC = re.compile(r"<function\s*=\s*([a-zA-Z0-9_]+)>")
_RE_PARAM = re.compile(r"(?s)<parameter\s*=\s*([a-zA-Z0-9_]+)>(.*?)</parameter>")


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


def _step_label(obj: dict, step_idx: int) -> dict:
    cto = obj.get("custom_test_outputs") or {}
    if isinstance(cto, dict):
        steps = cto.get("step_labels_0228") or []
        if isinstance(steps, list) and step_idx < len(steps) and isinstance(steps[step_idx], dict):
            return steps[step_idx]
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
class EncodedSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def _encode_sample(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> EncodedSample:
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    r_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        r_ids = r_ids + [tokenizer.eos_token_id]
    input_ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids[:]
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    attention_mask = [1] * len(input_ids)
    return EncodedSample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def _make_dataset(
    path: Path,
    context_steps: int,
    obs_max_chars: int,
    max_length: int,
    require_evaluable: bool,
    require_positive_terminal: bool,
    keep_tools: Optional[str],
    seed: int,
    tokenizer,
) -> Dataset:
    rng = random.Random(seed)
    keep_set = None
    if keep_tools:
        keep_set = {x.strip() for x in keep_tools.split(",") if x.strip()}

    rows = []
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
            action = str(step.get("action") or "")
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
            response = action.strip()
            enc = _encode_sample(tokenizer, prompt, response, max_length=max_length)
            if not enc.input_ids:
                continue
            rows.append(
                {
                    "input_ids": enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "labels": enc.labels,
                }
            )

    rng.shuffle(rows)
    return Dataset.from_list(rows)


class _PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            am = f["attention_mask"]
            lb = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append(am + [0] * pad)
            labels.append(lb + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--context-steps", type=int, default=6)
    ap.add_argument("--obs-max-chars", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.98)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--require-evaluable", action="store_true", default=True)
    ap.add_argument("--require-positive-terminal", action="store_true", default=False)
    ap.add_argument("--keep-tools", default=None)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--fp16", action="store_true", default=False)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--ddp-find-unused-parameters", action="store_true", default=False)
    ap.add_argument("--ddp-timeout", type=int, default=1800)
    ap.add_argument("--tf32", action="store_true", default=False)
    ap.add_argument("--disable-tqdm", action="store_true", default=False)
    ap.add_argument("--logging-first-step", action="store_true", default=True)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(args.model_name_or_path)
    is_local_dir = os.path.isdir(model_id)
    if (model_id.startswith("/") or model_id.startswith("./") or model_id.startswith("../")) and not is_local_dir:
        raise SystemExit(
            "model_path_not_found_or_not_a_dir: "
            + model_id
            + " (pass a local model directory containing config.json/tokenizer files, or a HF repo id like namespace/name)"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, local_files_only=is_local_dir
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    ds = _make_dataset(
        data_path,
        context_steps=int(args.context_steps),
        obs_max_chars=int(args.obs_max_chars),
        max_length=int(args.max_length),
        require_evaluable=bool(args.require_evaluable),
        require_positive_terminal=bool(args.require_positive_terminal),
        keep_tools=args.keep_tools,
        seed=int(args.seed),
        tokenizer=tokenizer,
    )

    n = len(ds)
    if n < 2:
        raise SystemExit(f"dataset_too_small={n}")
    n_train = max(1, int(n * float(args.train_ratio)))
    n_eval = max(1, n - n_train)
    train_ds = ds.select(range(0, n_train))
    eval_ds = ds.select(range(n_train, n_train + n_eval))

    rank = int(os.environ.get("RANK", "0") or "0")
    if rank == 0:
        print(
            json.dumps(
                {
                    "dataset_total": n,
                    "dataset_train": len(train_ds),
                    "dataset_eval": len(eval_ds),
                    "model": model_id,
                    "deepspeed": str(args.deepspeed or ""),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=is_local_dir)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if tokenizer.vocab_size != getattr(model.get_input_embeddings(), "num_embeddings", tokenizer.vocab_size):
        model.resize_token_embeddings(len(tokenizer))

    collator = _PadCollator(tokenizer.pad_token_id)

    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    targs_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": float(args.num_train_epochs),
        "warmup_ratio": float(args.warmup_ratio),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "eval_steps": int(args.eval_steps),
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": [],
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "seed": int(args.seed),
    }
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if args.deepspeed:
        targs_kwargs["deepspeed"] = str(args.deepspeed)
    if "ddp_find_unused_parameters" in sig_params:
        targs_kwargs["ddp_find_unused_parameters"] = bool(args.ddp_find_unused_parameters)
    if "ddp_timeout" in sig_params:
        targs_kwargs["ddp_timeout"] = int(args.ddp_timeout)
    if args.tf32 and "tf32" in sig_params:
        targs_kwargs["tf32"] = True
    if "logging_first_step" in sig_params:
        targs_kwargs["logging_first_step"] = bool(args.logging_first_step)
    if "disable_tqdm" in sig_params:
        targs_kwargs["disable_tqdm"] = bool(args.disable_tqdm)
    if "log_level" in sig_params:
        targs_kwargs["log_level"] = str(args.log_level)
    if "log_on_each_node" in sig_params:
        targs_kwargs["log_on_each_node"] = False
    if "eval_strategy" in sig_params:
        targs_kwargs["eval_strategy"] = "steps"
    else:
        targs_kwargs["evaluation_strategy"] = "steps"
    targs_kwargs = {k: v for k, v in targs_kwargs.items() if k in sig_params}
    targs = TrainingArguments(**targs_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
