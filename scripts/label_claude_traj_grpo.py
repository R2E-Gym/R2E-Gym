import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_FUNCTION_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_]+)\s*>")
_PARAM_RE = re.compile(
    r"(?s)<parameter\s*=\s*([a-zA-Z0-9_]+)\s*>(.*?)</parameter>"
)


_TEST_CMD_RE = re.compile(
    r"(?i)\b(pytest|go\s+test|npm\s+test|yarn\s+test|pnpm\s+test|jest|mocha)\b"
)


@dataclass
class ParsedAction:
    function_name: str
    parameters: Dict[str, str]


def _parse_action(xml_like: str) -> ParsedAction:
    text = xml_like or ""
    m = _FUNCTION_RE.search(text)
    fn = m.group(1) if m else ""
    params: Dict[str, str] = {}
    for k, v in _PARAM_RE.findall(text):
        params[k] = (v or "").strip()
    return ParsedAction(function_name=fn, parameters=params)


def _parse_exit_code(observation: str) -> Optional[int]:
    if not observation:
        return None
    m = re.search(r"^Exit code:\s*(-?\d+)\s*$", observation, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_error_observation(observation: str) -> bool:
    if not observation:
        return False
    if "Error:" in observation:
        return True
    if "Traceback (most recent call last)" in observation:
        return True
    if "APIError(" in observation:
        return True
    ec = _parse_exit_code(observation)
    if ec is not None and ec < 0:
        return True
    return False


def _is_env_409(observation: str) -> bool:
    return bool(observation) and ("409 Client Error: Conflict" in observation)


def _cmd_is_test(cmd: str) -> bool:
    return bool(cmd) and bool(_TEST_CMD_RE.search(cmd))


def _step_labels(step: Dict[str, Any]) -> Dict[str, Any]:
    action = step.get("action") or ""
    obs = step.get("observation") or ""
    parsed = _parse_action(action)
    fn = parsed.function_name
    params = parsed.parameters
    cmd = params.get("cmd") or ""
    path = params.get("path") or ""
    command = params.get("command") or ""
    is_error = _is_error_observation(obs)
    is_409 = _is_env_409(obs)

    is_dir_view = fn == "file_editor" and command == "view" and path == "/testbed"
    is_file_view = (
        fn == "file_editor"
        and command == "view"
        and path.startswith("/testbed/")
        and path != "/testbed"
    )
    is_edit = fn == "file_editor" and command in {
        "str_replace",
        "insert",
        "create",
        "undo_edit",
    }
    is_test_cmd = fn == "execute_bash" and _cmd_is_test(cmd)

    return {
        "tool": fn or "unknown",
        "tool_params": {
            "command": command,
            "path": path,
            "has_cmd": bool(cmd),
        },
        "exit_code": _parse_exit_code(obs),
        "is_error": is_error,
        "is_env_error_409": is_409,
        "is_dir_view": is_dir_view,
        "is_file_view": is_file_view,
        "is_edit": is_edit,
        "is_test_cmd": is_test_cmd,
    }


def _episode_labels(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_labels = [s.get("step_labels") for s in steps if s.get("step_labels")]
    total = len(step_labels)
    if total == 0:
        return {
            "env_unavailable": True,
            "opened_specific_file": False,
            "did_edit": False,
            "ran_tests": False,
            "effective_step_ratio": 0.0,
            "dir_view_count": 0,
            "dir_view_streak_max": 0,
            "error_step_count": 0,
            "env_409_step_count": 0,
        }

    error_cnt = sum(1 for t in step_labels if t.get("is_error"))
    env_409_cnt = sum(1 for t in step_labels if t.get("is_env_error_409"))
    opened_specific = any(t.get("is_file_view") for t in step_labels)
    did_edit = any(t.get("is_edit") for t in step_labels)
    ran_tests = any(t.get("is_test_cmd") for t in step_labels)
    dir_view_cnt = sum(1 for t in step_labels if t.get("is_dir_view"))
    ok_cnt = total - error_cnt
    eff = ok_cnt / total if total else 0.0
    env_unavailable = (env_409_cnt > 0) and (eff < 0.3)

    streak = 0
    streak_max = 0
    for t in step_labels:
        if t.get("is_dir_view"):
            streak += 1
            streak_max = max(streak_max, streak)
        else:
            streak = 0

    return {
        "env_unavailable": env_unavailable,
        "opened_specific_file": opened_specific,
        "did_edit": did_edit,
        "ran_tests": ran_tests,
        "effective_step_ratio": round(eff, 6),
        "dir_view_count": dir_view_cnt,
        "dir_view_streak_max": streak_max,
        "error_step_count": error_cnt,
        "env_409_step_count": env_409_cnt,
    }


def _episode_weight(
    reward: Optional[float],
    labels: Dict[str, Any],
    allow_env_unavailable: bool,
) -> float:
    eff = float(labels.get("effective_step_ratio") or 0.0)
    quality = 0.2 + 0.8 * eff
    base = 1.0
    if reward is not None:
        try:
            r = float(reward)
            base = 1.0 + 2.0 * max(0.0, min(1.0, r))
        except Exception:
            base = 1.0
    loop_penalty = 1.0 / (1.0 + 0.05 * float(labels.get("dir_view_count") or 0))
    w = base * quality * loop_penalty
    if labels.get("env_unavailable") and not allow_env_unavailable:
        return 0.0
    return float(max(0.0, min(10.0, w)))


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i + 1, json.loads(line)
            except Exception:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow-env-unavailable", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = output_path.open("w", encoding="utf-8")
    try:
        for _, obj in iter_jsonl(input_path):
            steps = obj.get("trajectory_steps") or []
            if isinstance(steps, list):
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    s["step_labels"] = _step_labels(s)

            obj["episode_labels"] = _episode_labels(steps if isinstance(steps, list) else [])
            reward = obj.get("reward")
            obj["episode_weight"] = _episode_weight(
                reward=reward if isinstance(reward, (int, float)) else None,
                labels=obj["episode_labels"],
                allow_env_unavailable=bool(args.allow_env_unavailable),
            )

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        out_f.close()


if __name__ == "__main__":
    main()

