import argparse
import hashlib
import json
import re
from pathlib import Path


_RE_FUNC = re.compile(r"<function\s*=\s*([a-zA-Z0-9_]+)>")
_RE_PARAM = re.compile(r"(?s)<parameter\s*=\s*([a-zA-Z0-9_]+)>(.*?)</parameter>")


def _loads_jsonl(path: Path):
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _dumps_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


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


def _parse_action_xml(action: str) -> dict:
    if not isinstance(action, str):
        return {}
    m = _RE_FUNC.search(action)
    fn = m.group(1).strip() if m else ""
    params = {}
    for k, v in _RE_PARAM.findall(action):
        params[k.strip()] = v.strip()
    return {"function": fn, "params": params}


def _is_diff_patch(patch: str) -> bool:
    if not patch:
        return False
    return patch.lstrip().startswith("diff --git ")


def _invert_unified_diff(patch: str) -> str:
    if not patch or not _is_diff_patch(patch):
        return ""
    out = []
    pending_old = None
    for ln in patch.splitlines(True):
        if ln.startswith("diff --git "):
            pending_old = None
            out.append(ln)
            continue
        if ln.startswith("index "):
            m = re.match(r"index\s+([0-9a-f]+)\.\.([0-9a-f]+)(.*)\n?$", ln)
            if m:
                out.append(f"index {m.group(2)}..{m.group(1)}{m.group(3)}\n")
            else:
                out.append(ln)
            continue
        if ln.startswith("--- "):
            pending_old = ln[4:].rstrip("\n")
            continue
        if ln.startswith("+++ "):
            newp = ln[4:].rstrip("\n")
            oldp = pending_old
            if oldp is None:
                out.append("--- " + newp + "\n")
                out.append("+++ " + newp + "\n")
            else:
                out.append("--- " + newp + "\n")
                out.append("+++ " + oldp + "\n")
            pending_old = None
            continue
        if ln.startswith("new file mode "):
            out.append("deleted file mode " + ln[len("new file mode ") :])
            continue
        if ln.startswith("deleted file mode "):
            out.append("new file mode " + ln[len("deleted file mode ") :])
            continue
        if ln.startswith("@@"):
            out.append(ln)
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            out.append("-" + ln[1:])
            continue
        if ln.startswith("-") and not ln.startswith("---"):
            out.append("+" + ln[1:])
            continue
        out.append(ln)
    return "".join(out)


def _detect_did_edit(steps: list[dict]) -> bool:
    for s in steps or []:
        a = s.get("action") or ""
        obs = s.get("observation") or ""
        pa = _parse_action_xml(a)
        if pa.get("function") != "file_editor":
            continue
        cmd = str(pa.get("params", {}).get("command") or "").strip()
        if cmd not in {"create", "insert", "str_replace", "undo_edit"}:
            continue
        if "has been edited" in str(obs) or "File created at" in str(obs):
            return True
        return True
    return False


def _detect_ran_tests(steps: list[dict]) -> bool:
    pat = re.compile(r"(?i)\b(pytest|go\s+test|npm\s+test|yarn\s+test|pnpm\s+test|jest|mocha)\b")
    for s in steps or []:
        a = s.get("action") or ""
        pa = _parse_action_xml(a)
        if pa.get("function") != "execute_bash":
            continue
        cmd = str(pa.get("params", {}).get("cmd") or "")
        if pat.search(cmd):
            return True
    return False


def _detect_test_summary(repo_language: str, test_output: str) -> dict:
    s = test_output or ""
    s_low = s.lower()
    if not s.strip():
        return {"passed": None, "reason": "no_test_output"}
    if "[PREFLIGHT]" in s:
        m = re.search(r"reward_unavailable:\s*([a-zA-Z0-9_]+)", s)
        return {"passed": None, "reason": f"preflight_{m.group(1) if m else 'unknown'}"}
    if "ERROR: file or directory not found:" in s:
        return {"passed": False, "reason": "runner_path_not_found"}

    lang = (repo_language or "").lower()
    if "python" in lang:
        if re.search(r"\b\d+\s+failed\b", s_low) or "error" in s_low and "errors" in s_low:
            return {"passed": False, "reason": "pytest_failed"}
        if re.search(r"\b\d+\s+passed\b", s_low) and not re.search(r"\b\d+\s+failed\b", s_low):
            return {"passed": True, "reason": "pytest_passed"}
        if "no tests ran" in s_low:
            return {"passed": False, "reason": "pytest_no_tests"}
        return {"passed": None, "reason": "pytest_unknown"}

    if lang in {"js", "javascript", "ts", "typescript"} or "node" in lang:
        neg = re.search(r"(?i)\b(failed|failing|assertions failed)\b", s)
        if neg:
            return {"passed": False, "reason": "node_failed"}
        if re.search(r"Test Suites:\s*\d+\s+passed,\s*\d+\s+total", s) and re.search(
            r"Tests:\s*\d+\s+passed,\s*\d+\s+total", s
        ):
            return {"passed": True, "reason": "jest_passed"}
        if re.search(r"All\s+\d+\s+assertions\s+passed", s):
            return {"passed": True, "reason": "assertions_passed"}
        if re.search(r"\b0 failing\b", s_low) or re.search(r"\b0 failed\b", s_low):
            return {"passed": True, "reason": "summary_zero_failed"}
        return {"passed": None, "reason": "node_unknown"}

    if "go" in lang:
        if re.search(r"--- FAIL:", s):
            return {"passed": False, "reason": "go_failed"}
        if re.search(r"PASS\b", s) and not re.search(r"FAIL\b", s):
            return {"passed": True, "reason": "go_passed"}
        return {"passed": None, "reason": "go_unknown"}

    return {"passed": None, "reason": "unknown_lang"}


def _label_episode(obj: dict, require_edit_for_success: bool) -> tuple[dict, list[dict], dict]:
    ds = _get_ds(obj)
    repo_language = str(ds.get("repo_language") or "").lower()
    steps = obj.get("trajectory_steps") or []
    if not isinstance(steps, list):
        steps = []

    test_output = obj.get("test_output") or ""
    reward_raw = obj.get("reward")
    exit_reason = obj.get("exit_reason")

    evaluable = True
    unevaluable_reason = ""
    if reward_raw is None and "[PREFLIGHT]" in test_output:
        evaluable = False
        m = re.search(r"reward_unavailable:\s*([a-zA-Z0-9_]+)", test_output)
        unevaluable_reason = m.group(1) if m else "unknown"

    patch = obj.get("output_patch") or ""
    has_patch = bool(patch.strip())
    is_diff = _is_diff_patch(patch)
    did_edit = _detect_did_edit(steps) or (has_patch and is_diff)
    ran_tests = _detect_ran_tests(steps)

    summary = _detect_test_summary(repo_language, test_output)
    t_passed = summary.get("passed")

    reward_gated = reward_raw
    if reward_raw is not None:
        try:
            r = float(reward_raw)
        except Exception:
            r = None
        if r is not None and r == 1.0 and require_edit_for_success and not did_edit:
            reward_gated = 0.0
        else:
            reward_gated = r

    step_labels = []
    dir_view_streak = 0
    for s in steps:
        a = s.get("action") or ""
        obs = s.get("observation") or ""
        pa = _parse_action_xml(a)
        fn = pa.get("function") or ""
        params = pa.get("params") or {}
        cmd = str(params.get("command") or "").strip()
        path = str(params.get("path") or "").strip()
        is_dir_view = fn == "file_editor" and cmd == "view" and path == "/testbed"
        is_file_view = bool(fn == "file_editor" and cmd == "view" and path and path != "/testbed")
        is_edit = fn == "file_editor" and cmd in {"create", "insert", "str_replace", "undo_edit"}
        is_test = False
        if fn == "execute_bash":
            cmdline = str(params.get("cmd") or "")
            is_test = bool(
                re.search(
                    r"(?i)\b(pytest|go\s+test|npm\s+test|yarn\s+test|pnpm\s+test|jest|mocha)\b",
                    cmdline,
                )
            )
        if is_dir_view:
            dir_view_streak += 1
        else:
            dir_view_streak = 0
        is_error = bool(
            re.search(r"(?i)\b(traceback|apierror|permissiondenied)\b", str(obs))
            or "Error: Exit code" in str(obs)
            or "Exception occurred" in str(obs)
        )
        step_labels.append(
            {
                "step_idx": s.get("step_idx"),
                "fn": fn,
                "cmd": cmd,
                "path": path,
                "is_dir_view": is_dir_view,
                "is_file_view": is_file_view,
                "is_edit": is_edit,
                "is_test_cmd": is_test,
                "dir_view_streak": dir_view_streak,
                "is_error": is_error,
            }
        )

    episode_labels = {
        "repo": str(ds.get("repo") or ""),
        "instance_id": str(ds.get("instance_id") or ""),
        "repo_language": repo_language,
        "exit_reason": exit_reason,
        "evaluable": evaluable,
        "unevaluable_reason": unevaluable_reason,
        "has_patch": has_patch,
        "patch_is_diff": is_diff,
        "patch_sha1": _sha1(patch) if has_patch else "",
        "did_edit": did_edit,
        "ran_tests": ran_tests,
        "test_summary_passed": t_passed,
        "test_summary_reason": summary.get("reason"),
        "reward_raw": reward_raw,
        "reward_gated": reward_gated,
    }

    inject_patch = ""
    if is_diff and has_patch:
        inject_patch = _invert_unified_diff(patch)
    inject_meta = {
        "inject_patch": inject_patch,
        "inject_patch_sha1": _sha1(inject_patch) if inject_patch else "",
    }

    return episode_labels, step_labels, inject_meta


def _build_grpo_samples(
    obj: dict,
    episode_labels: dict,
    step_labels: list[dict],
    context_steps: int,
    obs_max_chars: int,
):
    ds = _get_ds(obj)
    steps = obj.get("trajectory_steps") or []
    if not isinstance(steps, list):
        steps = []

    def clip(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if obs_max_chars <= 0 or len(s) <= obs_max_chars:
            return s
        head = s[: int(obs_max_chars * 0.7)]
        tail = s[-int(obs_max_chars * 0.2) :]
        return head + "\n...\n<response clipped>\n...\n" + tail

    problem = str(obj.get("problem_statement") or "")
    env_info = {
        "repo_language": str(ds.get("repo_language") or ""),
        "selected_test_files_to_run": ds.get("selected_test_files_to_run"),
        "repo_root": "/testbed",
    }

    final_reward = episode_labels.get("reward_gated")
    out = []
    for i, s in enumerate(steps):
        hist = []
        start = max(0, i - context_steps)
        for j in range(start, i):
            hist.append(
                {
                    "action": str((steps[j].get("action") or "")).strip(),
                    "observation": clip(steps[j].get("observation") or ""),
                }
            )
        sl = step_labels[i] if i < len(step_labels) else {}
        shaping = 0.0
        if sl.get("is_file_view"):
            shaping += 0.1
        if sl.get("is_edit"):
            shaping += 0.2
        if sl.get("is_test_cmd"):
            shaping += 0.2
        if sl.get("dir_view_streak", 0) >= 2:
            shaping -= 0.05
        if sl.get("is_error"):
            shaping -= 0.2
        out.append(
            {
                "episode_id": episode_labels.get("instance_id") or episode_labels.get("repo") or "",
                "step_idx": s.get("step_idx"),
                "state": {
                    "problem_statement": problem,
                    "env": env_info,
                    "history": hist,
                    "steps_remaining": max(int(obj.get("max_steps") or 0) - int(s.get("step_count") or 0), 0),
                },
                "action": str(s.get("action") or ""),
                "step_label": sl,
                "reward": {
                    "terminal": final_reward if i == len(steps) - 1 else None,
                    "shaping": shaping,
                },
                "episode": episode_labels,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-path", required=True)
    ap.add_argument("--out-labeled-jsonl", required=True)
    ap.add_argument("--out-grpo-jsonl", default=None)
    ap.add_argument("--out-inject-jsonl", default=None)
    ap.add_argument("--context-steps", type=int, default=6)
    ap.add_argument("--obs-max-chars", type=int, default=3000)
    ap.add_argument("--require-edit-for-success", action="store_true", default=False)
    args = ap.parse_args()

    src = Path(args.traj_path)
    labeled_rows = []
    grpo_rows = []
    inject_rows = []
    for obj in _loads_jsonl(src):
        episode_labels, step_labels, inject_meta = _label_episode(
            obj, require_edit_for_success=bool(args.require_edit_for_success)
        )
        cto = obj.get("custom_test_outputs")
        if not isinstance(cto, dict):
            cto = {}
        cto["labels_0228"] = episode_labels
        cto["step_labels_0228"] = step_labels
        if inject_meta.get("inject_patch"):
            cto["inject_0228"] = {
                "source_patch_sha1": episode_labels.get("patch_sha1"),
                **inject_meta,
            }
        obj["custom_test_outputs"] = cto
        labeled_rows.append(obj)

        if args.out_grpo_jsonl:
            grpo_rows.extend(
                _build_grpo_samples(
                    obj,
                    episode_labels,
                    step_labels,
                    context_steps=int(args.context_steps),
                    obs_max_chars=int(args.obs_max_chars),
                )
            )
        if args.out_inject_jsonl and inject_meta.get("inject_patch"):
            inject_rows.append(
                {
                    "repo": episode_labels.get("repo"),
                    "instance_id": episode_labels.get("instance_id"),
                    "repo_language": episode_labels.get("repo_language"),
                    "source_patch_sha1": episode_labels.get("patch_sha1"),
                    "inject_patch": inject_meta.get("inject_patch"),
                    "inject_patch_sha1": inject_meta.get("inject_patch_sha1"),
                }
            )

    _dumps_jsonl(Path(args.out_labeled_jsonl), labeled_rows)
    if args.out_grpo_jsonl:
        _dumps_jsonl(Path(args.out_grpo_jsonl), grpo_rows)
    if args.out_inject_jsonl:
        _dumps_jsonl(Path(args.out_inject_jsonl), inject_rows)

    summary = {
        "traj_path": str(src),
        "episodes": len(labeled_rows),
        "grpo_samples": len(grpo_rows),
        "inject_candidates": len(inject_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
