import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from fire import Fire

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.utils.log import get_logger


logger = get_logger(__name__)


def _extract_changed_files_from_unified_diff(patch_text: str) -> List[str]:
    files = set()
    for ln in str(patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            files.add(ln[6:].strip())
        elif ln.startswith("--- a/"):
            files.add(ln[6:].strip())
    return sorted(files)


def _get_ground_truth_changed_files(env: RepoEnv) -> List[str]:
    try:
        # Prefer env.get_stats if it contains gt patch
        st = env.get_stats()
        gt_patch = st.get("gt_patch") or st.get("ground_truth_patch")
        if gt_patch:
            return _extract_changed_files_from_unified_diff(gt_patch)
    except Exception:
        pass
    try:
        # Some environments expose convenience method
        gt_commit = getattr(env, "get_gt_commit", None)
        if callable(gt_commit):
            commit_info = gt_commit()
            gt_patch = commit_info.get("patch") or commit_info.get("diff")
            if gt_patch:
                return _extract_changed_files_from_unified_diff(gt_patch)
    except Exception:
        pass
    return []


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa and sb:
        return 0.0
    if sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union > 0 else 0.0


def _apply_patch(env: RepoEnv, patch_text: str) -> Tuple[bool, Optional[str]]:
    try:
        rt = env.runtime
        # Many runtimes provide apply_patch - use it if available
        ap = getattr(rt, "apply_patch", None)
        if callable(ap):
            ap(patch_text)
            return True, None
        # Fallback: attempt to run a generic patch apply command
        cmd = "git apply -p0 --reject --whitespace=fix -"
        out = rt.run(cmd, input=patch_text)
        if out and "error" in str(out).lower():
            return False, str(out)
        return True, None
    except Exception as e:
        return False, str(e)


def _calculate_reward(env: RepoEnv, timeout_s: int = 300) -> Tuple[int, Dict[str, Any]]:
    try:
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=timeout_s)
        return int(reward), {"test_output": test_output}
    except Exception as e:
        return 0, {"error": str(e)}


def _find_index_by_docker(ds: List[Dict[str, Any]], docker_image: str) -> Optional[int]:
    for i, entry in enumerate(ds):
        if str(entry.get("docker_image", "")) == str(docker_image):
            return i
    return None


def eval_patches(
    patch_jsonl: str,
    dataset: str = "R2E-Gym/R2E-Gym-Lite",
    split: str = "train",
    out_jsonl: str = "./adapter_eval_results.jsonl",
    timeout_s: int = 300,
):
    """
    Evaluate generated patches against R2E-Gym environments and produce accuracy and closeness scores.

    Input JSONL line format (one of the identifiers must be provided):
    {
      "dataset": "R2E-Gym/R2E-Gym-Lite",    # optional, falls back to CLI arg
      "split": "train",                      # optional, falls back to CLI arg
      "env_index": 100,                      # preferred identifier
      "docker_image": "namanjain12/xxx:tag", # alternative identifier
      "patch": "<unified-diff-patch-text>"
    }

    Output JSONL lines follow an eval_sampleresults-like structure:
    {
      "stage": "r2egym_eval_patch",
      "dataset": "...",
      "split": "...",
      "env_index": 100,
      "docker_image": "...",
      "passed": true/false,
      "accuracy": 0/1,
      "closeness": 0.0..1.0,
      "changed_files": [...],
      "gt_changed_files": [...],
      "reward": 0/1,
      "fail_detail": {"error": "..."}    # present if any error
    }
    """
    patch_path = Path(patch_jsonl)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset, split=split)

    with patch_path.open("r", encoding="utf-8") as fin, out_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            ds_name = str(rec.get("dataset") or dataset)
            ds_split = str(rec.get("split") or split)
            env_index = rec.get("env_index")
            docker_image = rec.get("docker_image")
            patch_text = rec.get("patch") or ""

            # resolve environment index
            idx = None
            if isinstance(env_index, int):
                idx = env_index
            elif docker_image:
                idx = _find_index_by_docker(ds, docker_image)

            if idx is None or idx < 0 or idx >= len(ds):
                out_obj = {
                    "stage": "r2egym_eval_patch",
                    "dataset": ds_name,
                    "split": ds_split,
                    "env_index": env_index,
                    "docker_image": docker_image,
                    "passed": False,
                    "accuracy": 0,
                    "closeness": 0.0,
                    "reward": 0,
                    "fail_detail": {"error": "invalid_env_index_or_docker_image"},
                }
                fout.write(json.dumps(out_obj) + "\n")
                continue

            ds_entry = ds[idx]
            env_args = EnvArgs(ds=ds_entry)
            env = RepoEnv(env_args)

            # apply patch
            applied, apply_err = _apply_patch(env, patch_text)
            if not applied:
                gt_files = _get_ground_truth_changed_files(env)
                env.close()
                out_obj = {
                    "stage": "r2egym_eval_patch",
                    "dataset": ds_name,
                    "split": ds_split,
                    "env_index": idx,
                    "docker_image": ds_entry.get("docker_image"),
                    "passed": False,
                    "accuracy": 0,
                    "closeness": 0.0,
                    "changed_files": _extract_changed_files_from_unified_diff(patch_text),
                    "gt_changed_files": gt_files,
                    "reward": 0,
                    "fail_detail": {"error": apply_err or "patch_apply_failed"},
                }
                fout.write(json.dumps(out_obj) + "\n")
                continue

            # calculate reward (execute unit tests)
            reward, test_info = _calculate_reward(env, timeout_s=timeout_s)
            gt_files = _get_ground_truth_changed_files(env)
            ch_files = _extract_changed_files_from_unified_diff(patch_text)
            closeness = _jaccard(ch_files, gt_files)
            env.close()

            passed = bool(reward == 1)
            out_obj = {
                "stage": "r2egym_eval_patch",
                "dataset": ds_name,
                "split": ds_split,
                "env_index": idx,
                "docker_image": ds_entry.get("docker_image"),
                "passed": passed,
                "accuracy": 1 if passed else 0,
                "closeness": float(closeness),
                "changed_files": ch_files,
                "gt_changed_files": gt_files,
                "reward": reward,
            }
            if "error" in test_info:
                out_obj["fail_detail"] = {"error": test_info["error"]}
            fout.write(json.dumps(out_obj) + "\n")


if __name__ == "__main__":
    Fire(eval_patches)
