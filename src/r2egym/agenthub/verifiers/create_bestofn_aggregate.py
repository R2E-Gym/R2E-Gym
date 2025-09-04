import glob
import json
from collections import defaultdict
import os
import fire
import tqdm
from r2egym.agenthub.trajectory.trajectory import Trajectory
import tqdm


def run_ef_verifier(sub_trajs: list[Trajectory], dataset_name: str) -> Trajectory:
    """
    Select the trajectory with the highest verifier probability.
    """
    return max(sub_trajs, key=lambda x: x.verifier_prob)


def run_eb_verifier(sub_trajs: list[Trajectory], dataset_name: str) -> Trajectory:
    """
    Select the trajectory with the highest execution-based verifier score.
    First, only keep the trajectories with the highest regression score.
    Next, among the trajectories with the highest regression score, select the one with the highest reproduction test score.
    """
    sub_trajs = [
        traj
        for traj in sub_trajs
        if traj.regression_pass_count
        == max(traj.regression_pass_count for traj in sub_trajs)
    ]
    return max(sub_trajs, key=lambda x: x.reproduction_test_score)


def run_hybrid_verifier(sub_trajs: list[Trajectory], dataset_name: str) -> Trajectory:
    """
    First, keep top-n trajectories with the highest verifier probability.
    Next, among the top-n trajectories, keep the trajectory with the highest regression score.
    Next, among the trajectories with the highest regression score, keep the ones with the highest reproduction test score.
    Finally, select the trajectory with the highest verifier probability.
    """

    if dataset_name == "sweverify":
        # filter by finished score, if none are finished, use all trajectories
        hybrid_candidates = [traj for traj in sub_trajs if traj.finished_score == 1]
        if not hybrid_candidates:
            # Fallback to all trajectories if none are finished
            hybrid_candidates = sub_trajs

        # just rerank, no filter
        n = len(hybrid_candidates) 
        hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x.verifier_prob, reverse=True)[:n]
        
        # filter by regression pass count
        max_regression_hybrid = max(traj.regression_pass_count for traj in hybrid_candidates)
        hybrid_candidates = [traj for traj in hybrid_candidates if traj.regression_pass_count == max_regression_hybrid]
        
        # filter by reproduction test score
        max_reproduction_hybrid = max(traj.reproduction_test_score for traj in hybrid_candidates)
        hybrid_candidates = [traj for traj in hybrid_candidates if traj.reproduction_test_score == max_reproduction_hybrid]

        # filter by verifier probability
        last_candidates = [traj for traj in hybrid_candidates if traj.verifier_prob > 0.01]
        if not last_candidates:
            last_candidates = hybrid_candidates

        hybrid_selected = max(last_candidates, key=lambda x: len(x.history))

    
    elif dataset_name == "swelite":
        # filter by finished score, if none are finished, use all trajectories
        hybrid_candidates = [traj for traj in sub_trajs if traj.finished_score == 1]
        if not hybrid_candidates:
            # Fallback to all trajectories if none are finished
            hybrid_candidates = sub_trajs

        # just rerank, no filter
        n = len(hybrid_candidates) 
        hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x.verifier_prob, reverse=True)[:n]

        # filter by verifier probability
        last_candidates = [traj for traj in hybrid_candidates if traj.verifier_prob > 0.01]
        if not last_candidates:
            last_candidates = hybrid_candidates

        # select the trajectory with the minimum history length
        hybrid_selected = min(last_candidates, key=lambda x: len(x.history))
        
    return hybrid_selected


def run(
    traj_file_glob: str,
    verifier_mode: str,
    output_json_path: str,
    dataset_name: str,
    exp_name: str,
    save_history_path: str,
):
    assert verifier_mode in ["ef", "eb", "hybrid"]
    print(f"Dataset name: {dataset_name}")
    assert dataset_name in ["sweverify", "swelite"]
    verifier_fn_dict = {
        "ef": run_ef_verifier,
        "eb": run_eb_verifier,
        "hybrid": run_hybrid_verifier,
    }
    verifier_fn = verifier_fn_dict[verifier_mode]

    traj_files = glob.glob(traj_file_glob)
    all_trajs_by_docker: dict[str, list[Trajectory]] = defaultdict(list)

    for traj_file in tqdm.tqdm(traj_files, desc="Loading trajectories"):
        with open(traj_file, "r") as f:
            for line in f:
                traj = Trajectory.model_validate_json(line)
                all_trajs_by_docker[traj.docker_image].append(traj)

    reward = 0
    submission = []
    for docker_image, sub_trajs in tqdm.tqdm(all_trajs_by_docker.items(), desc="Selecting trajectories"):
        selected_traj = verifier_fn(sub_trajs, dataset_name)
        
        # save the selected trajectory's history to a json file under save_history_path
        os.makedirs(f"{save_history_path}/{exp_name}", exist_ok=True)
        file_name = f"{save_history_path}/{exp_name}/{selected_traj.ds['instance_id']}.json"
        with open(file_name, "w") as f:
            json.dump(selected_traj.history, f)
        
        selected_traj.docker_image = docker_image
        selected_traj.exp_name = exp_name
        reward += selected_traj.reward
        submission.append(selected_traj.create_swebench_submission())
    print(f"Total reward: {reward}")
    with open(output_json_path, "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    fire.Fire(run)
