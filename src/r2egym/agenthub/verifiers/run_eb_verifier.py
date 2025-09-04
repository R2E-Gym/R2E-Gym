import glob

import fire

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.verifiers.run_regression_tests import add_regression_output
from r2egym.agenthub.verifiers.run_reproduction_tests import add_reproduction_tests
import tqdm

def process_trajectories_to_verifier_format(
    traj_file_glob: str,
    max_workers: int = 42,
):
    traj_files = glob.glob(traj_file_glob)
    for traj_file in tqdm.tqdm(traj_files):
        print(f"Processing {traj_file}")
        trajectories: list[Trajectory] = []
        with open(traj_file, "r") as f:
            for line in f:
                trajectories.append(Trajectory.model_validate_json(line))

        # Separate already processed trajectories from those needing processing
        needs_processing = []
        already_processed_indices = []
        
        for i, traj in enumerate(trajectories):
            # Check if trajectory already has regression test output or reproduction test scores
            if (traj.regression_test_output is not None or
                len(traj.reproduction_test_scores) > 0):
                already_processed_indices.append(i)
            else:
                needs_processing.append(traj)
        
        print(f"Processing {traj_file}: {len(already_processed_indices)} already processed, {len(needs_processing)} need processing")
        
        # Only process trajectories that need processing
        if needs_processing:
            processed_trajectories = add_regression_output(needs_processing, max_workers=max_workers)
            processed_trajectories = add_reproduction_tests(processed_trajectories, max_workers=max_workers)
            
            # Combine already processed with newly processed trajectories
            # Maintain original order
            final_trajectories = []
            processed_idx = 0
            
            for i, original_traj in enumerate(trajectories):
                if i in already_processed_indices:
                    # Keep the original trajectory (already processed)
                    final_trajectories.append(original_traj)
                else:
                    # Use the newly processed trajectory
                    final_trajectories.append(processed_trajectories[processed_idx])
                    processed_idx += 1
        else:
            # All trajectories were already processed
            final_trajectories = trajectories

        with open(traj_file, "w") as f:
            for traj in final_trajectories:
                f.write(traj.model_dump_json() + "\n")


if __name__ == "__main__":
    fire.Fire(process_trajectories_to_verifier_format)
