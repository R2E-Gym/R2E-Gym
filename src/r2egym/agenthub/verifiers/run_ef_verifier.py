import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import litellm
import numpy as np
from tqdm import tqdm

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.verifiers.prepare_ef_verifier_input import traj2verifier_data
import tqdm
MAX_RETRIES = 5


def compute_finished_score(trajectory: Trajectory) -> int:
    """
    Compute finished_score by checking if last assistant message contains '<function=finish>'.
    Returns 1 if finished properly, 0 otherwise.
    Note: All trajectories (including those with finished_score=0) will now be processed
    through the verifier, but finished_score is still computed and recorded.
    """
    try:
        # Get the history from trajectory
        if not trajectory.history:
            return 0
        
        # Find the last assistant message
        last_assistant_msg = None
        for msg in reversed(trajectory.history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg
                break
        
        if last_assistant_msg is None:
            return 0
            
        # Check if it contains '<function=finish>'
        content = last_assistant_msg.get('content', '')
        if '<function=finish>' in content:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error computing finished_score for {trajectory.docker_image}: {e}")
        return 0


def run_model(arg) -> float:
    message_list: list[dict]

    message_list, verifier_model_name = arg

    retries = 0

    # condense messages
    condensed_user_msg = message_list[1][
        "content"
    ]  # condense(input_str=message_list[1]['content'], max_tokens = 28000)
    message_list = [
        {"role": "system", "content": message_list[0]["content"]},
        {"role": "user", "content": condensed_user_msg},
    ]
    # query the model with retries
    while retries < MAX_RETRIES:
        try:
            response = litellm.completion(
                model=verifier_model_name,
                tools=[],
                messages=message_list,
                n=1,
                function_call=None,
                tool_choice="none",
                timeout=120,
                api_key=None,
                temperature=0,
                api_base=f"http://localhost:8083/v1",
                logprobs=True,
                top_logprobs=20,
                # extra_body={
                #     "guided_choice": ["YES", "NO"]
                # },
            )
            break
        except Exception as e:
            print(f"LLM query failed: {e}")
            retries += 1
            if retries >= MAX_RETRIES:
                return 0.0

    target_token = 4 # <YES> or <NO>
    all_logits = [
        {
            lp.token: lp.logprob
            for lp in response.choices[0].logprobs.content[target_token].top_logprobs
        }
    ]

    # compute probability of YES
    k = 0
    p_yes = all_logits[k].get("YES", -10000)
    p_no = all_logits[k].get("NO", -10000)
    yes_prob = (np.exp(p_yes)) / (np.exp(p_yes) + np.exp(p_no))
    return yes_prob


def process_trajectories_to_verifier_format(
    traj_file_glob: str,
    verifier_model_name: str,
    max_workers: int = 40,
    max_llm_workers: int = 8,
    max_tokens: int = 65536,
):
    traj_files = glob.glob(traj_file_glob)
    for traj_file in tqdm.tqdm(traj_files, desc="Processing trajectories"):
        print(f"Processing {traj_file}")
        trajectories: list[Trajectory] = []
        with open(traj_file, "r") as f:
            for line in f:
                trajectories.append(Trajectory.model_validate_json(line))

        # Separate already processed trajectories from unprocessed ones
        unprocessed_trajectories = []
        unprocessed_indices = []
        
        for idx, traj in enumerate(trajectories):
            # Check if trajectory is already processed (has both finished_score and verifier_prob)
            if traj.finished_score is not None and traj.verifier_prob is not None:
                print(f"Skipping already processed trajectory {idx} for {traj.docker_image}")
                continue
            else:
                unprocessed_trajectories.append(traj)
                unprocessed_indices.append(idx)

        if not unprocessed_trajectories:
            print(f"All trajectories in {traj_file} are already processed, skipping file")
            continue

        # Calculate finish scores for unprocessed trajectories
        print(f"Computing finish scores for {len(unprocessed_trajectories)} unprocessed trajectories...")
        for traj in unprocessed_trajectories:
            traj.finished_score = compute_finished_score(traj)

        # Prepare messages for all unprocessed trajectories (regardless of finished_score)
        messages = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for traj in unprocessed_trajectories:
                # Process all unprocessed trajectories, not just those with finished_score == 1
                futures.append(executor.submit(traj2verifier_data, traj.model_dump(), max_tokens=max_tokens))
            
            for future in as_completed(futures):
                data_entry, success = future.result()
                messages.append(data_entry)

        # Run model for all unprocessed trajectories
        yes_probs = []
        if messages:  # Only if there are messages to process
            with ProcessPoolExecutor(max_workers=max_llm_workers) as executor:
                futures = [
                    executor.submit(run_model, (message, verifier_model_name))
                    for message in messages
                ]
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running verifier"):
                    yes_prob = future.result()
                    yes_probs.append(yes_prob)

        # Assign verifier probabilities to unprocessed trajectories
        for idx, traj in enumerate(unprocessed_trajectories):
            if idx < len(yes_probs):
                traj.verifier_prob = yes_probs[idx]
            else:
                # Fallback in case of mismatch
                traj.verifier_prob = 0.0

        # Save trajectories with both finished_score and verifier_prob
        with open(traj_file, "w") as f:
            for traj in trajectories:
                f.write(traj.model_dump_json() + "\n")


if __name__ == "__main__":
    fire.Fire(process_trajectories_to_verifier_format)
