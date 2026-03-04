# editagent_script.py

import os
import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker
import traceback
import stat
import subprocess

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()

def _ensure_docker_host():
    if os.environ.get("DOCKER_HOST"):
        return
    for p in [
        os.path.expanduser("~/.colima/default/docker.sock"),
        os.path.expanduser("~/.docker/run/docker.sock"),
        "/var/run/docker.sock",
    ]:
        try:
            real = os.path.realpath(p)
            if os.path.exists(real) and stat.S_ISSOCK(os.stat(real).st_mode):
                os.environ["DOCKER_HOST"] = f"unix://{real}"
                return
        except Exception:
            pass
    try:
        with open(os.path.expanduser("~/.docker/config.json")) as f:
            cfg = json.load(f)
        if cfg.get("currentContext") != "colima":
            raise RuntimeError("skip_config_context")
    except Exception:
        cfg = None
    sock = os.path.expanduser("~/.colima/default/docker.sock")
    if os.path.exists(sock):
        os.environ["DOCKER_HOST"] = f"unix://{sock}"

    if os.environ.get("DOCKER_HOST"):
        return
    try:
        ctx = (
            subprocess.run(
                ["docker", "context", "show"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .stdout.strip()
        )
        if ctx:
            host = (
                subprocess.run(
                    ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}", ctx],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .stdout.strip()
            )
            if host:
                os.environ["DOCKER_HOST"] = host
    except Exception:
        pass


def _require_docker_access():
    try:
        docker.from_env(timeout=10).version()
        return
    except Exception as e:
        raise RuntimeError(
            "docker_unavailable: "
            + repr(e)
            + " DOCKER_HOST="
            + repr(os.environ.get("DOCKER_HOST"))
            + " (start Docker Desktop/colima and ensure a working docker.sock, or set DOCKER_HOST to the docker daemon endpoint)"
        )

def _normalize_ds_for_dockerruntime(ds: Dict[str, Any]) -> Dict[str, Any]:
    ds = dict(ds)

    if "docker_image" not in ds and "image_name" not in ds and ds.get("dockerhub_tag"):
        dockerhub_tag = str(ds["dockerhub_tag"])
        if "-" in dockerhub_tag:
            repo_prefix, tag_rest = dockerhub_tag.split("-", 1)
        else:
            repo_prefix, tag_rest = "unknown", dockerhub_tag

        repo_template = os.environ.get(
            "SWEBENCH_PRO_IMAGE_REPO_TEMPLATE", "jefzda/sweap-images"
        )
        if "{repo_prefix}" in repo_template:
            image_repo = repo_template.format(repo_prefix=repo_prefix)
            tag = tag_rest
        else:
            image_repo = repo_template
            tag = dockerhub_tag
        ds["docker_image"] = f"{image_repo}:{tag}"

    if "repo_name" not in ds and ds.get("repo"):
        ds["repo_name"] = str(ds["repo"]).split("/")[-1]

    if "problem_statement" not in ds and ds.get("problem"):
        ds["problem_statement"] = ds["problem"]

    if "parsed_commit_content" not in ds and ("patch" in ds or "test_patch" in ds):
        from r2egym.commit_models.parse_diff import CommitParser

        patch = ds.get("patch") or ""
        test_patch = ds.get("test_patch") or ""
        diff_text = "\n".join([patch.strip(), test_patch.strip()]).strip() + "\n"
        if diff_text.strip():
            base_commit = str(ds.get("base_commit") or "")
            parser = CommitParser()
            try:
                parsed = parser.parse_commit(
                    old_commit_hash=base_commit,
                    new_commit_hash=base_commit,
                    diff_text=diff_text,
                    commit_message="",
                    commit_date=datetime.now(),
                    repo_location=None,
                )
                ds["parsed_commit_content"] = parsed.model_dump_json()
            except Exception:
                if ds.get("parsed_commit") is not None:
                    ds["parsed_commit_content"] = ds["parsed_commit"]
                else:
                    from r2egym.commit_models.diff_classes import ParsedCommit

                    ds["parsed_commit_content"] = ParsedCommit(
                        file_diffs=[],
                        old_commit_hash=base_commit,
                        new_commit_hash=base_commit,
                        commit_message="",
                        commit_date=datetime.now(),
                    ).model_dump_json()

    return ds


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory

def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name: Optional[str] = "gpt-4o",
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    local_model_path: Optional[str] = None,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    if backend == "docker":
        _ensure_docker_host()
        _require_docker_access()
    if not llm_name:
        llm_name = "local"
    llm_base_url_env = os.environ.get("LLM_BASE_URL") or ""
    llm_provider_env = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if local_model_path is None and llm_base_url_env and "/" not in llm_name:
        provider = "openai"
        if llm_provider_env in {"anthropic", "openai"}:
            provider = llm_provider_env
        elif "/v1/messages" in llm_base_url_env or llm_base_url_env.rstrip("/").endswith(
            "/messages"
        ):
            provider = "anthropic"
        llm_name = f"{provider}/{llm_name}"

    ds = _normalize_ds_for_dockerruntime(ds)
    docker_image_for_log = ds.get("docker_image") or ds.get("image_name") or "unknown_image"

    logger = setup_logging(
        name=docker_image_for_log.replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{docker_image_for_log.replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {docker_image_for_log}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Local model path: {local_model_path}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    effective_use_fn_calling = bool(use_fn_calling)
    if local_model_path or (llm_name and "qwen" in llm_name.lower()):
        effective_use_fn_calling = False
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    try:
        env = RepoEnv(env_args, logger=logger, backend=backend)
    except Exception as e:
        logger.error(f"Error initializing environment for Docker image {docker_image_for_log}: {repr(e)}")
        try:
            tb = re.sub(r"\x1b\[[0-9;]*m|\r", "", traceback.format_exc())
        except Exception:
            tb = ""
        trajectory = Trajectory(
            trajectory_steps=[],
            problem_statement=str(ds.get("problem_statement") or ""),
            docker_image=docker_image_for_log,
            exp_name=exp_name,
            env_args=asdict(env_args),
            agent_args={},
            ds=ds,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_tokens,
            max_llm_time=7200,
            max_exec_time=90,
            max_total_time=50000,
            exit_reason="env_exception",
            output_patch="",
            reward=0.0,
            reward_calc_time=0.0,
            test_output=f"[ENV_EXCEPTION]\\n{repr(e)}\\n{tb}".strip() + "\n",
        )
        return trajectory.model_dump_json()
    # set agent args
    if effective_use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
        )
    else:
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
        )
    agent_args.llm_name = llm_name
    agent_args.local_model_path = local_model_path

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    trajectory = None
    try:
        trajectory = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=effective_use_fn_calling,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Error during agent run for Docker image {docker_image_for_log}: {repr(e)}")
        try:
            tb = re.sub(r"\x1b\[[0-9;]*m|\r", "", traceback.format_exc())
        except Exception:
            tb = ""
        try:
            env.close()
        except Exception:
            pass
        trajectory = Trajectory(
            trajectory_steps=[],
            problem_statement=str(ds.get("problem_statement") or ""),
            docker_image=docker_image_for_log,
            exp_name=exp_name,
            env_args=asdict(env_args),
            agent_args=asdict(agent_args),
            ds=ds,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_tokens,
            max_llm_time=7200,
            max_exec_time=90,
            max_total_time=50000,
            exit_reason="agent_exception",
            output_patch="",
            reward=0.0,
            reward_calc_time=0.0,
            test_output=f"[EXCEPTION]\\n{repr(e)}\\n{tb}".strip() + "\n",
        )
        return trajectory.model_dump_json()

    # also get the gt outputs
    reward_calc_time = time.time()
    try:
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    except Exception as e:
        reward = 0.0
        try:
            tb = re.sub(r"\x1b\[[0-9;]*m|\r", "", traceback.format_exc())
        except Exception:
            tb = ""
        test_output = f"[REWARD_EXCEPTION]\\n{repr(e)}\\n{tb}".strip() + "\n"
    reward_calc_time = time.time() - reward_calc_time
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    return trajectory.model_dump_json()


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name: Optional[str] = "gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
    local_model_path: Optional[str] = None,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    if not llm_name:
        llm_name = "local"
    # Load the dataset
    ds = load_dataset(dataset, split=split)
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # get selected idxs
    selected_idx = range(start_idx, start_idx + k)
    ds_selected = [ds[i] for i in selected_idx]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                local_model_path=local_model_path,
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for ds_entry in ds_selected
        }

        with open(jsonl_file, "a") as f:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")

def runagent_multiple_parquet(
    parquet_path: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx: int = 0,
    indices: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    max_steps: int = 40,
    num_restarts: int = 1,
    max_steps_absolute: int = 50,
    llm_name: Optional[str] = "gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "docker",
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    local_model_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    parallel: bool = False,
    dry_run: bool = False,
):
    if backend == "docker":
        _ensure_docker_host()
        _require_docker_access()
    if not llm_name:
        llm_name = "local"
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if indices:
        selected_idx = [int(x) for x in indices.split(",") if x.strip()]
    else:
        selected_idx = list(range(start_idx, start_idx + k))

    ds_selected: List[Dict[str, Any]] = []
    for i in selected_idx:
        ds_selected.append(_normalize_ds_for_dockerruntime(df.iloc[i].to_dict()))

    logger.info(
        f"Parquet: {parquet_path}, Num_total: {len(df)}, Selected: {len(ds_selected)}, Start Index: {start_idx}, k: {k}"
    )

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    existing_instance_ids = set()
    if use_existing and jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                try:
                    traj = Trajectory.load_from_model_dump_json(line)
                    if traj.ds and traj.ds.get("instance_id"):
                        existing_instance_ids.add(traj.ds["instance_id"])
                except Exception:
                    continue
        ds_selected = [x for x in ds_selected if x.get("instance_id") not in existing_instance_ids]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                for line in f:
                    try:
                        loadline = json.loads(line)
                        if loadline.get("reward") == 1 and loadline.get("ds", {}).get("instance_id"):
                            existing_instance_ids.add(loadline["ds"]["instance_id"])
                    except Exception:
                        continue
        ds_selected = [x for x in ds_selected if x.get("instance_id") not in existing_instance_ids]

    logger.info(f"Starting editagent on {len(ds_selected)} instances after filtering.")

    if dry_run:
        for x in ds_selected:
            logger.info(
                json.dumps(
                    {
                        "instance_id": x.get("instance_id"),
                        "repo": x.get("repo"),
                        "docker_image": x.get("docker_image"),
                        "dockerhub_tag": x.get("dockerhub_tag"),
                    },
                    ensure_ascii=False,
                )
            )
        return

    def _run_one(ds_entry: Dict[str, Any]) -> Optional[str]:
        return runagent(
            ds=ds_entry,
            exp_name=exp_name,
            max_steps=max_steps,
            num_restarts=num_restarts,
            max_steps_absolute=max_steps_absolute,
            llm_name=llm_name,
            temperature=temperature,
            use_fn_calling=use_fn_calling,
            backend=backend,
            max_reward_calc_time=max_reward_calc_time,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
            local_model_path=local_model_path,
        )

    if parallel and (max_workers or 0) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance = {
                executor.submit(_run_one, ds_entry): ds_entry.get("instance_id")
                for ds_entry in ds_selected
            }
            with open(jsonl_file, "a") as f:
                for future in concurrent.futures.as_completed(future_to_instance):
                    instance_id = future_to_instance[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with file_lock:
                                f.write(result + "\n")
                    except Exception as e:
                        logger.error(f"Exception for instance {instance_id}: {e}")
    else:
        with open(jsonl_file, "a") as f:
            for ds_entry in ds_selected:
                instance_id = ds_entry.get("instance_id")
                try:
                    result = _run_one(ds_entry)
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                except Exception as e:
                    logger.error(f"Exception for instance {instance_id}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} instances.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
            "runagent_multiple_parquet": runagent_multiple_parquet,
        }
    )
