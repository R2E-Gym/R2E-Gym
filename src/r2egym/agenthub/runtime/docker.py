import os, sys
import json
from time import sleep
import time
import uuid
import tempfile
import docker
from docker.models.containers import Container

from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.agenthub.runtime.base import (
    ExecutionEnvironment,
)
import base64
import subprocess
import datetime
import hashlib
import shutil

import docker
import tarfile
import io
import os
from r2egym.agenthub.utils.log import get_logger
import re
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
import concurrent.futures

from r2egym.agenthub.trajectory.swebench_utils import (
    make_test_spec,
    swebench_parse,
    TestSpec,
)
from r2egym.agenthub.utils.utils import get_logger
from r2egym.commit_models.diff_classes import ParsedCommit


DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status


##############################################################################
# Docker runtime
##############################################################################
class DockerRuntime(ExecutionEnvironment):
    """
    docker runtime is responsible for the interacting with the docker environment.
    In particular, it should allow for accomodating the features of the particualr docker envs used for r2e-edits
    - collect files
    - list files excluding test files etc
    """

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        **docker_kwargs,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, f"Dataset not provided for docker image: {docker_image}"

        # swebench specific setup
        self.ds = ds
        self.docker_image = (
            self.ds["docker_image"] if not docker_image else docker_image
        )
        self.swebench_verified = "swebench" in self.docker_image
        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            self.test_spec = make_test_spec(self.ds)

        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.repo_name = (
            self.ds["repo"] if self.swebench_verified else self.ds["repo_name"]
        )
        self.commit_json = (
            self.ds["parsed_commit"]
            if self.swebench_verified
            else self.ds["parsed_commit_content"]
        )
        self.commit = ParsedCommit(**json.loads(self.commit_json))
        self.docker_kwargs = docker_kwargs
        if logger is None:
            self.logger = get_logger(
                "DockerRuntime"
            )  # Pass the module name for clarity
        else:
            self.logger = logger
        self.client = docker.from_env()

        # Start the container
        self.container_name = self._get_container_name(self.docker_image)
        self.start_container(
            self.docker_image, command, self.container_name, **docker_kwargs
        )

        # Initialize the environment
        self.setup_env()
        self.logger.info("Docker environment initialized")
        self.logger.info("repo name: %s", self.repo_name)
        self.logger.info("Docker image: %s", self.docker_image)
        self.logger.info("Container ID: %s", self.container.id)

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        # Start or reuse a container
        try:
            containers = self.client.containers.list(
                all=True, filters={"name": ctr_name}
            )
            if containers:
                self.container = containers[0]
                if self.container.status != "running":
                    self.container.start()
            else:
                self.container = self.client.containers.run(
                    docker_image,
                    command,
                    name=ctr_name,
                    detach=True,
                    tty=True,
                    stdin_open=True,
                    # environment={"PATH": "/commands"},
                    **docker_kwargs,
                )
        except Exception as e:
            print("Container start error:", repr(e))
            self.stop_container()
            return

        # Prepare the subprocess for interaction
        startup_cmd = ["docker", "exec", "-i", ctr_name, "/bin/bash", "-l"]
        self.container_subprocess = subprocess.Popen(
            startup_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
        )

    def stop_container(self):
        try:
            if self.container:
                self.container.stop()
                self.container.remove()
            if self.container_subprocess:
                self.container_subprocess.terminate()  # Correct reference
        except Exception as e:
            print("Container stop error:", repr(e))

    def setup_env_swebench(self):
        try:
            # make the run_tests.sh executable
            self.run("chmod +x /run_tests.sh")

            # # move all skip files (if present) to /root
            # for skip_file in SKIP_FILES:
            #     self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
            self.alt_path = (
                "/"  # the run_test is in the "/" directory for swebench dockers
            )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")

            # install required packages TODO: check if working
            # self.run(
            #     "python -m pip install tree-sitter==0.20.4 tree_sitter_languages==1.10.2"
            # )
            self.run("python -m pip install chardet")
            # sudo apt-get install patchutils
            # self.run("apt-get update")
            # self.run("apt-get install -y patchutils")
        except Exception as e:
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def setup_env(self):
        if self.swebench_verified:
            return self.setup_env_swebench()

        try:
            # setup venv
            # modify the repo path to a common path
            # self.run(f"cp -r {self.repo_path} /workspace")

            # create a symlink from repo_path/.venv to /root/.venv
            self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")

            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self.run(
                f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )
            # print(self.run(f"ls -l {self.alt_path}/.local/bin"))

            # self.run(f"mv {self.repo_path} /workspace")
            # self.repo_path = "/workspace"

            # install required packages
            # self.run("uv pip install tree_sitter_languages") # remove since already installed in new dockers

            self.run("uv pip install chardet")

            self.run("find . -name '*.pyc' -delete")

            self.run("find . -name '__pycache__' -exec rm -rf {} +")

            # also delete pycache and pyc from /r2e_tests
            self.run("find /r2e_tests -name '*.pyc' -delete")
            self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")

            # move all skip files (if present) to /root
            for skip_file in SKIP_FILES_NEW:
                self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")

            # r2e_tests are in the / directory, move them to /root
            self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")

            # make a softlink for /root/r2e_tests (if present)
            self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")
            # self.run(f"ln -s /r2e_tests {self.repo_path}/r2e_tests")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def get_task_instruction(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        try:
            content = self.ds["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds["problem_statement"]

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (int).
        """
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Notice we do NOT set tty=True here
                future = executor.submit(
                    self.container.exec_run,
                    cmd=["/bin/sh", "-c", command],
                    # cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    stdout=True,
                    stderr=True,
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Retrieve output and exit code
            output = exec_result.output.decode("utf-8", errors="replace")
            error_code = exec_result.exit_code

            if error_code == 124:
                self.logger.error(f"Internal Timeout: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nError Message: {output}"
                )
                return output, f"Error: Exit code {error_code}"

            # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        ## timeout
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout: {timeout}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Set demux=True to get separate stdout and stderr streams
                future = executor.submit(
                    self.container.exec_run,
                    cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    demux=True,  # This is the key change
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            output_data, error_data = exec_result.output
            error_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = (
                output_data.decode("utf-8", errors="replace") if output_data else ""
            )
            stderr = error_data.decode("utf-8", errors="replace") if error_data else ""

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nStdout Message: {stdout}, \nError Message: {stderr}"
                )
                return stdout, stderr, f"Error: Exit code {error_code}"

            return stdout, stderr, str(error_code)
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"

    def copy_to_container(self, src_path: str, dest_path: str):
        """
        Copies a file or directory from the host to the Docker container.

        Args:
            src_path: Path to the file or directory on the host.
            dest_path: Destination path inside the container.
        """
        # Create a tar archive of the source file/directory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        # Use Docker API to put the archive into the container
        self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())

    @DeprecationWarning  # TODO: remove dependency on this method with new dockers
    def read_file(self, rel_file_path: str) -> str:
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output

    def run_tests(self) -> tuple[str, str]:
        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=300)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def demux_run_tests(self) -> tuple[str, str, str]:
        stdout, stderr, error_code = self.demux_run(
            f"bash {self.alt_path}/run_tests.sh"
        )
        # Remove ANSI escape codes and \r characters
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
        return stdout, stderr, error_code

    def checkout(self, commit_hash: str) -> tuple[str, str]:
        output, error_code = self.run(f"git checkout {commit_hash}")
        return output, error_code

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository.
        """
        # git add -A && git diff --cached
        # self.run("git add -A")
        output, _ = self.run("git add -A && git diff --cached")
        # output, _ = self.run("git diff")
        return output

    def create_file(self, file_path: str, content: str) -> tuple[str, str]:
        # create a local file with the content
        uuid_ = uuid.uuid4()
        file_path_ = f"{file_path}_{uuid_}"
        file_path__ = os.path.join("/tmp", file_path_)
        with open(file_path__, "w") as f:
            f.write(content)
        # copy the file to the container
        self.copy_to_container(file_path__, f"/testbed/{file_path_}")
        self.run(f"mv /testbed/{file_path_} /{file_path}")

    def apply_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply --whitespace=fix /{patch_path}")
        return output, error_code

    def reverse_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        patch_path = f"{self.container_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply -R /{patch_path}")
        return output, error_code

    def get_logs_eval(
        self, test_spec: TestSpec, content: str
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     self.logger.error("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        content = content.split(test_cmd)[-1]
        self.logger.info(f"using swebench log_parser for repo: {repo}")
        return log_parser(content, test_spec), True

    def parse_logs(self, log_output: str) -> dict:
        if self.swebench_verified:
            parsed_output, patch_apply_success = self.get_logs_eval(
                self.test_spec, log_output
            )
            return parsed_output
        else:
            return parse_log_fn(f"{self.repo_name}")(log_output)

    def _calculate_reward_swebench(self, get_test_output=False) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        out, _ = self.run(
            "/run_tests.sh", 300
        )  # run the tests after applying the patch
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, out
        return int(success)

    def _calculate_reward_r2e(self, get_test_output=False) -> float:
        # calculate reward based for r2e-edit dockers
        output, error_code = self.run_tests()
        # print(output)x
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception as e:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        # Compare
        if len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        if get_test_output:
            return reward, output
        return reward

    def _calculate_reward(self, get_test_output=False) -> float:
        if self.swebench_verified:
            return self._calculate_reward_swebench(get_test_output=get_test_output)
        else:
            return self._calculate_reward_r2e(get_test_output=get_test_output)

    def reset(self):
        self.stop_container()
        self.start_container(self.docker_image, self.command, self.container_name, **self.docker_kwargs)

    def close(self):
        self.stop_container()
        self.client.close()

    def run_swebv_regression(
        self, run_tests_regression: str | None = None, timeout: int = 300
    ) -> dict[str, str]:
        # run the regression tests for swebench verified dockers
        # copy the 'run_tests_regression' thing from ds into the container at /run_tests_regression.sh
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(run_tests_regression)
            f.flush()
            self.copy_to_container(f.name, "/run_tests_regression.sh")
        # make the script executable
        self.run("chmod +x /run_tests_regression.sh")

        # run the regression tests
        output, error_code = self.run("/run_tests_regression.sh", timeout=timeout)
        return output
        # return swebench_parse(self.ds, output)

    def start_new_branch(self, branch_name: str = "exp") -> tuple[str, str]:
        # ## save current branch-name
        # output, error_code = self.run("git branch --show-current")
        # self.current_branch = output.strip()
        # # new branch
        # output, error_code = self.run(f"git checkout -b {branch_name}")
        # # save commit hash

        output, error_code = self.run(
            "git config --global user.email 'you@example.com'"
        )
        output, error_code = self.run("git config --global user.name 'Your Name'")
        output, error_code = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output, error_code

    def commit_after_step(self, step_idx: int) -> tuple[str, str]:
        # commit
        output, error_code = self.run("git add .")
        output, error_code = self.run(f"git commit -m '{step_idx}'")
        return output, error_code

    def undo_last_commit(self) -> tuple[str, str]:
        # undo last commit
        output, error_code = self.run("git reset --hard HEAD~1")
        return output, error_code

    def get_current_commit_hash(self) -> str:
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> tuple[str, str]:
        # soft reset to saved commit
        output, error_code = self.run(f"git reset --soft {self.current_commit}")

        # # checkout to saved branch
        # output, error_code = self.run(f"git checkout {self.current_branch}")

        return output, error_code
