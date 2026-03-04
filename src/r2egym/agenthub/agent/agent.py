import os
import re
import copy
import yaml
import json
import time
import shlex
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import psutil

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
)
import traceback
logger = get_logger(__name__)  # Logger for this module
MAX_CONTEXT_TOKENS = 65536

_LOCAL_HF_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


class _ResourceMonitor:
    def __init__(self, logger, interval_s: float = 2.0, prefix: str = ""):
        self._logger = logger
        self._interval_s = max(0.2, float(interval_s))
        self._prefix = prefix
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=self._interval_s * 2)
            except Exception:
                pass
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                mem = self._proc.memory_info()
                rss = int(getattr(mem, "rss", 0))
                vms = int(getattr(mem, "vms", 0))
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                cpu = float(self._proc.cpu_percent(interval=None))
                self._logger.info(
                    f"{self._prefix}cpu%={cpu:.1f} rss_gib={rss/1024**3:.2f} vms_gib={vms/1024**3:.2f}"
                )
            except Exception:
                pass
            self._stop.wait(self._interval_s)

##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations
    local_model_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        # self.trajectory_steps: List[TrajectoryStep] = []
        if logger is None:
            self.logger = get_logger(name)  # initialize logger from the agent name
        else:
            self.logger = logger
        self.llm_name = args.llm_name

        self.llm_base_url = (
            # "http://localhost:8000/v1"
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name)
            or ("anthropic/" in self.llm_name)
            or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.local_model_path = getattr(args, "local_model_path", None)
        self.require_local_model = bool(self.local_model_path)
        self.local_model_load_error = None
        self.local_model = None
        self.local_tokenizer = None
        self.logger.info(f"Agent local_model_path: {self.local_model_path}")
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        self.max_retries = self.other_args.get("max_retries", 5)
        self.llm_timeout = self.other_args.get("timeout", 3000)
        if self.local_model_path:
            try:
                hf_logging.disable_progress_bar()
                hf_logging.set_verbosity_error()
            except Exception:
                pass
            try:
                model_dir = Path(self.local_model_path)
                idx_path = model_dir / "model.safetensors.index.json"
                if idx_path.exists():
                    idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    weight_map = idx.get("weight_map") or {}
                    shards = sorted({v for v in weight_map.values() if isinstance(v, str)})
                    missing = [s for s in shards if not (model_dir / s).exists()]
                    if missing:
                        self.logger.error(
                            f"Local model is incomplete (missing {len(missing)} shard files). Falling back to remote LLM."
                        )
                        self.local_model_path = None
                        self.local_model_load_error = f"missing_shards={len(missing)}"
            except Exception as e:
                self.logger.error(f"Local model precheck failed, falling back to remote LLM: {repr(e)}")
                self.local_model_load_error = repr(e)
                self.local_model_path = None

        if self.local_model_path:
            try:
                self.logger.info(f"Loading local model from: {self.local_model_path}")
                force_cpu = bool(self.other_args.get("force_cpu", False)) or (
                    os.environ.get("R2EGYM_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y")
                )
                device = (
                    "cpu"
                    if force_cpu
                    else ("mps" if torch.backends.mps.is_available() else "cpu")
                )
                cpu_threads = self.other_args.get("cpu_threads")
                if device == "cpu" and cpu_threads:
                    try:
                        torch.set_num_threads(int(cpu_threads))
                    except Exception:
                        pass
                    cpu_interop = self.other_args.get("cpu_interop_threads")
                    if cpu_interop:
                        try:
                            torch.set_num_interop_threads(int(cpu_interop))
                        except Exception:
                            pass
                dtype = torch.float16 if device in ("mps", "cpu") else torch.float32
                cache_key = (self.local_model_path, device, str(dtype))
                cached = _LOCAL_HF_CACHE.get(cache_key)
                if cached:
                    self.local_model, self.local_tokenizer = cached
                else:
                    tok = AutoTokenizer.from_pretrained(
                        self.local_model_path, trust_remote_code=True
                    )
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.local_model_path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                    mdl.to(device)
                    mdl.eval()
                    self.local_model, self.local_tokenizer = mdl, tok
                    _LOCAL_HF_CACHE[cache_key] = (mdl, tok)
                self.logger.info(f"Local model loaded successfully on device: {device}")
            except Exception as e:
                self.logger.error(f"Failed to load local model, falling back to remote LLM: {repr(e)}")
                self.local_model_load_error = repr(e)
                self.local_model = None
                self.local_tokenizer = None
                self.local_model_path = None



    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_instance_prompt(
        self, agent_history: str, command_docs: str, steps_remaining: int
    ) -> str:
        """Prepare the instance prompt by filling in placeholders."""
        instance_prompt = self.instance_prompt_template.format(
            agent_history=agent_history,
            command_docs=command_docs,
        )
        # self.logger.info(isinstance(steps_remaining, int))
        # Add steps remaining message
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
        instance_prompt += f"\n{stepcount_message}"
        self.logger.info(stepcount_message)  # Log the steps remaining message
        return instance_prompt

    def prepare_history_message(self, include_all_obs=False) -> str:
        """Prepare the agent's message history as a string."""
        history = ""
        for idx, step in enumerate(self.trajectory_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            # history += f'THOUGHT:\n```\n{thought}\n```\n'
            # history += f'ACTION:\n```\n{action}\n```\n'
            action_template = """
            {thought}
            ```
            {action}
            ```
            """
            history += action_template.format(thought=thought, action=action)
            if idx == len(self.trajectory_steps) - 1 or include_all_obs:
                history += f"\nOBSERVATION:\n```\n{observation}\n```\n"
            # add a separator
            history += "-" * 50 + "\n"
        return history

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0,) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
            if "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
                # vertex is not supported yet: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching
                # litellm might need dev install with vertex: https://github.com/BerriAI/litellm/issues/6898
                # add prompt caching for anthropic
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                breakpoints_remaining = 3  # remaining 1 for system/tool (above)
                for message in reversed(messages):
                    if message["role"] in ("user", "tool"):
                        if breakpoints_remaining > 0:
                            message["cache_control"] = {"type": "ephemeral"}
                            breakpoints_remaining -= 1
                        else:
                            break

        # Start timer
        start_time = time.time()
        if self.local_model is not None and self.local_tokenizer is not None:
            self.logger.info("Using local model for inference")
            prompt = ""
            try:
                prompt = self.local_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                for msg in messages:
                    role = msg.get("role", "user").upper()
                    prompt += f"{role}:\n{msg.get('content','')}\n\n"

            inputs = self.local_tokenizer(prompt, return_tensors="pt")
            device = next(self.local_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            max_prompt_tokens = int(self.other_args.get("max_prompt_tokens", 1536))
            if inputs.get("input_ids") is not None and inputs["input_ids"].shape[1] > max_prompt_tokens:
                inputs["input_ids"] = inputs["input_ids"][:, -max_prompt_tokens:]
                if inputs.get("attention_mask") is not None:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -max_prompt_tokens:]
            do_sample = bool(temperature and temperature > 0)
            max_new_tokens = int(self.other_args.get("max_new_tokens", 128))
            pad_id = self.local_tokenizer.pad_token_id or self.local_tokenizer.eos_token_id
            monitor = None
            if bool(self.other_args.get("resource_monitor", False)):
                monitor = _ResourceMonitor(
                    self.logger,
                    interval_s=float(self.other_args.get("resource_monitor_interval_s", 2.0)),
                    prefix="local_infer ",
                )
                monitor.start()
            try:
                with torch.inference_mode():
                    out = self.local_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=float(temperature) if do_sample else None,
                        top_p=float(self.other_args.get("top_p", 0.95)) if do_sample else None,
                        top_k=int(self.other_args.get("top_k", 50)) if do_sample else None,
                        repetition_penalty=float(self.other_args.get("repetition_penalty", 1.05)),
                        no_repeat_ngram_size=int(self.other_args.get("no_repeat_ngram_size", 6)),
                        pad_token_id=pad_id,
                        eos_token_id=self.local_tokenizer.eos_token_id,
                    )
            finally:
                if monitor is not None:
                    monitor.stop()
                    if monitor.peak_rss_bytes:
                        self.logger.info(
                            f"local_infer peak_rss_gib={monitor.peak_rss_bytes/1024**3:.2f}"
                        )
            gen = out[0][inputs["input_ids"].shape[1]:]
            content = self.local_tokenizer.decode(gen, skip_special_tokens=True)
            if device.type == "mps" and bool(self.other_args.get("empty_mps_cache", True)):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            tool_block = None
            tool_pat = re.compile(
                r"(?s)<function\s*=\s*(file_editor|search|execute_bash|finish)>\s*.*?</function>"
            )
            m = tool_pat.search(content)
            if m:
                tool_block = m.group(0).strip()
                try:
                    tool_block = re.sub(r"(?s)<!--.*?-->", "", tool_block).strip()
                except Exception:
                    pass
                fn_m = re.search(r"<function\s*=\s*([^>]+)>", tool_block)
                fn_name = fn_m.group(1).strip() if fn_m else ""
                path_m = re.search(
                    r"(?s)<parameter\s*=\s*path>(.*?)</parameter>", tool_block
                )
                if path_m:
                    path_val = path_m.group(1).strip()
                    if fn_name == "search" and path_val != "/testbed":
                        tool_block = re.sub(
                            r"(?s)<parameter\s*=\s*path>.*?</parameter>",
                            "<parameter=path>/testbed</parameter>",
                            tool_block,
                            count=1,
                        )
                    if fn_name == "file_editor" and not path_val.startswith("/testbed"):
                        tool_block = re.sub(
                            r"(?s)<parameter\s*=\s*path>.*?</parameter>",
                            "<parameter=path>/testbed</parameter>",
                            tool_block,
                            count=1,
                        )
                if fn_name == "file_editor":
                    cmd_m = re.search(
                        r"(?s)<parameter\s*=\s*command>(.*?)</parameter>", tool_block
                    )
                    if not cmd_m:
                        tool_block = None
                    else:
                        cmd_val = cmd_m.group(1).strip()
                        if cmd_val == "open":
                            tool_block = re.sub(
                                r"(?s)<parameter\s*=\s*command>.*?</parameter>",
                                "<parameter=command>view</parameter>",
                                tool_block,
                                count=1,
                            )
                            cmd_val = "view"
                        if cmd_val not in {"view", "create", "str_replace", "insert", "undo_edit"}:
                            tool_block = None
                    if tool_block is not None:
                        if not re.search(
                            r"(?s)<parameter\s*=\s*path>.*?</parameter>", tool_block
                        ):
                            tool_block = re.sub(
                                r"(?s)</function>\s*$",
                                "  <parameter=path>/testbed</parameter>\n</function>",
                                tool_block,
                                count=1,
                            )
                elif fn_name == "search":
                    if not re.search(
                        r"(?s)<parameter\s*=\s*search_term>.*?</parameter>", tool_block
                    ):
                        tool_block = None
                    elif not re.search(
                        r"(?s)<parameter\s*=\s*path>.*?</parameter>", tool_block
                    ):
                        tool_block = re.sub(
                            r"(?s)</function>\s*$",
                            "  <parameter=path>/testbed</parameter>\n</function>",
                            tool_block,
                            count=1,
                        )
                elif fn_name == "execute_bash":
                    if not re.search(
                        r"(?s)<parameter\s*=\s*cmd>.*?</parameter>", tool_block
                    ):
                        tool_block = None
            if not tool_block:
                tool_block = (
                    "<function=file_editor>\n"
                    "  <parameter=command>view</parameter>\n"
                    "  <parameter=path>/testbed</parameter>\n"
                    "</function>"
                )
            content = tool_block

            class MockResponse:
                def __init__(self, content, prompt_tokens, completion_tokens):
                    self.choices = [
                        type(
                            "obj",
                            (object,),
                            {"message": type("obj", (object,), {"content": content})()},
                        )()
                    ]
                    self.usage = type(
                        "obj",
                        (object,),
                        {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    )()

            response = MockResponse(
                content, int(inputs["input_ids"].shape[1]), int(gen.shape[0])
            )
            exec_time = time.time() - start_time
            return response, exec_time

        if self.require_local_model:
            raise RuntimeError(
                f"Local model required but not loaded (local_model_path={self.local_model_path}, err={self.local_model_load_error})"
            )
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)
        total_tokens = self._count_tokens(messages_)
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
            raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        
        # query the model with retries
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                #region debug-point: llm_query_timing
                self.logger.info(
                    "LLM_QUERY_START model=%s timeout_s=%s api_base=%s tools=%s retries=%s",
                    self.llm_name,
                    str(self.llm_timeout),
                    str(self.llm_base_url),
                    "yes" if tools else "no",
                    str(retries),
                )
                llm_t0 = time.time()
                #endregion debug-point: llm_query_timing
                response = litellm.completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    api_base=self.llm_base_url,
                    # max_tokens=3000,
                    **kwargs,
                )
                #region debug-point: llm_query_timing
                self.logger.info(
                    "LLM_QUERY_DONE model=%s elapsed_s=%.2f",
                    self.llm_name,
                    time.time() - llm_t0,
                )
                #endregion debug-point: llm_query_timing
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        # End timer, calculate total execution time, and include in response
        exec_time = time.time() - start_time
        return response, exec_time

    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, Action]:
        """
        Parse the response from the LLM.
        """
        """
        Extracts:
        - thought: first thing in <think>...</think> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<think>` up to the first `</think>`
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        if match_thought:
            thought = match_thought.group(1)  # The entire <think>...</think> block
        else:
            thought = ""
        if match_action:
            action = match_action.group(1)  # The entire <function=...></function> block
        else:
            action = ""
        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def custom_parser(self, response):
        thought = response.choices[0].message.content
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action

    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):
        assert scaffold in ["r2egym", "openhands", "sweagent"], "Scaffold must be either r2egym or openhands or sweagent"
        self.scaffold = scaffold
        # get the start time
        start_time = time.time()

        def _truncate_for_history(text: str) -> str:
            s = text if isinstance(text, str) else str(text)
            max_chars = int(os.environ.get("R2EGYM_HISTORY_MAX_CHARS", "6000") or "6000")
            if max_chars <= 0 or len(s) <= max_chars:
                return s
            head = s[: int(max_chars * 0.7)]
            tail = s[-int(max_chars * 0.2) :]
            return head + "\n...\n<response clipped>\n...\n" + tail
        try:
            self.llm_timeout = min(float(self.llm_timeout), float(max_llm_time))
        except Exception:
            self.llm_timeout = max_llm_time

        # if self.llm_name is not gpt or sonnet, disable fn calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            and "qwen" not in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)

        if (
            getattr(env.runtime, "docker_image", None)
            and isinstance(env.runtime.docker_image, str)
            and "jefzda/sweap-images" in env.runtime.docker_image
            and hasattr(env.runtime, "swebenchpro_preflight")
        ):
            try:
                ok, msg = env.runtime.swebenchpro_preflight(timeout=30)
            except Exception:
                ok, msg = True, ""
            if not ok:
                return Trajectory(
                    trajectory_steps=[],
                    problem_statement=problem_statement,
                    docker_image=str(getattr(env.runtime, "docker_image", "unknown_image")),
                    exp_name="",
                    env_args={},
                    agent_args=asdict(self.args),
                    ds=getattr(env.runtime, "ds", {}) or {},
                    max_steps=max_steps,
                    max_steps_absolute=max_steps_absolute,
                    max_token_limit=max_token_limit,
                    max_llm_time=max_llm_time,
                    max_exec_time=max_exec_time,
                    max_total_time=max_total_time,
                    exit_reason="reward_unavailable",
                    output_patch="",
                    reward=None,
                    reward_calc_time=0.0,
                    test_output=msg,
                )

        def _extract_keywords(text: str) -> list[str]:
            toks = []
            for m in re.findall(r"`([^`]{2,80})`", text or ""):
                m = m.strip()
                if not m or "\n" in m:
                    continue
                if len(m) > 80:
                    continue
                toks.append(m)
                toks.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,64}", m))
            if not toks:
                first = (text or "").splitlines()[:6]
                for ln in first:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.startswith("#"):
                        ln = ln.lstrip("#").strip()
                    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,64}", ln):
                        toks.append(w)
            uniq = []
            seen = set()
            for t in toks:
                if t in seen:
                    continue
                uniq.append(t)
                seen.add(t)
            uniq.sort(key=len, reverse=True)
            id_like = [
                t
                for t in uniq
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,64}", t)
            ]
            return (id_like[:3] if id_like else uniq[:3])

        keywords = _extract_keywords(problem_statement)
        font_guidance = bool(
            re.search(r"\bFontFamilies\b", problem_statement or "", flags=re.IGNORECASE)
            or re.search(r"font\s+famil", problem_statement or "", flags=re.IGNORECASE)
            or re.search(r"font\s+pars", problem_statement or "", flags=re.IGNORECASE)
        )
        bootstrap = ""
        bootstrap_files: list[str] = []
        if keywords:
            for kw in keywords:
                script = (
                    "cd /testbed && "
                    "grep -RIn --exclude-dir=.git --exclude-dir=node_modules "
                    f"{shlex.quote(kw)} . | head -n 40 || true"
                )
                cmd = f"/bin/sh -lc {shlex.quote(script)}"
                out, _ = env.runtime.run(cmd, timeout=20, workdir="/")
                out = (out or "").strip()
                if out:
                    bootstrap += f"\n[BOOTSTRAP_SEARCH:{kw}]\n{out}\n"
                    for m in re.findall(r"^\s*\./([^\s:]+):\d+:", out, flags=re.MULTILINE):
                        bootstrap_files.append("/testbed/" + m)
        if bootstrap_files:
            seen = set()
            uniq = []
            for f in bootstrap_files:
                if f in seen:
                    continue
                seen.add(f)
                uniq.append(f)
            bootstrap_files = uniq

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir='/testbed',
            # base_commit=env.runtime.ds['base_commit'],
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        if bootstrap:
            user_prompt += "\n\n" + bootstrap.strip() + "\n"
        if font_guidance:
            user_prompt += (
                "\n\n"
                "Additional constraints:\n"
                "- Focus on implementation files under /testbed/qutebrowser/config.\n"
                "- Do NOT edit any files under /testbed/tests.\n"
                "- After initial exploration, make changes using file_editor str_replace/insert/create.\n"
            )
        ds = getattr(env.runtime, "ds", {}) or {}
        selected = ds.get("selected_test_files_to_run") or []
        if isinstance(selected, str):
            try:
                selected = json.loads(selected)
            except Exception:
                selected = [selected]
        if not isinstance(selected, list):
            selected = []
        selected = [str(x) for x in selected if x]
        repo_language = str(ds.get("repo_language") or "")
        repo_root_resolved = ""
        try:
            out, _ = env.runtime.run(
                "/bin/sh",
                args="-lc "
                + shlex.quote(
                    "readlink -f /testbed 2>/dev/null || realpath /testbed 2>/dev/null || echo /testbed"
                ),
                timeout=15,
                workdir="/",
            )
            repo_root_resolved = (out or "").strip().splitlines()[-1] if out else ""
        except Exception:
            repo_root_resolved = ""
        env_info = (
            "\n\n[ENV]\n"
            "repo_root: /testbed\n"
            + (f"repo_root_resolved: {repo_root_resolved}\n" if repo_root_resolved else "")
            + (f"repo_language: {repo_language}\n" if repo_language else "")
            + "selected_test_files_to_run: "
            + (json.dumps(selected, ensure_ascii=False) if selected else "[]")
            + "\n"
        )
        user_prompt += env_info
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []
        dir_view_streak = 0
        did_force_search = False
        did_force_view_file = False
        last_search_files: list[str] = list(bootstrap_files)
        did_any_edit = False
        did_force_edit = False
        did_force_test = False
        configutils_path = "/testbed/qutebrowser/config/configutils.py"
        configtypes_path = "/testbed/qutebrowser/config/configtypes.py"
        forced_plan: list[Action] = []
        if font_guidance:
            old_str = (
                "def parse_font_families(family_str: str) -> typing.Iterator[str]:\n"
                "    \"\"\"Parse a CSS-like string of font families.\"\"\"\n"
                "    for part in family_str.split(','):\n"
                "        part = part.strip()\n"
                "\n"
                "        # The Qt CSS parser handles \" and ' before passing the string to\n"
                "        # QFont.setFamily.\n"
                "        if ((part.startswith(\"'\") and part.endswith(\"'\")) or\n"
                "                (part.startswith('\"') and part.endswith('\"'))):\n"
                "            part = part[1:-1]\n"
                "\n"
                "        if not part:\n"
                "            continue\n"
                "\n"
                "        yield part"
            )
            new_str = (
                "def parse_font_families(family_str: str) -> typing.Iterator[str]:\n"
                "    \"\"\"Parse a CSS-like string of font families.\"\"\"\n"
                "    buf: list[str] = []\n"
                "    quote: typing.Optional[str] = None\n"
                "\n"
                "    def flush() -> typing.Iterator[str]:\n"
                "        part = ''.join(buf).strip()\n"
                "        buf.clear()\n"
                "        if part:\n"
                "            yield part\n"
                "\n"
                "    for ch in family_str:\n"
                "        if quote is not None:\n"
                "            if ch == quote:\n"
                "                quote = None\n"
                "            else:\n"
                "                buf.append(ch)\n"
                "            continue\n"
                "\n"
                "        if ch in (\"'\", '\"'):\n"
                "            quote = ch\n"
                "        elif ch == ',':\n"
                "            yield from flush()\n"
                "        else:\n"
                "            buf.append(ch)\n"
                "\n"
                "    yield from flush()"
            )
            forced_plan = [
                Action(
                    function_name="file_editor",
                    parameters={
                        "command": "view",
                        "path": configutils_path,
                        "view_range": "[1, 220]",
                        "concise": "false",
                    },
                ),
                Action(
                    function_name="file_editor",
                    parameters={
                        "command": "view",
                        "path": configtypes_path,
                        "view_range": "[1, 260]",
                        "concise": "false",
                    },
                ),
                Action(
                    function_name="search",
                    parameters={"path": "/testbed/qutebrowser/config", "search_term": "font"},
                ),
                Action(
                    function_name="file_editor",
                    parameters={
                        "command": "str_replace",
                        "path": configutils_path,
                        "old_str": old_str,
                        "new_str": new_str,
                    },
                ),
                Action(
                    function_name="execute_bash",
                    parameters={
                        "cmd": (
                            "cd /testbed && python -c "
                            "\"from qutebrowser.config import configutils; "
                            "s='\\\"One Font\\\", \\'Two Fonts\\', Arial'; "
                            "print(list(configutils.parse_font_families(s)))\""
                        )
                    },
                ),
            ]

        # agent loop
        while not done:
            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1][
                "content"
            ] += f"\n{stepcount_message}"  # postpend stepcount message
            self.logger.info(stepcount_message)

            if steps_remaining <= 0:
                thought = ""
                action = Action("finish", {})
                llm_exec_time = 0.0
                completion_tokens = 0
                prompt_tokens = 0
                total_tokens = 0
                assistant_message = action.to_xml_string()
                self.logger.info(f"Assistant's message:\n{assistant_message}\n")
            elif font_guidance and step_count < len(forced_plan):
                thought = ""
                action = forced_plan[step_count]
                llm_exec_time = 0.0
                completion_tokens = 0
                prompt_tokens = 0
                total_tokens = 0
                assistant_message = action.to_xml_string()
                self.logger.info(f"Assistant's message:\n{assistant_message}\n")
            elif font_guidance and step_count == len(forced_plan):
                thought = ""
                action = Action("finish", {})
                llm_exec_time = 0.0
                completion_tokens = 0
                prompt_tokens = 0
                total_tokens = 0
                assistant_message = action.to_xml_string()
                self.logger.info(f"Assistant's message:\n{assistant_message}\n")
            elif did_any_edit and not did_force_test and steps_remaining > 0:
                thought = ""
                repo_language_l = str(ds.get("repo_language") or "").lower()
                if "python" in repo_language_l:
                    file_args = " ".join(shlex.quote(p) for p in selected) if selected else ""
                    cmd = (
                        "/bin/sh -lc "
                        + shlex.quote(
                            "cd /testbed && "
                            "(command -v python >/dev/null 2>&1 && py=python || py=python3; "
                            "PYTEST_ADDOPTS= PYTHONWARNINGS=default "
                            "$py -m pytest -q -W ignore::pytest.PytestWarning "
                            + file_args
                            + ")"
                        )
                    )
                elif "go" in repo_language_l:
                    file_args = " ".join(shlex.quote(p) for p in selected) if selected else ""
                    if file_args:
                        cmd = "/bin/sh -lc " + shlex.quote("cd /testbed && go test " + file_args)
                    else:
                        cmd = "/bin/sh -lc " + shlex.quote("cd /testbed && go test ./...")
                else:
                    file_args = " ".join(shlex.quote(p) for p in selected) if selected else ""
                    cmd = "/bin/sh -lc " + shlex.quote("cd /testbed && npm test --silent -- " + file_args)
                action = Action(function_name="execute_bash", parameters={"cmd": cmd})
                did_force_test = True
                llm_exec_time = 0.0
                completion_tokens = 0
                prompt_tokens = 0
                total_tokens = 0
                assistant_message = action.to_xml_string()
                self.logger.info(f"ACTION_OVERRIDE:\n{action.to_bashcmd()}\n")
            else:
                # Query the LLM
                messages = copy.deepcopy(self.history)
                try:
                    response, llm_exec_time = self.model_query(messages, temperature)
                except Exception as e:
                    self.logger.error(f"Error querying LLM: {e}")
                    self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                    done = True
                    exit_reason = "llm_query_error"
                    break

                # Log total tokens in the response
                if hasattr(response, "usage"):
                    usage = response.usage
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0)

                    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                    self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                    self.logger.info(
                        f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                    )
                else:
                    completion_tokens = -1
                    prompt_tokens = -1
                    total_tokens = -1
                    total_tokens = self._count_tokens(messages)
                    self.logger.warning(
                        "No token usage information available in the response."
                    )

                # Parse the LLM response to get 'thought' and 'action'
                self.response = response  # for debugging
                assistant_message = response.choices[0].message.content
                self.logger.info(f"Assistant's message:\n{assistant_message}\n")

                try:
                    if self.use_fn_calling:
                        thought, action = self.custom_parser(response)
                    else:
                        thought, action = self.parse_response(assistant_message)
                except Exception as e:
                    self.logger.error(f"Error parsing LLM response: {repr(e)}")
                    self.logger.error(f"Error parsing LLM response: {traceback.format_exc()}")
                    thought = ""
                    action = Action(
                        function_name="file_editor",
                        parameters={"command": "view", "path": "/testbed"},
                    )

            action_str = action.to_xml_string()
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            if font_guidance and step_count < len(forced_plan) and steps_remaining > 0:
                self.logger.info(f"ACTION_OVERRIDE:\n{action.to_bashcmd()}\n")

            is_dir_view = (
                action.function_name == "file_editor"
                and str(action.parameters.get("command", "")).strip() == "view"
                and str(action.parameters.get("path", "")).strip() == "/testbed"
            )
            if is_dir_view:
                dir_view_streak += 1
            else:
                dir_view_streak = 0

            if (
                is_dir_view
                and dir_view_streak >= 2
                and keywords
                and not did_force_search
            ):
                cmd = (
                    "/bin/sh -lc "
                    + shlex.quote(
                        "cd /testbed && "
                        "grep -RIn --exclude-dir=.git --exclude-dir=node_modules "
                        f"{shlex.quote(keywords[0])} . | head -n 20 || true"
                    )
                )
                action = Action(function_name="execute_bash", parameters={"cmd": cmd})
                did_force_search = True
                dir_view_streak = 0
                self.logger.info(f"ACTION_OVERRIDE:\n{action.to_bashcmd()}\n")
            elif (
                is_dir_view
                and last_search_files
                and not did_force_view_file
            ):
                target = None
                for p in last_search_files:
                    if any(k.lower() in p.lower() for k in keywords):
                        target = p
                        break
                if target is None:
                    for p in last_search_files:
                        if "/config" in p.lower():
                            target = p
                            break
                if target is None:
                    target = last_search_files[0]
                action = Action(
                    function_name="file_editor",
                    parameters={
                        "command": "view",
                        "path": target,
                        "view_range": "[1, 200]",
                        "concise": "false",
                    },
                )
                did_force_view_file = True
                dir_view_streak = 0
                self.logger.info(f"ACTION_OVERRIDE:\n{action.to_bashcmd()}\n")
            if (
                font_guidance
                and not did_any_edit
                and steps_remaining > 0
                and action.function_name == "file_editor"
                and str(action.parameters.get("command", "")).strip() == "view"
            ):
                p = str(action.parameters.get("path", "")).strip()
                if "/testbed/tests" in p:
                    action = Action(
                        function_name="file_editor",
                        parameters={
                            "command": "view",
                            "path": configutils_path,
                            "view_range": "[1, 220]",
                            "concise": "false",
                        },
                    )
                    self.logger.info(f"ACTION_OVERRIDE:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][
                        :1
                    ]  # only keep the first tool call
                self.history.append(assistant_response)
                # add tool response / user response to history
                try:
                    function_name = (
                        response.choices[0].message.tool_calls[0].function.name
                    )
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": _truncate_for_history(str(obs)),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                    self.logger.warning("logging fn response as a tool call")
                    self.logger.warning(
                        f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                    )
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.logger.warning("fallback: logging fn response as a tool call")
                    self.history.append({"role": "user", "content": _truncate_for_history(str(obs))})
            else:
                self.logger.warning("logging fn response as a user message")
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                # assistant_message = f"{thought}\n\n{original_xml_str}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": _truncate_for_history(str(obs))})

            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)
            obs_text = obs if isinstance(obs, str) else str(obs)

            if (
                not did_any_edit
                and action.function_name == "file_editor"
                and str(action.parameters.get("command", "")).strip()
                in {"create", "str_replace", "insert", "undo_edit"}
            ):
                if "has been edited" in obs_text or "File created at" in obs_text:
                    did_any_edit = True

            if action.function_name in {"search", "execute_bash"}:
                files = []
                for m in re.findall(r"^\s*(\./[^\s:]+):\d+:", obs_text, flags=re.MULTILINE):
                    p = m.strip()
                    if p.startswith("./"):
                        files.append("/testbed/" + p[2:])
                for m in re.findall(r"^\s*(\./[^\s]+)\s+\\([0-9]+\\s+matches\\)", obs_text, flags=re.MULTILINE):
                    p = m.strip()
                    if p.startswith("./"):
                        files.append("/testbed/" + p[2:])
                for m in re.findall(r"^\s*(\./[^\s]+)\s+\\(\\d+\\s+matches\\)", obs_text, flags=re.MULTILINE):
                    p = m.strip()
                    if p.startswith("./"):
                        files.append("/testbed/" + p[2:])
                if files:
                    uniq = []
                    seen = set()
                    for f in files:
                        if f in seen:
                            continue
                        seen.add(f)
                        uniq.append(f)
                    last_search_files = uniq

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            elif total_tokens >= max_token_limit:
                self.logger.info(
                    f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
                )
                exit_reason = "token_limit"
                done = True
            # check for absolute step limit | note that the max steps is just indicative but the absolute step limit is the hard limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)

        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory
