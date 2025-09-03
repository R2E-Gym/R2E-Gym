import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset

# load gym dataset [R2E-Gym/R2E-Gym-Subset, R2E-Gym/R2E-Gym-Full, R2E-Gym/SWE-Bench-Verified, R2E-Gym/SWE-Bench-Lite]
ds = load_dataset("R2E-Gym/SWE-Bench-Verified")
split = 'test' # split of the dataset [train, test]

# load gym environment
env_index = 0 # index of the environment [0, len(ds)]
env_args = EnvArgs(ds = ds[split][env_index])
env = RepoEnv(env_args)

# load agent
agent_args = AgentArgs.from_yaml(Path('src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml'))
# define llm: ['claude-3-5-sonnet-20241022', 'gpt-4o', 'vllm/R2E-Gym/R2EGym-32B-Agent']
agent_args.llm_name = 'gemini-2.5-pro'
agent = Agent(name="EditingAgent", args=agent_args)

# run the agent (note: disable fn_calling for R2E-Gym agents)
output = agent.run(env, max_steps=50, use_fn_calling=False)