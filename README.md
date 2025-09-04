<h1 align="center"> Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization </h1>

<p align="center">
  <a href="https://sherdencooper.github.io/" style="text-decoration: none;">Jiahao Yu<sup>*,1</sup></a>, 
  <a href="https://sites.northwestern.edu/zelei/about/" style="text-decoration: none;">Zelei Cheng<sup>*,1</sup></a>,
  <a href="https://nuwuxian.github.io/" style="text-decoration: none;">Xian Wu<sup>2</sup></a>,
  <a href="http://xinyuxing.org/" style="text-decoration: none;">Xinyu Xing<sup>1</sup></a>,
</p>

<p align="center">
  <sup>1</sup>Northwestern University, <sup>2</sup>Meta </br>
  <sub><sup>*</sup>Equal contribution</sub>
</p>

<p align="center">
  <img src="./assets/42_logo.png" alt="Logo" width="20%">
</p>

---

## Introduction

LLM-powered software engineering agents are rapidly advancing, showing great promise in automating complex coding tasks. However, as these agents tackle real-world problems, a core challenge has emerged: while we can generate many potential solutions to a problem (a strategy known as test-time scaling), the performance gains are often limited if the solutions are too similar to one another, and this is especially obvious for offline learning which depends on the given offline dataset.

This is because modern alignment techniques, such as Direct Preference Optimization (DPO), tend to inadvertently reduce the diversity of the model's outputs. This "diversity collapse" means the model becomes overconfident in a narrow range of solutions, making it less likely to find the correct one for complex problems. If you ask an agent to generate ten solutions and it gives you the same idea repackaged ten times, you haven't really explored the solution space.

To address this, we introduce **EntroPO**, an entropy-enhanced preference optimization method tailored for multi-turn, tool-using coding agents. EntroPO is designed to preserve policy diversity during fine-tuning, unlocking significant performance gains from test-time scaling.

We would like to thank the authors of the R2E framework for their great work and open source. Our project is built upon their flexible and easy-to-use framework. 
This framework is highly useful and recommended for anyone who wants to work on software engineering agents with machine learning experiences but without SE experience.

For more details on the original R2E framework, please refer to the original [README_R2E.md](./README_R2E.md).

---

## 🔧 Setup

```bash
## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# activate venv
uv venv
source .venv/bin/activate
uv sync && uv pip install -e .
```

## 🚀 Quickstart

We have modified the uv environment to include the dotenv file for the environment variables. You can load different api keys by modifying the .env file.
After you have entered one of the LLM providers API key in the .env file, you can run the following command to test the installation.

Run the following command to test the installation:
```bash
python install_test.py
```

---

## Training EntroPO

### Dataset

The training instances are collected from [SWE-smith](https://github.com/SWE-bench/SWE-smith) and [R2E](https://github.com/R2E-Gym/R2E-Gym). We both use a subset of the two datasets. For SWE-smith, as the lastest version in huggingface contains many instances without problem statement and some non-python instances, we use a cleaned version from [r2e-edits/swesmith-clean](https://huggingface.co/datasets/r2e-edits/swesmith-clean). For R2E, we use the original R2E-Gym-subset to avoid the data leakage for SWE-Bench evaluation. 

### SFT Trajectories (Optional)

We use the [r2e-edits/swesmith-clean](https://huggingface.co/datasets/r2e-edits/swesmith-clean) to collect SFT trajectories as the patches are not verifiable. We have already collected the SFT trajectories using GLM-4.5 and uploaded to [EntroPO-SFT](https://huggingface.co/datasets/hubert233/EntroPO-SFT). If you want to collect the SFT trajectories yourself, you can run the following command:

```bash
python collect_swe_smith_trajectories.py
```

### SFT Training

We provide the training config files in [llamafactory-train](./llamafactory-train). Due to the fast updates of llama-factory, we provided one fork for our EntroPO implementation in [llamafactory-entropo](https://github.com/sherdencooper/LLaMA-Factory).

You can use the following command to generate the llama-factory format SFT data:
```bash
python ./llamafactory-train/generate_sft_data.py
```

Note that by default, this script use the [EntroPO-SFT](https://huggingface.co/datasets/hubert233/EntroPO-SFT) for training, if you need more SFT data, you can also uncomment the line `"hubert233/R2E-Smith"` in the script. This complementary SFT data is provided by R2E and we processed and stored in [R2E-Smith](https://huggingface.co/datasets/hubert233/R2E-Smith). The processing script is [process_traj/merge_trajectory_datasets.py](./process_traj/merge_trajectory_datasets.py).


After you get the SFT data, you can use the following command to train the model:
```bash
llamafactory-cli train path_to_config/qwen3_sft.yaml
```

### Preference Training

After learning the SFT policy, we can use the finetuned model to collect the preference data on R2E-Gym-Subset instances with the following command:
```bash
python collect_r2e_trajectories.py
```

It is also suggested to use a stronger model to rollout the preference data, we use GLM-4.5 in our experiments. We use GLM-4.5 to run the instances twice and upload the preference data to [hubert233/R2E-GLM45](https://huggingface.co/datasets/hubert233/R2E-GLM45) and [hubert233/R2E-GLM45_1](https://huggingface.co/datasets/hubert233/R2E-GLM45_1). We also provide the collected preference data by SFT-tuned Qwen model in [hubert233/R2E-QwenCoder30BA3-sft](https://huggingface.co/datasets/hubert233/R2E-QwenCoder30BA3-sft) and [hubert233/R2E-QwenCoder30BA3-sft_1](https://huggingface.co/datasets/hubert233/R2E-QwenCoder30BA3-sft_1).

You can use ```python ./process_traj/process_r2e_trajectories.py``` to process and upload the preference data to HuggingFace for future use.

After you have collected the preference data, you can run the following command to generate the preference data in llama-factory format:
```bash
python ./llamafactory-train/generate_dpo_qwen.py
python ./llamafactory-train/generate_kto_qwen.py
```
We have tweaked the llamafactory script to support multi-turn DPO/KTO learning instead of only learning on the last response. The entro_alpha parameter controls the importance of the entropy regularization. It is suggested to set it to 0.105 to 0.15, otherwise it may have gradient norm vanishing problem.

After you have generated the preference data, you can use the following command to train the model:
```bash
llamafactory-cli train path_to_config/qwen3_dpo.yaml
llamafactory-cli train path_to_config/qwen3_kto.yaml
```

## Evaluation
You can run the evaluation on SWE-bench-Verified and SWE-bench-Lite with the following command:
```bash
python collect_sweverified_trajectories.py
python collect_swelite_trajectories.py
```

## Test-Time Scaling 
For test-time scaling, you can simply run the exp multiple times and refer to the [R2E TTS guidance](./reproduction/DEEPSWE_TTS_REPRODUCTION.MD). Specifically, we have made the following modifications to its original TTS workflow:
- We condense the user message instead of the llm response as llm response is short compared with user message when condensing. It does not help a lot to condense the llm response.
- Instead of highly relying on verifier model probability to select the best trajectory, we use it to filter out the bad trajectories with very low probability.
- Before hybrid selection, we first use the finished score to filter out the unfinished trajectories as they are likely to be incorrect.
- After reproduction test and regression score filter, we select the trajectory with the most iterations for SWE-bench-Verified and the fewest iterations for SWE-bench-Lite.

We provide the [trained verifier model](https://huggingface.co/hubert233/qwen3-coder-30b-verifier-merged) for use. If you want to train the verifier model yourself, you can use the ```process_traj/prepare_ef_verifier_dataset.py``` to prepare the verifier training dataset and the training config to train the verifier model.

## SWEBench Submission

Also refer to the [R2E TTS guidance](./reproduction/DEEPSWE_TTS_REPRODUCTION.MD) to generate the submission file for SWE-bench.

## State-of-the-Art Performance on SWE-bench

| Method         | SWE-bench-Verified | SWE-bench-Lite |
|----------------|--------------------|----------------|
| origin         | 37.4%              | 28.00%         |
| sft            | 43.8%              | 33.67%         |
| sft+ekto       | 51.6%              | 44.67%         |
| sft+ekto@bo16  | 59.8%              | 49.33%         |

---

## Citation

TBD
