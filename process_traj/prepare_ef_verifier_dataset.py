#!/usr/bin/env python3
"""
Script to prepare execution-free (EF) verifier training dataset.
Processes trajectory JSONL files and converts them to verifier training format
for fine-tuning the verifier model.

Usage:
    python preapare_ef_verifier_dataset.py --jsonl_file_paths "path1.jsonl,path2.jsonl" --dataset_name "my-verifier-dataset"
"""

import os
import json
import fire
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Any

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.verifiers.prepare_ef_verifier_input import traj2verifier_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_trajectory_finished(traj_data: Dict) -> bool:
    """
    Check if trajectory is properly finished by looking for '<function=finish>' in last assistant message.
    
    Args:
        traj_data: Raw trajectory data
        
    Returns:
        bool: True if trajectory is finished, False otherwise
    """
    try:
        # Get the history from trajectory
        history = traj_data.get('history', [])
        if not history:
            return False
        
        # Find the last assistant message
        last_assistant_msg = None
        for msg in reversed(history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg
                break
        
        if last_assistant_msg is None:
            return False
            
        # Check if it contains '<function=finish>'
        content = last_assistant_msg.get('content', '')
        return '<function=finish>' in content
        
    except Exception as e:
        logger.debug(f"Error checking if trajectory is finished for {traj_data.get('docker_image', 'unknown')}: {e}")
        return False


def load_trajectories_from_file(jsonl_file_path: str) -> List[Dict]:
    """Load trajectory data from a single JSONL file."""
    trajectories = []
    logger.info(f"Loading trajectories from {jsonl_file_path}...")
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                traj_data = json.loads(line.strip())
                trajectories.append(traj_data)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num} in {jsonl_file_path}: JSON decode error: {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num} in {jsonl_file_path}: Unexpected error: {e}")
                continue
    
    logger.info(f"Loaded {len(trajectories)} trajectories from {jsonl_file_path}")
    return trajectories


def process_trajectory_to_verifier_format(traj_data: Dict, max_tokens: int = 60000) -> Dict[str, Any]:
    """
    Process a single trajectory into verifier training format.
    
    Args:
        traj_data: Raw trajectory data
        max_tokens: Maximum tokens for compression
        
    Returns:
        Dict with processed data or None if failed
    """
    try:
        # Convert trajectory to verifier training format
        verifier_data, success = traj2verifier_data(traj_data, max_tokens=max_tokens)
        
        if not success or not verifier_data:
            return None
        
        # Extract metadata
        docker_image = traj_data.get('docker_image', '')
        reward = traj_data.get('reward', 0)
        instance_id = traj_data.get('ds', {}).get('instance_id', docker_image.split('.')[-1] if docker_image else '')
        
        # Calculate token count
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        
        # Calculate tokens for each message
        system_tokens = len(tokenizer.encode(verifier_data[0]['content']))
        user_tokens = len(tokenizer.encode(verifier_data[1]['content']))
        assistant_tokens = len(tokenizer.encode(verifier_data[2]['content']))
        total_tokens = system_tokens + user_tokens + assistant_tokens
        
        return {
            'messages': verifier_data,
            'instance_id': instance_id,
            'docker_image': docker_image,
            'reward': reward,
            'label': 'YES' if reward == 1 else 'NO',
            'system_tokens': system_tokens,
            'user_tokens': user_tokens,
            'assistant_tokens': assistant_tokens,
            'total_tokens': total_tokens,
            'source': 'ef_verifier_training'
        }
        
    except Exception as e:
        logger.error(f"Error processing trajectory {traj_data.get('docker_image', 'unknown')}: {e}")
        return None


def process_trajectories_to_verifier_dataset(
    jsonl_file_paths: List[str], 
    max_tokens: int = 60000,
    max_workers: int = 40
) -> List[Dict[str, Any]]:
    """
    Process multiple trajectory files into verifier training dataset.
    
    Args:
        jsonl_file_paths: List of JSONL file paths
        max_tokens: Maximum tokens for compression
        max_workers: Number of parallel workers
        
    Returns:
        List of processed training examples
    """
    # Load all trajectories
    all_trajectories = []
    for file_path in jsonl_file_paths:
        if os.path.exists(file_path):
            trajectories = load_trajectories_from_file(file_path)
            all_trajectories.extend(trajectories)
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info(f"Total trajectories loaded: {len(all_trajectories)}")
    
    # Filter out unfinished trajectories
    logger.info("Filtering out unfinished trajectories...")
    finished_trajectories = []
    unfinished_count = 0
    
    for traj in all_trajectories:
        if is_trajectory_finished(traj):
            finished_trajectories.append(traj)
        else:
            unfinished_count += 1
    
    logger.info(f"Filtered out {unfinished_count} unfinished trajectories")
    logger.info(f"Processing {len(finished_trajectories)} finished trajectories")
    
    if not finished_trajectories:
        logger.warning("No finished trajectories found!")
        return []
    
    # Process trajectories in parallel
    processed_data = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_trajectory_to_verifier_format, traj, max_tokens)
            for traj in finished_trajectories
        ]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            try:
                result = future.result()
                if result is not None:
                    processed_data.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                continue
    
    logger.info(f"Successfully processed {len(processed_data)}/{len(finished_trajectories)} finished trajectories")
    logger.info(f"Total filtered out: {unfinished_count} unfinished + {len(finished_trajectories) - len(processed_data)} processing failures")
    return processed_data


def calculate_dataset_statistics(processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for the processed dataset."""
    if not processed_data:
        return {}
    
    df = pd.DataFrame(processed_data)
    
    # Convert numpy types to native Python types for JSON serialization
    stats = {
        'total_samples': int(len(processed_data)),
        'label_distribution': {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        'reward_distribution': {str(k): int(v) for k, v in df['reward'].value_counts().to_dict().items()},
        'avg_total_tokens': float(df['total_tokens'].mean()),
        'max_total_tokens': int(df['total_tokens'].max()),
        'min_total_tokens': int(df['total_tokens'].min()),
        'avg_user_tokens': float(df['user_tokens'].mean()),
        'max_user_tokens': int(df['user_tokens'].max()),
        'avg_system_tokens': float(df['system_tokens'].mean()),
        'avg_assistant_tokens': float(df['assistant_tokens'].mean()),
        'unique_instances': int(df['instance_id'].nunique()),
        'source_distribution': {str(k): int(v) for k, v in df['source'].value_counts().to_dict().items()},
        'docker_images_count': int(df['docker_image'].nunique()),
    }
    
    return stats


def upload_to_huggingface(dataset_df: pd.DataFrame, dataset_name: str):
    """
    Upload the processed dataset to HuggingFace Hub as a private dataset.
    
    Args:
        dataset_df: The processed dataset
        dataset_name: Name of the dataset on HF Hub
    """
    try:
        logger.info(f"Preparing to upload dataset to HuggingFace as '{dataset_name}'...")
        
        # Convert pandas DataFrame to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(dataset_df)
        
        # Upload to HuggingFace Hub as private dataset
        logger.info("Uploading to HuggingFace Hub...")
        hf_dataset.push_to_hub(
            dataset_name,
            private=True,
            split="train"
        )
        
        logger.info(f"Successfully uploaded dataset to HuggingFace Hub as '{dataset_name}' (private)")
        
    except Exception as e:
        logger.error(f"Error uploading to HuggingFace: {e}")
        raise


def main(
    jsonl_file_paths: str="traj/r2e.jsonl,traj/r2e_1.jsonl",
    dataset_name: str = "ef-verifier-training-dataset-qwen",
    max_tokens: int = 60000,
    max_workers: int = 40,
    output_dir: str = "./process_traj"
):
    """
    Main function to create EF verifier training dataset.
    
    Args:
        jsonl_file_paths: Comma-separated list of JSONL file paths or glob patterns
        dataset_name: Name for the HuggingFace dataset
        max_tokens: Maximum tokens for message compression
        max_workers: Number of parallel workers
        output_dir: Directory to save local backup files
    """
    try:
        # Parse file paths
        file_paths = []
        for path_pattern in jsonl_file_paths.split(','):
            path_pattern = path_pattern.strip()
            # Handle glob patterns
            if '*' in path_pattern:
                matching_files = glob.glob(path_pattern)
                file_paths.extend(matching_files)
            else:
                file_paths.append(path_pattern)
        
        logger.info(f"Processing files: {file_paths}")
        
        # Process trajectories
        processed_data = process_trajectories_to_verifier_dataset(
            file_paths, max_tokens=max_tokens, max_workers=max_workers
        )
        
        if not processed_data:
            logger.error("No data processed successfully. Exiting.")
            return
        
        # Calculate statistics
        stats = calculate_dataset_statistics(processed_data)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier handling
        dataset_df = pd.DataFrame(processed_data)
        
        # Save locally for backup
        logger.info("Saving processed dataset locally...")
        
        # Save as parquet (efficient for large datasets)
        dataset_df.to_parquet(f'{output_dir}/ef_verifier_dataset.parquet', index=False)
        
        # Save as JSONL for compatibility
        with open(f'{output_dir}/ef_verifier_dataset.jsonl', 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
        
        # Save statistics
        with open(f'{output_dir}/ef_verifier_dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset processing completed successfully!")
        logger.info(f"Files saved locally in {output_dir}:")
        logger.info("- ef_verifier_dataset.parquet")
        logger.info("- ef_verifier_dataset.jsonl")
        logger.info("- ef_verifier_dataset_statistics.json")
        
        # Print summary statistics
        print("\n=== EF VERIFIER DATASET SUMMARY ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Unique instances: {stats['unique_instances']}")
        print(f"Docker images: {stats['docker_images_count']}")
        print(f"Average total tokens: {stats['avg_total_tokens']:.1f}")
        print(f"Max total tokens: {stats['max_total_tokens']}")
        print(f"Average user tokens: {stats['avg_user_tokens']:.1f}")
        print(f"Max user tokens: {stats['max_user_tokens']}")
        
        print(f"\nLabel distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        print(f"\nReward distribution:")
        for reward, count in stats['reward_distribution'].items():
            print(f"  {reward}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        # Upload to HuggingFace Hub
        logger.info(f"Uploading to HuggingFace Hub as '{dataset_name}'...")
        upload_to_huggingface(dataset_df, dataset_name)
        
        print(f"\n✅ Dataset successfully created and uploaded as '{dataset_name}'")
        print(f"📊 {stats['total_samples']} training examples ready for verifier fine-tuning")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)