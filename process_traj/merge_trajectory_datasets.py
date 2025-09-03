#!/usr/bin/env python3
"""
Script to merge two trajectory datasets from HuggingFace:
1. R2E-Gym/R2EGym-SFT-Trajectories
2. r2e-edits/SWE-smith-trajectories-R2E-v2

Each dataset requires specific processing before merging.
"""

import re
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import logging
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_problem_statement(user_message):
    """
    Extract problem_statement from user message using <github_issue> tags.
    
    Args:
        user_message (str): The user message containing github issue
        
    Returns:
        str: Extracted problem statement or None if not found
    """
    # Look for <github_issue> tags in the message
    pattern = r'<github_issue>(.*?)</github_issue>'
    matches = re.findall(pattern, user_message, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def count_assistant_messages(messages):
    """
    Count the number of assistant messages to determine step number.
    
    Args:
        messages (list): List of message dictionaries
        
    Returns:
        int: Number of assistant messages
    """
    return sum(1 for msg in messages if msg.get('role') == 'assistant')

def get_exit_reason(messages):
    """
    Determine exit reason based on the last message.
    
    Args:
        messages (list): List of message dictionaries
        
    Returns:
        str: 'agent', 'exceed_limit', or 'query_error'
    """
    if not messages:
        return 'query_error'
    
    last_message = messages[-1]
    
    # If last message is not from assistant, it's a query error
    if last_message.get('role') != 'assistant':
        return 'query_error'
    
    # Check if assistant message contains '<function=finish>'
    content = last_message.get('content', '')
    if '<function=finish>' in content:
        return 'agent'
    else:
        return 'exceed_limit'

def calculate_token_count(messages, tokenizer):
    """
    Calculate token count using tokenizer.apply_chat_template.
    
    Args:
        messages (list): List of message dictionaries
        tokenizer: The tokenizer instance
        
    Returns:
        int: Number of tokens
    """
    try:
        # Apply chat template and tokenize
        formatted_messages = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        tokens = tokenizer.encode(formatted_messages)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error calculating tokens: {e}")
        return 0

def process_r2egym_sft_trajectories():
    """
    Process R2E-Gym/R2EGym-SFT-Trajectories dataset.
    
    Returns:
        pd.DataFrame: Processed dataset
    """
    logger.info("Loading R2E-Gym/R2EGym-SFT-Trajectories dataset...")
    
    # Load the main trajectory dataset
    trajectory_dataset = load_dataset("R2E-Gym/R2EGym-SFT-Trajectories", split="train")
    
    # Load the subset dataset to get problem_statement to instance_id mapping
    logger.info("Loading R2E-Gym/R2E-Gym-Subset for instance_id mapping...")
    subset_dataset = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    
    # Create mapping from problem_statement to instance_id (docker_image)
    problem_to_instance = {}
    for item in subset_dataset:
        problem_statement = item['problem_statement']
        docker_image = item['docker_image']
        problem_to_instance[problem_statement] = docker_image
    
    # Load tokenizer
    logger.info("Loading Qwen/Qwen3-32B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    
    processed_data = []
    
    logger.info("Processing R2EGym-SFT-Trajectories...")
    for item in tqdm(trajectory_dataset):
        messages = item['messages']
        
        # Extract problem_statement from the second message (user message)
        if len(messages) >= 2:
            user_message = messages[1].get('content', '')
            problem_statement = extract_problem_statement(user_message)[:30]
            
            if problem_statement:
                # Get instance_id from mapping
                instance_id = None
                for key, value in problem_to_instance.items():
                    if problem_statement in key:
                        instance_id = value
                        break
                
                if instance_id:
                    # Calculate step number (number of assistant messages)
                    step_number = count_assistant_messages(messages)
                    
                    # Determine exit reason
                    exit_reason = get_exit_reason(messages)
                    
                    # Calculate token count
                    token_count = calculate_token_count(messages, tokenizer)
                    
                    processed_item = {
                        'messages': messages,
                        'instance_id': instance_id,
                        'step_number': step_number,
                        'exit_reasons': exit_reason,
                        'token_count': token_count,
                        'source': 'R2E-subset'
                    }
                    processed_data.append(processed_item)
                else:
                    logger.warning(f"No instance_id found for problem_statement: {problem_statement[:100]}...")
            else:
                logger.warning("No problem_statement found in user message")
    
    logger.info(f"Processed {len(processed_data)} items from R2EGym-SFT-Trajectories")
    return pd.DataFrame(processed_data)

def process_swe_smith_trajectories():
    """
    Process r2e-edits/SWE-smith-trajectories-R2E-v2 dataset.
    
    Returns:
        pd.DataFrame: Processed dataset
    """
    logger.info("Loading r2e-edits/SWE-smith-trajectories-R2E-v2 dataset...")
    
    # Load the trajectory dataset
    trajectory_dataset = load_dataset("r2e-edits/SWE-smith-trajectories-R2E-v2", split="train")
    
    # Load swesmith-clean dataset to check instance_id membership
    logger.info("Loading r2e-edits/swesmith-clean for source classification...")
    swesmith_clean = load_dataset("r2e-edits/swesmith-clean", split="train")
    
    # Create set of instance_ids in swesmith-clean
    swesmith_instance_ids = set(item['instance_id'] for item in swesmith_clean)
    
    # Load tokenizer
    logger.info("Loading Qwen/Qwen3-32B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    
    processed_data = []
    
    logger.info("Processing SWE-smith-trajectories...")
    for item in tqdm(trajectory_dataset):
        instance_id = item.get('instance_id')
        messages = item.get('messages', [])
        
        if instance_id and messages:
            # Calculate step number (number of assistant messages)
            step_number = count_assistant_messages(messages)
            
            # Determine exit reason
            exit_reason = get_exit_reason(messages)
            
            # Calculate token count
            token_count = calculate_token_count(messages, tokenizer)
            
            # Determine source based on instance_id membership in swesmith-clean
            if instance_id in swesmith_instance_ids:
                source = 'in-swesmith-clean'
            else:
                source = 'not-in-swesmith-clean'
            
            processed_item = {
                'messages': messages,
                'instance_id': instance_id,
                'step_number': step_number,
                'exit_reasons': exit_reason,
                'token_count': token_count,
                'source': source
            }
            processed_data.append(processed_item)
    
    logger.info(f"Processed {len(processed_data)} items from SWE-smith-trajectories")
    return pd.DataFrame(processed_data)

def merge_datasets():
    """
    Main function to process and merge both datasets.
    
    Returns:
        pd.DataFrame: Merged dataset
    """
    # Process both datasets
    r2egym_data = process_r2egym_sft_trajectories()
    swe_smith_data = process_swe_smith_trajectories()
    
    # Merge datasets
    logger.info("Merging datasets...")
    merged_data = pd.concat([r2egym_data, swe_smith_data], ignore_index=True)
    
    logger.info(f"Total merged dataset size: {len(merged_data)} items")
    logger.info(f"Source distribution:")
    print(merged_data['source'].value_counts())
    
    return merged_data

def upload_to_huggingface(merged_dataset, dataset_name="R2E-Smith"):
    """
    Upload the merged dataset to HuggingFace Hub as a private dataset.
    
    Args:
        merged_dataset (pd.DataFrame): The merged dataset
        dataset_name (str): Name of the dataset on HF Hub
    """
    try:
        logger.info(f"Preparing to upload dataset to HuggingFace as '{dataset_name}'...")
        
        # Convert pandas DataFrame to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(merged_dataset)
        
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

def main():
    """
    Main execution function.
    """
    try:
        # Merge datasets
        merged_dataset = merge_datasets()
        
        # Save to various formats locally
        logger.info("Saving merged dataset locally...")
        
        # Save as parquet (efficient for large datasets)
        merged_dataset.to_parquet('./process_traj/merged_trajectory_datasets.parquet', index=False)
        
        # Save as JSON for compatibility
        merged_dataset.to_json('./process_traj/merged_trajectory_datasets.json', orient='records', lines=True)
        
        # Save basic statistics
        stats = {
            'total_samples': len(merged_dataset),
            'source_distribution': merged_dataset['source'].value_counts().to_dict(),
            'avg_step_number': merged_dataset['step_number'].mean(),
            'exit_reasons_distribution': merged_dataset['exit_reasons'].value_counts().to_dict(),
            'avg_token_count': merged_dataset['token_count'].mean()
        }
        
        with open('./process_traj/dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset merging completed successfully!")
        logger.info(f"Files saved locally:")
        logger.info("- merged_trajectory_datasets.parquet")
        logger.info("- merged_trajectory_datasets.json")
        logger.info("- dataset_statistics.json")
        
        # Print summary statistics
        print("\n=== DATASET SUMMARY ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Average steps per trajectory: {stats['avg_step_number']:.2f}")
        print(f"Average tokens per trajectory: {stats['avg_token_count']:.2f}")
        print(f"\nSource distribution:")
        for source, count in stats['source_distribution'].items():
            print(f"  {source}: {count}")
        print(f"\nExit reasons distribution:")
        for reason, count in stats['exit_reasons_distribution'].items():
            print(f"  {reason}: {count}")
        
        # Upload to HuggingFace Hub
        upload_to_huggingface(merged_dataset, "R2E-Smith")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()