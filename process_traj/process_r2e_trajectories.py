#!/usr/bin/env python3
"""
Script to process R2E trajectory data (e.g., QwenCoder30BA3_r2e.jsonl).
Converts 'history' to 'messages', extracts instance_id from 'docker_image' field,
and calculates step_number, exit_reasons, token_count.
"""

import json
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_assistant_messages(messages):
    """
    Count the number of assistant messages to determine step number.
    
    Args:
        messages (list): List of message dictionaries
        
    Returns:
        int: Number of assistant messages
    """
    return sum(1 for msg in messages if msg.get('role') == 'assistant')

def clean_messages(messages):
    """
    Clean messages by removing the final "<<< Finished >>>" user message if present.
    
    Args:
        messages (list): List of message dictionaries
        
    Returns:
        list: Cleaned messages list
    """
    if not messages:
        return messages
    
    # Check if last message is the "<<< Finished >>>" user message
    last_message = messages[-1]
    if (last_message.get('role') == 'user' and
        last_message.get('content', '').strip() == '<<< Finished >>>'):
        return messages[:-1]  # Remove the last message
    
    return messages

def get_exit_reason(messages):
    """
    Determine exit reason based on the last message.
    
    Args:
        messages (list): List of message dictionaries (should be cleaned first)
        
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

def process_r2e_trajectories(jsonl_file_path):
    """
    Process R2E trajectory data.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file (e.g., QwenCoder30BA3_r2e.jsonl)
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    logger.info("Loading Qwen/Qwen3-32B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    
    processed_data = []
    
    logger.info(f"Processing {jsonl_file_path}...")
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing lines"), 1):
            try:
                data = json.loads(line.strip())
                
                # Extract messages from 'history' field
                raw_messages = data.get('history', [])
                
                # Clean messages by removing "<<< Finished >>>" if present
                messages = clean_messages(raw_messages)
                
                # Extract instance_id from 'docker_image' field for R2E dataset
                instance_id = data.get('docker_image')
                
                # Extract reward from the data
                reward = data.get('reward', None)
                
                if instance_id and messages:
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
                        'reward': reward,
                        'source': 'r2e'
                    }
                    processed_data.append(processed_item)
                else:
                    if not instance_id:
                        logger.warning(f"Line {line_num}: No instance_id found in 'docker_image' field")
                    if not messages:
                        logger.warning(f"Line {line_num}: No messages found in 'history' field")
                        
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}")
                continue
    
    logger.info(f"Processed {len(processed_data)} items from {jsonl_file_path}")
    return pd.DataFrame(processed_data)

def upload_to_huggingface(dataset_df, dataset_name="PLACEHOLDER_DATASET_NAME"):
    """
    Upload the processed dataset to HuggingFace Hub as a private dataset.
    
    Args:
        dataset_df (pd.DataFrame): The processed dataset
        dataset_name (str): Name of the dataset on HF Hub (to be filled by user)
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

def main():
    """
    Main execution function.
    """
    try:
        # Process the R2E trajectories
        # TODO: Fill in the actual path to your R2E JSONL file
        jsonl_file_path = "./traj/r2e_1.jsonl"
        processed_dataset = process_r2e_trajectories(jsonl_file_path)
        
        # Save locally for backup
        logger.info("Saving processed dataset locally...")
        
        # Save as parquet (efficient for large datasets)
        processed_dataset.to_parquet('./process_traj/r2e_trajectories_processed.parquet', index=False)
        
        # Save as JSON for compatibility
        processed_dataset.to_json('./process_traj/r2e_trajectories_processed.json', orient='records', lines=True)
        
        # Save basic statistics
        stats = {
            'total_samples': len(processed_dataset),
            'source_distribution': processed_dataset['source'].value_counts().to_dict(),
            'avg_step_number': processed_dataset['step_number'].mean(),
            'exit_reasons_distribution': processed_dataset['exit_reasons'].value_counts().to_dict(),
            'avg_token_count': processed_dataset['token_count'].mean(),
            'avg_reward': processed_dataset['reward'].mean(),
            'reward_distribution': processed_dataset['reward'].value_counts().to_dict() if processed_dataset['reward'].dtype == 'object' else None,
            'reward_min': processed_dataset['reward'].min(),
            'reward_max': processed_dataset['reward'].max()
        }
        
        with open('./process_traj/r2e_dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset processing completed successfully!")
        logger.info(f"Files saved locally:")
        logger.info("- r2e_trajectories_processed.parquet")
        logger.info("- r2e_trajectories_processed.json")
        logger.info("- r2e_dataset_statistics.json")
        
        # Print summary statistics
        print("\n=== R2E DATASET SUMMARY ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Average steps per trajectory: {stats['avg_step_number']:.2f}")
        print(f"Average tokens per trajectory: {stats['avg_token_count']:.2f}")
        print(f"Average reward: {stats['avg_reward']:.4f}")
        print(f"Reward range: {stats['reward_min']:.4f} to {stats['reward_max']:.4f}")
        print(f"\nSource distribution:")
        for source, count in stats['source_distribution'].items():
            print(f"  {source}: {count}")
        print(f"\nExit reasons distribution:")
        for reason, count in stats['exit_reasons_distribution'].items():
            print(f"  {reason}: {count}")
        if stats['reward_distribution']:
            print(f"\nReward distribution:")
            for reward, count in stats['reward_distribution'].items():
                print(f"  {reward}: {count}")
        
        # Upload to HuggingFace Hub
        # TODO: Replace with your desired HuggingFace dataset name
        upload_to_huggingface(processed_dataset, "R2E-GLM45_1")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()