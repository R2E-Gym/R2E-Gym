import json
import os
import argparse
from datasets import load_dataset, DownloadMode
from typing import List, Dict, Any

def convert_to_kto_format(messages: List[Dict[str, str]], reward: Any, tools: str = None) -> Dict[str, Any]:
    """
    Convert messages to KTO format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        reward: Reward value from dataset (1 for True, otherwise False)
        tools: Optional tool description
    
    Returns:
        Dictionary in KTO format
    """
    conversations = []
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        # Map roles to KTO format (same as ShareGPT)
        if role == 'user':
            from_role = 'human'
        elif role == 'assistant':
            from_role = 'gpt'
        elif role == 'system':
            from_role = 'system'
        elif role == 'function':
            from_role = 'function_call'
        elif role == 'observation':
            from_role = 'observation'
        else:
            from_role = role  # Keep original role if not recognized
        
        conversations.append({
            "from": from_role,
            "value": content
        })
    
    # Convert reward to kto_tag: True if reward is 1, False otherwise
    kto_tag = True if reward == 1 else False
    
    result = {
        "conversations": conversations,
        "kto_tag": kto_tag
    }
    
    if tools:
        result["tools"] = tools
    
    return result

def load_and_filter_dataset(dataset_name: str, force_download: bool = False) -> List[Dict[str, Any]]:
    """
    Load dataset from Hugging Face and filter by exit_reason='agent'.
    
    Args:
        dataset_name: Name of the dataset to load
        force_download: Whether to force redownload the dataset
    
    Returns:
        List of filtered dataset rows
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Only pass force_download if it's True, to avoid the BuilderConfig error
    if force_download:
        try:
            dataset = load_dataset(dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD)
        except Exception as e:
            print(f"Warning: force_download failed, trying without it: {str(e)}")
            dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    # Assuming the dataset has a 'train' split, adjust if needed
    if 'train' in dataset:
        data = dataset['train']
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")
    
    # Filter by exit_reason='agent'
    filtered_data = []
    for row in data:
        if row.get('exit_reasons') == 'agent':
            filtered_data.append(row)
    
    print(f"Filtered {len(filtered_data)} rows with exit_reasons='agent' from {len(data)} total rows")
    return filtered_data

def process_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single dataset row and convert to KTO format.
    
    Args:
        row: Dataset row containing messages, reward, and other fields
    
    Returns:
        KTO formatted dictionary
    """
    messages = row.get('messages', [])
    reward = row.get('reward', 0)  # Default to 0 if reward not found
    tools = row.get('tools', None)
    
    # Add system message from separate field to messages if it exists
    system = row.get('system', None)
    if system:
        # Insert system message at the beginning
        system_message = {"role": "system", "content": system}
        messages = [system_message] + messages
    
    return convert_to_kto_format(messages, reward, tools)

def main():
    """
    Main function to load datasets, process them, and save in KTO format.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate KTO format dataset from Qwen datasets')
    parser.add_argument('--force-download', action='store_true',
                       help='Force redownload of datasets (time-costly, use only when needed)')
    parser.add_argument('--output-file', type=str, default='qwen_kto_dat.json',
                       help='Output filename (default: qwen_kto_data.json)')
    
    args = parser.parse_args()
    
    # Dataset names
    datasets = [
        "hubert233/R2E-GLM45_1",
        "hubert233/R2E-GLM45",
        "hubert233/R2E-QwenCoder30BA3-sft_1",
        "hubert233/R2E-QwenCoder30BA3-sft"
    ]
    
    all_data = []
    
    print(f"Force download: {'Yes' if args.force_download else 'No'}")
    
    # Load and process each dataset
    for dataset_name in datasets:
        try:
            filtered_data = load_and_filter_dataset(dataset_name, force_download=args.force_download)
            
            # Process each row
            for row in filtered_data:
                processed_row = process_dataset_row(row)
                all_data.append(processed_row)
                
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    print(f"Total processed rows: {len(all_data)}")
    
    # Print statistics about kto_tag distribution
    true_count = sum(1 for item in all_data if item.get('kto_tag') == True)
    false_count = sum(1 for item in all_data if item.get('kto_tag') == False)
    print(f"KTO tag distribution - True: {true_count}, False: {false_count}")
    
    # Create output directory if it doesn't exist
    output_dir = "./dataset/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, args.output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to: {output_file}")
    print(f"Total conversations: {len(all_data)}")

if __name__ == "__main__":
    main()