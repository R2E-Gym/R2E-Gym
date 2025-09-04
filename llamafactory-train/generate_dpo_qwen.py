import json
import os
import argparse
from datasets import load_dataset, DownloadMode
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

def load_dataset_with_filtering(dataset_name: str, force_download: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load dataset from Hugging Face and filter by exit_reason='agent'.
    Returns a dictionary with instance_id as key.
    
    Args:
        dataset_name: Name of the dataset to load
        force_download: Whether to force redownload the dataset
    
    Returns:
        Dictionary mapping instance_id to dataset row
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
    
    # Filter by exit_reason='agent' and organize by instance_id
    filtered_data = {}
    for row in data:
        if row.get('exit_reasons') == 'agent':
            instance_id = row.get('instance_id')
            if instance_id:
                filtered_data[instance_id] = row
    
    print(f"Filtered {len(filtered_data)} rows with exit_reasons='agent' from {len(data)} total rows")
    return filtered_data

def messages_match_prefix(messages1: List[Dict], messages2: List[Dict]) -> bool:
    """
    Check if the system message and first user message are the same in both datasets.
    
    Args:
        messages1: Messages from first dataset
        messages2: Messages from second dataset
    
    Returns:
        True if system and first user messages match
    """
    # Extract system and first user messages from both
    def extract_prefix(messages):
        system_msg = None
        first_user_msg = None
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system' and system_msg is None:
                system_msg = content
            elif role == 'user' and first_user_msg is None:
                first_user_msg = content
                break  # Stop after first user message
        
        return system_msg, first_user_msg
    
    sys1, user1 = extract_prefix(messages1)
    sys2, user2 = extract_prefix(messages2)
    
    return sys1 == sys2 and user1 == user2

def convert_messages_to_conversations(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert messages to DPO conversation format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        List of conversations in DPO format
    """
    conversations = []
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        # Map roles to DPO format
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
    
    return conversations

def extract_common_prefix_and_divergent_parts(messages1: List[Dict[str, str]], messages2: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extract the common prefix (system + first user message) and the divergent conversation parts.
    
    Args:
        messages1: Messages from first dataset
        messages2: Messages from second dataset
    
    Returns:
        Tuple of (common_prefix, divergent_part1, divergent_part2)
    """
    conv1 = convert_messages_to_conversations(messages1)
    conv2 = convert_messages_to_conversations(messages2)
    
    # Find the end of common prefix (system + first user message)
    common_prefix = []
    prefix_end_idx = -1
    
    # Add system message if present
    if conv1 and conv1[0]["from"] == "system":
        common_prefix.append(conv1[0])
        prefix_end_idx = 0
    
    # Add first user message
    for i, conv in enumerate(conv1):
        if conv["from"] == "human":
            common_prefix.append(conv)
            prefix_end_idx = i
            break
    
    # Extract divergent parts (everything after the common prefix)
    divergent_part1 = conv1[prefix_end_idx + 1:] if prefix_end_idx >= 0 else conv1
    divergent_part2 = conv2[prefix_end_idx + 1:] if prefix_end_idx >= 0 else conv2
    
    return common_prefix, divergent_part1, divergent_part2

def create_dpo_entry(instance_id: str, row1: Dict[str, Any], row2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create a DPO format entry from two dataset rows.
    
    Args:
        instance_id: The instance ID
        row1: First dataset row
        row2: Second dataset row
    
    Returns:
        DPO formatted dictionary or None if invalid
    """
    messages1 = row1.get('messages', [])
    messages2 = row2.get('messages', [])
    reward1 = row1.get('reward', 0)
    reward2 = row2.get('reward', 0)
    
    # Add system message from separate field if it exists
    system1 = row1.get('system', None)
    system2 = row2.get('system', None)
    if system1:
        messages1 = [{"role": "system", "content": system1}] + messages1
    if system2:
        messages2 = [{"role": "system", "content": system2}] + messages2
    
    # Check if rewards are different (one positive, one negative)
    if (reward1 == 1 and reward2 == 1) or (reward1 != 1 and reward2 != 1):
        return None  # Skip if both positive or both negative
    
    # Extract common prefix and divergent parts
    common_prefix, divergent_part1, divergent_part2 = extract_common_prefix_and_divergent_parts(messages1, messages2)
    
    if not divergent_part1 or not divergent_part2:
        return None  # Skip if no divergent conversation parts
    
    # Determine which is chosen and which is rejected based on reward
    if reward1 == 1:
        chosen_divergent = divergent_part1
        rejected_divergent = divergent_part2
    else:
        chosen_divergent = divergent_part2
        rejected_divergent = divergent_part1
    
    # Create DPO entry
    dpo_entry = {
        "conversations": common_prefix,
        "chosen": chosen_divergent,
        "rejected": rejected_divergent
    }
    
    return dpo_entry

def find_dataset_pairs(datasets: List[str]) -> List[Tuple[str, str]]:
    """
    Find dataset pairs based on naming patterns.
    
    Args:
        datasets: List of dataset names
    
    Returns:
        List of dataset pairs
    """
    pairs = []
    
    # Group datasets by base name
    base_groups = defaultdict(list)
    for dataset in datasets:
        # Remove trailing numbers and underscores to find base name
        base_name = dataset.rstrip('_0123456789')
        base_groups[base_name].append(dataset)
    
    # Create pairs within each group
    for base_name, group_datasets in base_groups.items():
        if len(group_datasets) >= 2:
            # Sort to ensure consistent pairing
            group_datasets.sort()
            for i in range(0, len(group_datasets) - 1, 2):
                if i + 1 < len(group_datasets):
                    pairs.append((group_datasets[i], group_datasets[i + 1]))
    
    return pairs

def main():
    """
    Main function to load datasets, find pairs, and generate DPO format data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate DPO format dataset from Qwen datasets')
    parser.add_argument('--force-download', action='store_true',
                       help='Force redownload of datasets (time-costly, use only when needed)')
    parser.add_argument('--output-file', type=str, default='qwen_dpo_data.json',
                       help='Output filename (default: qwen_dpo_data.json)')
    
    args = parser.parse_args()
    
    # Dataset names
    datasets = [
        "hubert233/R2E-GLM45_1",
        "hubert233/R2E-GLM45",
        "hubert233/R2E-QwenCoder30BA3-sft_1",
        "hubert233/R2E-QwenCoder30BA3-sft"
    ]
    
    # Find dataset pairs
    dataset_pairs = find_dataset_pairs(datasets)
    print(f"Found {len(dataset_pairs)} dataset pairs: {dataset_pairs}")
    
    all_dpo_data = []
    
    print(f"Force download: {'Yes' if args.force_download else 'No'}")
    
    # Process each pair
    for dataset1_name, dataset2_name in dataset_pairs:
        print(f"\nProcessing pair: {dataset1_name} <-> {dataset2_name}")
        
        try:
            # Load both datasets
            data1 = load_dataset_with_filtering(dataset1_name, force_download=args.force_download)
            data2 = load_dataset_with_filtering(dataset2_name, force_download=args.force_download)
            
            # Find common instance_ids
            common_ids = set(data1.keys()) & set(data2.keys())
            print(f"Found {len(common_ids)} common instance_ids")
            
            pair_count = 0
            skipped_count = 0
            
            # Process each common instance_id
            for instance_id in common_ids:
                row1 = data1[instance_id]
                row2 = data2[instance_id]
                
                messages1 = row1.get('messages', [])
                messages2 = row2.get('messages', [])
                
                # Add system messages if they exist in separate fields
                system1 = row1.get('system', None)
                system2 = row2.get('system', None)
                if system1:
                    messages1 = [{"role": "system", "content": system1}] + messages1
                if system2:
                    messages2 = [{"role": "system", "content": system2}] + messages2
                
                # Check if system and first user messages match
                if not messages_match_prefix(messages1, messages2):
                    skipped_count += 1
                    continue
                
                # Create DPO entry
                dpo_entry = create_dpo_entry(instance_id, row1, row2)
                if dpo_entry:
                    all_dpo_data.append(dpo_entry)
                    pair_count += 1
                else:
                    skipped_count += 1
            
            print(f"Created {pair_count} DPO pairs, skipped {skipped_count} instances")
            
        except Exception as e:
            print(f"Error processing pair {dataset1_name} <-> {dataset2_name}: {str(e)}")
            continue
    
    print(f"\nTotal DPO entries created: {len(all_dpo_data)}")
    
    # Create output directory if it doesn't exist
    output_dir = "./dataset/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, args.output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to: {output_file}")
    print(f"Total DPO pairs: {len(all_dpo_data)}")

if __name__ == "__main__":
    main()