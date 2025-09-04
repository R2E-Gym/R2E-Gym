import json
import os
from datasets import load_dataset
from typing import List, Dict, Any

def convert_to_sharegpt_format(messages: List[Dict[str, str]], tools: str = None) -> Dict[str, Any]:
    """
    Convert messages to ShareGPT format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        tools: Optional tool description
    
    Returns:
        Dictionary in ShareGPT format
    """
    conversations = []
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        # Map roles to ShareGPT format
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
    
    result = {"conversations": conversations}
    
    if tools:
        result["tools"] = tools
    
    return result

def load_and_filter_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load dataset from Hugging Face and filter by exit_reason='agent'.
    
    Args:
        dataset_name: Name of the dataset to load
    
    Returns:
        List of filtered dataset rows
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Assuming the dataset has a 'train' split, adjust if needed
    if 'train' in dataset:
        data = dataset['train']
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")
    
    filtered_data = []
    for row in data:
        filtered_data.append(row)
    
    print(f"Filtered {len(filtered_data)} rows with exit_reasons='agent' from {len(data)} total rows")
    return filtered_data

def process_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single dataset row and convert to ShareGPT format.
    
    Args:
        row: Dataset row containing messages and other fields
    
    Returns:
        ShareGPT formatted dictionary
    """
    messages = row.get('messages', [])
    tools = row.get('tools', None)
    
    # Add system message from separate field to messages if it exists
    system = row.get('system', None)
    if system:
        # Insert system message at the beginning
        system_message = {"role": "system", "content": system}
        messages = [system_message] + messages
    
    return convert_to_sharegpt_format(messages, tools)

def main():
    """
    Main function to load datasets, process them, and save in ShareGPT format.
    """
    # Dataset names
    datasets = [
        # "hubert233/R2E-Smith",
        "hubert233/ef-verifier-training-dataset-qwen"
    ]
    
    all_data = []
    
    # Load and process each dataset
    for dataset_name in datasets:
        try:
            filtered_data = load_and_filter_dataset(dataset_name)
            
            # Process each row
            for row in filtered_data:
                processed_row = process_dataset_row(row)
                all_data.append(processed_row)
                
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    print(f"Total processed rows: {len(all_data)}")
    
    # Create output directory if it doesn't exist
    output_dir = "./dataset/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, "qwen_verify_data_sharegpt.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to: {output_file}")
    print(f"Total conversations: {len(all_data)}")

if __name__ == "__main__":
    main()