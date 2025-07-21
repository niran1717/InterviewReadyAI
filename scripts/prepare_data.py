
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import Dataset # Only import Dataset, as we're not using load_dataset directly
import os
import json
import pandas as pd # Import pandas for data loading

# --- Configuration ---
# IMPORTANT: Adjust this path to where your JSONL file is located locally.
# Assuming 'interview_data.jsonl' is in a 'data' subfolder within your project.
data_path = "data/aws_dataset.jsonl"

# Output directory for processed datasets or other local artifacts.
# This is a local path on your PC.
local_processed_data_output_dir = "processed_datasets/"

# Create directories if they don't exist
os.makedirs(local_processed_data_output_dir, exist_ok=True)


# --- 1. Load JSONL Data using Pandas and Create Dataset Object ---
print(f"Loading JSONL data using pandas from: {data_path}...")
raw_data = []
try:
    # Read the JSONL file into a pandas DataFrame
    df = pd.read_json(data_path, lines=True)
    raw_data = df.to_dict(orient='records') # Convert DataFrame to list of dictionaries
    print(f"Raw data loaded successfully using pandas. Number of entries: {len(raw_data)}")
    print("First entry example from raw data:")
    print(raw_data[0]) # Print first entry to verify structure
except FileNotFoundError:
    print(f"Error: The file '{data_path}' was not found.")
    print("Please ensure your 'interview_data.jsonl' is in the 'data/' subdirectory of your project.")
    exit() # Exit if file not found
except Exception as e:
    print(f"Error loading JSONL data with pandas: {e}")
    print("Please ensure the JSONL file is valid.")
    exit()


# --- 2. Define the Instruction Template and Process Raw Data ---
NEWLINE = "\n"

def format_instruction_entry(entry, instruction_type):
    # This function takes a single raw data entry and returns a dictionary
    # with a 'text' key, containing the formatted instruction-response string.

    base_instruction_text = (
        f"Generate a {entry['difficulty']} difficulty {entry['question_type']} "
        f"interview question for a {entry['level']} {entry['role']} "
        f"in the {entry['domain']} domain, specifically on the topic of {entry['topic']}."
    )

    # Context for subsequent requests (hints, explanation, code)
    question_context = (
        f"for a {entry['level']} {entry['role']} "
        f"in the {entry['domain']} domain on topic {entry['topic']}:{NEWLINE}"
        f"Question: {entry['question']}"
    )

    if instruction_type == "question":
        return {
            "text": (
                f"### Instruction:{NEWLINE}"
                f"{base_instruction_text}{NEWLINE}{NEWLINE}"
                f"### Question:{NEWLINE}{entry['question']}"
            )
        }
    elif instruction_type == "hints":
        # Ensure 'hints' key exists and is a list
        hints = entry.get('hints', [])
        if not isinstance(hints, list):
            hints = [str(hints)] # Convert to list if it's a single string/value
        hints_str = NEWLINE.join([f"- {h}" for h in hints])
        return {
            "text": (
                f"### Instruction:{NEWLINE}"
                f"Provide hints for the following {entry['question_type']} question {question_context}{NEWLINE}{NEWLINE}"
                f"### Hints:{NEWLINE}{hints_str}"
            )
        }
    elif instruction_type == "explanation":
        # Ensure 'explanation' key exists and is a string
        explanation = entry.get('explanation', '')
        return {
            "text": (
                f"### Instruction:{NEWLINE}"
                f"Explain the concept or solution for the following {entry['question_type']} question {question_context}{NEWLINE}{NEWLINE}"
                f"### Explanation:{NEWLINE}{explanation}"
            )
        }
    return None # Return None if no valid instruction type or missing data


# Create a list of all individual instruction-response examples by iterating through raw_data
all_formatted_entries = []
for entry in raw_data:
    all_formatted_entries.append(format_instruction_entry(entry, "question"))
    all_formatted_entries.append(format_instruction_entry(entry, "hints"))
    all_formatted_entries.append(format_instruction_entry(entry, "explanation"))
    
# Filter out any None values (e.g., if a solution_code was missing for some entries)
all_formatted_entries = [e for e in all_formatted_entries if e is not None]

# Now, create the Hugging Face Dataset from your list of dictionaries
processed_dataset = Dataset.from_list(all_formatted_entries)

print(f"\nProcessed dataset has {len(processed_dataset)} instruction-response pairs.")
print("Example of a processed entry (first one):")
print(processed_dataset[0]['text'])


# --- 3. Split Data ---
train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"\nTraining dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")


# --- Optional: Save Processed Datasets Locally ---
# This is a good practice to save your prepared datasets so you don't have to
# re-process the raw JSONL every time.
train_dataset_path = os.path.join(local_processed_data_output_dir, "train_dataset")
eval_dataset_path = os.path.join(local_processed_data_output_dir, "eval_dataset")

train_dataset.save_to_disk(train_dataset_path)
eval_dataset.save_to_disk(eval_dataset_path)

print(f"\nProcessed train dataset saved to: {train_dataset_path}")
print(f"Processed eval dataset saved to: {eval_dataset_path}")
