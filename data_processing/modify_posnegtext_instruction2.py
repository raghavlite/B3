import os
import pandas as pd
import re

# Define core dataset names for classification, VQA, and retrieval tasks
classification_datasets = ["ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"]
vqa_datasets = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
retrieval_datasets = ["MSCOCO_i2t", "VisualNews_i2t"]

base_input_dir = "./MMEB-train/"  # Directory where the original folders are
base_output_dir = "../VLM2Vec-pro/MMEB-train2/"  # Directory where modified files will be saved

# Ensure output base directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Function to match folder names with core dataset names
def match_dataset_category(folder_name):
    for dataset in classification_datasets:
        if dataset in folder_name:
            return "classification"
    for dataset in vqa_datasets:
        if dataset in folder_name:
            return "vqa"
    for dataset in retrieval_datasets:
        if dataset in folder_name:
            return "retrieval"
    return None  # If no match is found

# Process all folders dynamically
for folder in os.listdir(base_input_dir):
    folder_path = os.path.join(base_input_dir, folder)
    if not os.path.isdir(folder_path):
        continue  # Skip files, only process directories
    
    category = match_dataset_category(folder)
    if folder=="images":
        print(f"Skipping unknown dataset: {folder}")
        continue  # Skip if not recognized

    output_folder_path = os.path.join(base_output_dir, folder)
    if os.path.exists(output_folder_path):
        print(f"Skipping already processed dataset: {folder}")
        continue  # Skip if already processed

    output_folder_path = os.path.join(base_output_dir, folder)
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Get all parquet files in the folder
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    
    # Ensure only one parquet file is present
    assert len(parquet_files) == 1, f"More than one or no Parquet file found in {folder_path}"
    
    parquet_file_name = parquet_files[0]
    input_parquet_path = os.path.join(folder_path, parquet_file_name)
    output_parquet_path = os.path.join(output_folder_path, parquet_file_name)
    
    # Load the Parquet file
    df = pd.read_parquet(input_parquet_path)
    
    # Modify text fields based on category
    if category is None or "HN" not in folder:
        print(f"Copying unknown dataset: {folder}")
    elif category == "classification":
        df["pos_text"] = "Represent the class label: " + df["pos_text"]
        df["neg_text"] = df["neg_text"].apply(lambda neg_list: ["Represent the class label: " + text for text in neg_list])
    elif category == "vqa":
        df["pos_text"] = "Represent the answer: " + df["pos_text"]
        df["neg_text"] = df["neg_text"].apply(lambda neg_list: ["Represent the answer: " + text for text in neg_list])
    elif category == "retrieval":
        df["pos_text"] = "Represent the image caption: " + df["pos_text"]
        df["neg_text"] = df["neg_text"].apply(lambda neg_list: ["Represent the image caption: " + text for text in neg_list])
    
    # Save the modified dataframe to the output directory
    df.to_parquet(output_parquet_path, index=False)
    print(f">>>>>>>>>>>>>>>>>>> Processed and saved: {output_parquet_path}", flush=True)

print("All folders processed successfully.")