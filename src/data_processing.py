# %% [markdown]
# # Project: Intelligent Complaint Analysis
# ## Module: EDA & Preprocessing
# 
# This script performs the following:
# 1. Loads the raw CFPB complaint data.
# 2. Conducts Exploratory Data Analysis (EDA) on products and narratives.
# 3. Filters the data for specific products and non-empty narratives.
# 4. Cleans the text (normalization, special character removal).
# 5. Saves the processed dataset.

# %% [code]
import pandas as pd
import re
import os
import sys

# Configuration
INPUT_FILE = r'data/raw/complaints.csv/Complaints.csv'
OUTPUT_FILE = r'data/filtered_complaints.csv'
CHUNK_SIZE = 100000

# Regex Patterns
# Matches "Credit card", "Personal loan", "Savings account", "Money transfer"
# "Payday loan, title loan, or personal loan" contains "Personal loan"
# "Checking or savings account" contains "Savings account"
TARGET_PRODUCTS_PATTERN = r"Credit card|Personal loan|Savings account|Money transfer"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Starting processing of {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # Metrics for EDA
    product_counts = {}
    total_rows = 0
    total_narratives = 0
    total_empty_narratives = 0
    narrative_lengths = []
    
    # Setup output file (remove if exists)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    first_chunk = True
    processed_rows = 0
    
    # Iterate with chunks
    try:
        reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(reader):
            sys.stdout.write(f"\rProcessing chunk {i+1}...")
            sys.stdout.flush()
            
            # --- EDA Update ---
            total_rows += len(chunk)
            
            # Count products
            p_counts = chunk['Product'].value_counts()
            for prod, count in p_counts.items():
                product_counts[prod] = product_counts.get(prod, 0) + count
            
            # Count narratives
            narrative_col = 'Consumer complaint narrative'
            
            # Identify non-empty narratives
            non_empty_mask = chunk[narrative_col].notna() & (chunk[narrative_col].str.strip() != "")
            
            chunk_filled = non_empty_mask.sum()
            chunk_empty = len(chunk) - chunk_filled
            
            total_narratives += chunk_filled
            total_empty_narratives += chunk_empty
            
            # Word counts (only for non-empty to save time)
            # We'll take a sample or just calculate all? With regex cleaning it might be slower but let's try raw split
            if chunk_filled > 0:
                lengths = chunk.loc[non_empty_mask, narrative_col].astype(str).str.split().str.len().tolist()
                narrative_lengths.extend(lengths)
            
            # --- Filtering ---
            # 1. Product Filter
            mask_product = chunk['Product'].str.contains(TARGET_PRODUCTS_PATTERN, case=False, na=False)
            
            # 2. Final Selection
            final_mask = mask_product & non_empty_mask
            
            filtered_chunk = chunk[final_mask].copy()
            
            if not filtered_chunk.empty:
                # --- Cleaning ---
                filtered_chunk['cleaned_narrative'] = filtered_chunk[narrative_col].apply(clean_text)
                
                # Save
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                filtered_chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
                first_chunk = False
                processed_rows += len(filtered_chunk)
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return

    print("\n\nProcessing Complete.")
    
    # --- EDA Report Output ---
    report = []
    report.append("--- EDA Report ---")
    report.append(f"Total Rows Processed: {total_rows}")
    report.append(f"Rows Saved to {OUTPUT_FILE}: {processed_rows}")
    
    report.append("\n1. Complaints per Product (Top 20):")
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
    for p, c in sorted_products[:20]:
        report.append(f"   {p}: {c}")
        
    report.append(f"\n2. Narratives vs Empty:")
    report.append(f"   With Narrative: {total_narratives}")
    report.append(f"   Empty: {total_empty_narratives}")
    
    if narrative_lengths:
        s_lengths = pd.Series(narrative_lengths)
        desc = s_lengths.describe()
        report.append("\n3. Narrative Length Statistics (Word Count):")
        report.append(desc.to_string())
        
        short = (s_lengths < 10).sum()
        long_ = (s_lengths > 1000).sum()
        report.append(f"\n4. Extreme Lengths:")
        report.append(f"   Very short (<10 words): {short}")
        report.append(f"   Very long (>1000 words): {long_}")
    else:
        report.append("No narratives found.")
        
    print("\n".join(report))
    
    # Save report to text file
    with open("data/eda_report.txt", "w") as f:
        f.write("\n".join(report))

# %% [code]
if __name__ == "__main__":
    main()
