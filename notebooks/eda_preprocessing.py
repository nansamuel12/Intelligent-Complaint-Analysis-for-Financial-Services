# %% [markdown]
# # Exploratory Data Analysis & Preprocessing
# 
# This notebook analyzes the CFPB complaint data and prepares a filtered, cleaned dataset for the RAG chatbot.
# 
# ## Objectives
# 1. Load the data (Iteratively to handle large file sizes)
# 2. EDA: Inspect products, narratives, and missing values
# 3. Filter: Keep target products and non-empty narratives
# 4. Clean: Normalize text
# 5. Save: `data/filtered_complaints.csv`

# %% [code]
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

# Configuration
INPUT_FILE = r'../data/raw/complaints.csv/Complaints.csv'
OUTPUT_FILE = r'../data/filtered_complaints.csv'
CHUNK_SIZE = 100000

# %% [markdown]
# ## 1. Load and EDA
# Since the file is large (~6GB), we will process it in chunks to gather statistics.

# %% [code]
# Initialize counters
product_counts = {}
total_narratives = 0
total_empty_narratives = 0
narrative_lengths = []

print("Starting EDA scan...")

if os.path.exists(INPUT_FILE):
    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        if i % 10 == 0:
            print(f"Scanning chunk {i}...")
        
        # Count products
        counts = chunk['Product'].value_counts()
        for p, c in counts.items():
            product_counts[p] = product_counts.get(p, 0) + c
            
        # Check narratives
        # Assuming column 'Consumer complaint narrative'
        if 'Consumer complaint narrative' in chunk.columns:
            non_empty = chunk['Consumer complaint narrative'].notna() & (chunk['Consumer complaint narrative'].str.strip() != "")
            total_narratives += non_empty.sum()
            total_empty_narratives += (~non_empty).sum()
            
            # Lengths (word count) - collecting all might be memory intensive, let's collect for first 1M rows or subset
            # to keep it responsive, or just collecting basic stats.
            # We'll collect all for distribution.
            lengths = chunk.loc[non_empty, 'Consumer complaint narrative'].astype(str).str.split().str.len().tolist()
            narrative_lengths.extend(lengths)
            
else:
    print("File not found! Please ensure data is in data/raw/")

print("EDA Scan Complete.")

# %% [markdown]
# ## 2. EDA Results
# ### How many complaints per product?

# %% [code]
# Convert to series for easy plotting/viewing
s_products = pd.Series(product_counts).sort_values(ascending=False)
print(s_products.head(20))

plt.figure(figsize=(10,6))
s_products.head(10).plot(kind='bar')
plt.title("Top 10 Products by Complaint Count")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ### How many complaints have narratives vs empty?

# %% [code]
print(f"With Narratives: {total_narratives}")
print(f"Empty Narratives: {total_empty_narratives}")
print(f"Percentage with Narratives: {total_narratives / (total_narratives + total_empty_narratives) * 100:.2f}%")

# %% [markdown]
# ### How long are complaint narratives (word count)?
# ### Are there very short or extremely long narratives?

# %% [code]
if narrative_lengths:
    s_lengths = pd.Series(narrative_lengths)
    print(s_lengths.describe())
    
    plt.figure(figsize=(10,4))
    plt.hist(s_lengths, bins=50, range=(0, 1000), edgecolor='k', alpha=0.7)
    plt.title("Distribution of Narrative Lengths (0-1000 words)")
    plt.xlabel("Word Count")
    plt.show()
    
    print(f"Very short (<10 words): {(s_lengths < 10).sum()}")
    print(f"Extremely long (>1000 words): {(s_lengths > 1000).sum()}")
else:
    print("No narratives data collected.")

# %% [markdown]
# ## 3. Filtering and Cleaning
# **Criteria:**
# - **Keep Products:** "Credit card", "Personal loan", "Savings account", "Money transfer"
# - **Remove:** Empty narratives
# - **Clean:** Lowercase, remove special chars, normalize whitespace

# %% [code]
# Target Products Regex
# Matches "Credit card", "Personal loan", "Savings account", "Money transfer"
TARGET_PRODUCTS_PATTERN = r"Credit card|Personal loan|Savings account|Money transfer"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Starting Filtering and Cleaning...")

# Reset file reader
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_chunk = True
saved_rows = 0

reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
for i, chunk in enumerate(reader):
    if i % 10 == 0:
        print(f"Processing chunk {i}...")

    # Filter Product
    mask_product = chunk['Product'].str.contains(TARGET_PRODUCTS_PATTERN, case=False, na=False)
    
    # Filter Narrative
    mask_narrative = chunk['Consumer complaint narrative'].notna() & (chunk['Consumer complaint narrative'].str.strip() != "")
    
    filtered_df = chunk[mask_product & mask_narrative].copy()
    
    if not filtered_df.empty:
        # Clean
        filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)
        
        # Save
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        filtered_df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
        first_chunk = False
        saved_rows += len(filtered_df)

print(f"Processing Complete. Saved {saved_rows} rows to {OUTPUT_FILE}")
