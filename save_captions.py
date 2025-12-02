import torch
import pyarrow.parquet as pq
from datetime import datetime

print("="*60)
print("Save captions for latent indices")
print("="*60)

# Load PT keys
pt_file = "000000-000009.pt"
print(f"\nLoading {pt_file}...")
pt_data = torch.load(pt_file, map_location="cpu")
pt_keys = pt_data['keys'].numpy()

# Get parquet keys
parquet_file = "coyo11m-meta.parquet"
print(f"Getting parquet keys...")
table = pq.read_table(parquet_file, columns=['key'])
parquet_keys = set(table['key'].to_pylist())

# Find available indices
print(f"\nFinding available indices...")
available_indices = []
for idx, key in enumerate(pt_keys):
    if key in parquet_keys:
        available_indices.append(idx)

print(f"Found {len(available_indices)} available indices")

# Get first 10 available indices
selected_indices = available_indices[:10]
print(f"\nSelected first 10 indices: {selected_indices}")

# Get captions and save to file
output_file = "captions_10.txt"
print(f"\nSaving captions to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total: 10 captions\n")
    f.write("="*60 + "\n\n")
    
    for i, idx in enumerate(selected_indices, 1):
        key = pt_keys[idx]
        print(f"  [{i}/10] Processing PT[{idx}] (key={key})...")
        
        # Read this specific key from parquet
        table = pq.read_table(
            parquet_file,
            columns=['caption', 'caption_llava', 'url'],
            filters=[('key', '==', int(key))]
        )
        
        if len(table) > 0:
            df = table.to_pandas()
            row = df.iloc[0]
            
            f.write(f"[{i}] PT Index: {idx}, Key: {key}\n")
            f.write(f"URL: {row['url']}\n")
            f.write(f"\nCaption:\n{row['caption']}\n")
            f.write(f"\nCaption LLaVA:\n{row['caption_llava']}\n")
            f.write("\n" + "-"*60 + "\n\n")

print(f"\n✓ Saved to {output_file}")
