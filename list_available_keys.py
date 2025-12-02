import torch
import pyarrow.parquet as pq

print("="*60)
print("Available PT indices with parquet matches")
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
print(f"\nSearching for matches...")
available_indices = []
for idx, key in enumerate(pt_keys):
    if key in parquet_keys:
        available_indices.append(idx)

print(f"\n{'='*60}")
print(f"Available: {len(available_indices)} indices")
print(f"{'='*60}")

print(f"\nFirst 20 usable indices:")
for idx in available_indices[:20]:
    print(f"  python load_with_pt_key10.py {idx}  # key={pt_keys[idx]}")

print(f"\nAll available indices:")
print(available_indices)
