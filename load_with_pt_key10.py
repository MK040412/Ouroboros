import pandas as pd
import torch
import sys
import pyarrow.parquet as pq

print("="*60)
print("Load PT latent data with parquet metadata")
print("="*60)

# Get key index from command line argument
if len(sys.argv) > 1:
    try:
        key_index = int(sys.argv[1])
    except ValueError:
        print(f"\nError: '{sys.argv[1]}' is not a valid integer")
        print("\nUsage: python load_with_pt_key10.py <key_index>")
        print("Example: python load_with_pt_key10.py 10")
        print("\nTo find available indices, run:")
        print("  python list_available_keys.py | grep 'First 20'")
        sys.exit(1)
else:
    print("\nUsage: python load_with_pt_key10.py <key_index>")
    print("Example: python load_with_pt_key10.py 10")
    print("\nNote: Not all PT indices have corresponding parquet data.")
    print("To find available indices, run:")
    print("  python list_available_keys.py")
    sys.exit(1)

# 1. PT 파일에서 해당 index의 key 가져오기
pt_file = "000000-000009.pt"
print(f"\n1. Loading from {pt_file}...")
pt_data = torch.load(pt_file, map_location="cpu")
keys = pt_data['keys'].numpy()
target_key = int(keys[key_index])
print(f"   PT key[{key_index}]: {target_key}")

# 2. Parquet에서 해당 key로 조회
parquet_file = "coyo11m-meta.parquet"
print(f"\n2. Loading from {parquet_file}...")

# Read with filter - key is stored as index column
table = pq.read_table(
    parquet_file,
    filters=[('key', '==', target_key)]
)
df = table.to_pandas()  # key becomes the index automatically

# 3. Key로 행 찾기
print(f"\n{'='*60}")
print(f"Finding row with key={target_key}")
print(f"{'='*60}")

# Check if key exists in index
if len(df) > 0 and target_key in df.index:
    row = df.loc[target_key]
    print(f"\nFound match!")
    print(f"\nURL:\n{row['url']}")
    print(f"\nCaption:\n{row['caption']}")
    print(f"\nCaption LLaVA:\n{row['caption_llava']}")
else:
    if len(df) == 0:
        print(f"No match for key={target_key}")
        print(f"(Filter returned empty result)")
    else:
        print(f"No match for key={target_key}")
        print(f"\nDebug info:")
        print(f"  - Parquet index: {df.index.tolist()}")
