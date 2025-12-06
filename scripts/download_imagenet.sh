#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ImageNet Download and Distribution Script for TPU Pods
#
# Usage:
#   # Download ImageNet to GCS (run once on any machine with ImageNet)
#   ./download_imagenet.sh upload /path/to/imagenet gs://bucket/imagenet
#
#   # Distribute to TPU workers (run on each TPU worker)
#   ./download_imagenet.sh download gs://bucket/imagenet ~/imagenet
#
#   # Full setup on TPU pod (coordinator runs on worker 0)
#   ./download_imagenet.sh setup gs://bucket/imagenet ~/imagenet
# ==============================================================================

LOG_PREFIX="[imagenet]"

log() {
    echo "${LOG_PREFIX} $*"
}

error() {
    echo "${LOG_PREFIX} ERROR: $*" >&2
    exit 1
}

# ==============================================================================
# Upload ImageNet to GCS
# ==============================================================================
upload_to_gcs() {
    local src_dir="$1"
    local gcs_dest="$2"

    if [[ ! -d "$src_dir" ]]; then
        error "Source directory not found: $src_dir"
    fi

    log "Uploading ImageNet to GCS..."
    log "  Source: $src_dir"
    log "  Destination: $gcs_dest"

    # Upload with parallel transfers
    gsutil -m cp -r "${src_dir}/train" "${gcs_dest}/"

    # Upload validation set if exists
    if [[ -d "${src_dir}/val" ]]; then
        gsutil -m cp -r "${src_dir}/val" "${gcs_dest}/"
    fi

    log "Upload complete!"

    # Show summary
    log "Contents:"
    gsutil ls "${gcs_dest}/"
}

# ==============================================================================
# Download ImageNet from GCS to local disk
# ==============================================================================
download_from_gcs() {
    local gcs_src="$1"
    local local_dest="$2"

    log "Downloading ImageNet from GCS..."
    log "  Source: $gcs_src"
    log "  Destination: $local_dest"

    # Create destination directory
    mkdir -p "$local_dest"

    # Download with parallel transfers
    # Using rsync for resumable downloads
    gsutil -m rsync -r "${gcs_src}/train" "${local_dest}/train"

    # Show size
    local size=$(du -sh "$local_dest" 2>/dev/null | cut -f1)
    log "Download complete! Size: $size"
}

# ==============================================================================
# Setup for TPU Pod (all workers)
# ==============================================================================
setup_tpu_pod() {
    local gcs_src="$1"
    local local_dest="$2"

    # Get worker info
    local worker_id="${JAX_PROCESS_INDEX:-0}"
    local num_workers="${JAX_NUM_PROCESSES:-1}"

    log "Setting up ImageNet for TPU Pod"
    log "  Worker: $worker_id / $num_workers"
    log "  GCS source: $gcs_src"
    log "  Local dest: $local_dest"

    # Check if already downloaded
    if [[ -d "${local_dest}/train" ]]; then
        local num_classes=$(ls -1 "${local_dest}/train" 2>/dev/null | wc -l)
        if [[ "$num_classes" -ge 1000 ]]; then
            log "ImageNet already present: $num_classes classes found"
            return 0
        fi
    fi

    # Download full dataset (each worker downloads independently)
    # This is simpler than sharding for <100GB datasets
    download_from_gcs "$gcs_src" "$local_dest"
}

# ==============================================================================
# Shard ImageNet across workers (memory-efficient alternative)
# ==============================================================================
shard_download() {
    local gcs_src="$1"
    local local_dest="$2"

    local worker_id="${JAX_PROCESS_INDEX:-0}"
    local num_workers="${JAX_NUM_PROCESSES:-8}"

    log "Sharded download for worker $worker_id / $num_workers"

    # Get list of all synset directories
    local synsets=$(gsutil ls "${gcs_src}/train/" | grep '/$' | sed 's|.*/||; s|/$||')
    local synset_array=($synsets)
    local total_synsets=${#synset_array[@]}

    log "  Total synsets: $total_synsets"

    # Calculate which synsets this worker should download
    local start_idx=$((worker_id * total_synsets / num_workers))
    local end_idx=$(((worker_id + 1) * total_synsets / num_workers))

    log "  Synset range: $start_idx to $end_idx"

    # Create local directory
    mkdir -p "${local_dest}/train"

    # Download assigned synsets
    local count=0
    for ((i=start_idx; i<end_idx; i++)); do
        local synset="${synset_array[$i]}"
        if [[ -n "$synset" ]]; then
            gsutil -m cp -r "${gcs_src}/train/${synset}" "${local_dest}/train/" 2>/dev/null || true
            count=$((count + 1))
            if (( count % 50 == 0 )); then
                log "  Downloaded $count / $((end_idx - start_idx)) synsets"
            fi
        fi
    done

    log "Shard download complete: $count synsets"
}

# ==============================================================================
# Create ImageNet classes JSON
# ==============================================================================
create_classes_json() {
    local imagenet_dir="$1"
    local output_file="${2:-imagenet_classes.json}"

    log "Creating imagenet_classes.json..."

    # Generate JSON from directory structure
    python3 << EOF
import json
import os
from pathlib import Path

imagenet_dir = Path("$imagenet_dir")
train_dir = imagenet_dir / "train"

if not train_dir.exists():
    print(f"Train directory not found: {train_dir}")
    exit(1)

# Get all synset directories
synsets = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

# Create mapping (synset -> synset for now, can be enhanced with labels)
classes = {s: s for s in synsets}

output_file = "$output_file"
with open(output_file, 'w') as f:
    json.dump(classes, f, indent=2)

print(f"Created {output_file} with {len(classes)} classes")
EOF
}

# ==============================================================================
# Main
# ==============================================================================
case "${1:-help}" in
    upload)
        if [[ $# -lt 3 ]]; then
            error "Usage: $0 upload <local_imagenet_dir> <gs://bucket/path>"
        fi
        upload_to_gcs "$2" "$3"
        ;;

    download)
        if [[ $# -lt 3 ]]; then
            error "Usage: $0 download <gs://bucket/path> <local_dest>"
        fi
        download_from_gcs "$2" "$3"
        ;;

    setup)
        if [[ $# -lt 3 ]]; then
            error "Usage: $0 setup <gs://bucket/path> <local_dest>"
        fi
        setup_tpu_pod "$2" "$3"
        ;;

    shard)
        if [[ $# -lt 3 ]]; then
            error "Usage: $0 shard <gs://bucket/path> <local_dest>"
        fi
        shard_download "$2" "$3"
        ;;

    classes)
        if [[ $# -lt 2 ]]; then
            error "Usage: $0 classes <imagenet_dir> [output_file]"
        fi
        create_classes_json "$2" "${3:-imagenet_classes.json}"
        ;;

    *)
        echo "ImageNet Download and Distribution Script"
        echo ""
        echo "Usage:"
        echo "  $0 upload <local_dir> <gs://bucket/path>  - Upload ImageNet to GCS"
        echo "  $0 download <gs://bucket/path> <local>    - Download from GCS"
        echo "  $0 setup <gs://bucket/path> <local>       - Setup for TPU pod"
        echo "  $0 shard <gs://bucket/path> <local>       - Sharded download"
        echo "  $0 classes <imagenet_dir> [output]        - Create classes JSON"
        echo ""
        echo "Examples:"
        echo "  # Upload ImageNet to GCS (run once)"
        echo "  $0 upload /data/imagenet gs://my-bucket/imagenet"
        echo ""
        echo "  # Download on each TPU worker"
        echo "  $0 download gs://my-bucket/imagenet ~/imagenet"
        echo ""
        echo "  # Or use sharded download (each worker gets subset)"
        echo "  $0 shard gs://my-bucket/imagenet ~/imagenet"
        ;;
esac
