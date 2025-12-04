#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Pre-compute Embeddings 진행 상황 모니터링
# - 4 Workers × 4 TPU chips (pmap 병렬화)
# - Output: gs://rdy-tpu-data-2025/.../latents-3crop-emb/
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=4
TPU_CHIPS_PER_WORKER=4
BATCH_SIZE=2048

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

get_worker_log() {
    local worker_id=$1
    if [[ $worker_id -eq 0 ]]; then
        tail -150 /tmp/precompute_worker_0.log 2>/dev/null | tr -d '\0' | tr '\r' '\n' || echo ""
    else
        /snap/bin/gcloud compute tpus tpu-vm ssh "$INSTANCE" \
            --zone="$ZONE" \
            --worker="$worker_id" \
            --command="tail -150 /tmp/precompute_worker_$worker_id.log 2>/dev/null" 2>/dev/null | tr -d '\0' | tr '\r' '\n' || echo ""
    fi
}

format_time() {
    local seconds=$1
    if [[ $seconds -lt 0 ]]; then seconds=0; fi
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm" $hours $minutes
    else
        printf "%dm" $minutes
    fi
}

print_table() {
    echo "=============================================================================="
    echo "  Pre-compute Embeddings Progress Monitor (pmap: ${TPU_CHIPS_PER_WORKER} chips/worker)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="
    echo ""
    printf "%-8s | %-10s | %-18s | %-10s | %-8s | %-10s | %-12s\n" \
        "Worker" "File" "Filename" "Samples" "Progress" "Speed" "Status"
    echo "---------|------------|--------------------|-----------|---------:|------------|-------------"

    local total_files_done=0
    local total_files=0
    local total_speed_sum=0
    local speed_count=0

    for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
        local log_content=$(get_worker_log $worker_id)

        # 기본값
        local file_current="-"
        local file_total="-"
        local filename="-"
        local samples="-"
        local progress="-"
        local speed="-"
        local status="init"

        if [[ -z "$log_content" ]]; then
            status="no_log"
        else
            # File 정보 파싱: "--- Worker X: File Y/Z ---"
            local file_line=$(echo "$log_content" | grep "Worker $worker_id: File" | tail -1 || true)
            if [[ -n "$file_line" ]]; then
                file_current=$(echo "$file_line" | sed -n 's/.*File \([0-9]*\)\/\([0-9]*\).*/\1/p' || echo "-")
                file_total=$(echo "$file_line" | sed -n 's/.*File \([0-9]*\)\/\([0-9]*\).*/\2/p' || echo "-")
            fi

            # 현재 파일명 파싱
            local fname_match=$(echo "$log_content" | grep -oE '\[[0-9]+-[0-9]+\.pt\]' | tail -1 || true)
            if [[ -n "$fname_match" ]]; then
                filename=$(echo "$fname_match" | tr -d '[]')
            fi

            # Samples 파싱
            local samples_match=$(echo "$log_content" | grep -oE '[0-9,]+ samples' | tail -1 || true)
            if [[ -n "$samples_match" ]]; then
                samples=$(echo "$samples_match" | sed 's/ samples//')
            fi

            # tqdm 진행률 파싱
            local tqdm_line=$(echo "$log_content" | grep "TPU Encoding" | tail -1 || true)
            if [[ -n "$tqdm_line" ]]; then
                progress=$(echo "$tqdm_line" | grep -oE '[0-9]+%' | tail -1 || echo "-")
                speed=$(echo "$tqdm_line" | grep -oE '[0-9]+\.[0-9]+s/it' | tail -1 || echo "-")
                status="encoding"
            fi

            # 상태 확인 (우선순위 순)
            if echo "$log_content" | grep -q "All files processed"; then
                status="completed"
                file_current="$file_total"
                progress="100%"
            elif echo "$log_content" | grep -q "Error:"; then
                status="error"
            elif [[ -n "$tqdm_line" ]]; then
                status="encoding"
            elif echo "$log_content" | grep -q "Uploading"; then
                local last_line=$(echo "$log_content" | grep -E "(Uploading|Done)" | tail -1 || true)
                if [[ "$last_line" == *"Uploading"* ]]; then
                    status="uploading"
                fi
            elif echo "$log_content" | grep -q "Saving"; then
                status="saving"
            elif echo "$log_content" | grep -q "Computing embeddings"; then
                status="computing"
            elif echo "$log_content" | grep -q "Loading PT"; then
                status="loading"
            elif echo "$log_content" | grep -q "Downloading"; then
                status="download"
            elif echo "$log_content" | grep -q "pmap ready"; then
                status="pmap_ready"
            elif echo "$log_content" | grep -q "Model loaded"; then
                status="model_ok"
            elif echo "$log_content" | grep -q "Loading bert"; then
                status="load_bert"
            fi
        fi

        # 색상 설정
        local status_color=$NC
        case $status in
            "encoding") status_color=$GREEN ;;
            "completed") status_color=$BLUE ;;
            "download"|"loading"|"uploading"|"saving"|"computing") status_color=$YELLOW ;;
            "pmap_ready"|"model_ok"|"load_bert"|"init") status_color=$CYAN ;;
            "no_log"|"error") status_color=$RED ;;
            *) status_color=$NC ;;
        esac

        # 파일 진행률 표시
        local file_progress="-"
        if [[ "$file_current" != "-" ]] && [[ "$file_total" != "-" ]] && [[ -n "$file_current" ]] && [[ -n "$file_total" ]]; then
            file_progress="$file_current/$file_total"
            local done_files=$(awk "BEGIN {print int($file_current - 1)}")
            if [[ $done_files -lt 0 ]]; then done_files=0; fi
            total_files_done=$(awk "BEGIN {print $total_files_done + $done_files}")
            total_files=$((total_files + file_total))

            # 현재 파일 진행 중이면 부분 추가
            if [[ "$progress" != "-" ]] && [[ "$status" == "encoding" ]]; then
                local pct=$(echo "$progress" | tr -d '%')
                if [[ -n "$pct" ]] && [[ "$pct" =~ ^[0-9]+$ ]]; then
                    total_files_done=$(awk "BEGIN {printf \"%.2f\", $total_files_done + ($pct/100)}")
                fi
            fi
        fi

        # 속도 집계
        if [[ "$speed" != "-" ]] && [[ -n "$speed" ]]; then
            local speed_val=$(echo "$speed" | sed 's/s\/it//')
            if [[ -n "$speed_val" ]] && [[ "$speed_val" =~ ^[0-9.]+$ ]]; then
                total_speed_sum=$(awk "BEGIN {printf \"%.2f\", $total_speed_sum + $speed_val}")
                speed_count=$((speed_count + 1))
            fi
        fi

        printf "%-8s | %-10s | %-18s | %10s | %8s | %10s | ${status_color}%-12s${NC}\n" \
            "Worker $worker_id" "$file_progress" "${filename:0:18}" "$samples" "$progress" "$speed" "$status"
    done

    echo "=============================================================================="

    # 전체 진행률 및 예상 시간 계산
    if [[ $total_files -gt 0 ]]; then
        local overall_pct=$(awk "BEGIN {printf \"%.1f\", ($total_files_done / $total_files) * 100}")

        echo ""
        echo "Overall Progress: $overall_pct% ($total_files_done / $total_files files)"

        # 예상 시간 계산
        if [[ $speed_count -gt 0 ]]; then
            local avg_speed=$(awk "BEGIN {printf \"%.2f\", $total_speed_sum / $speed_count}")
            # 평균 배치 수: 120k samples / 2048 batch_size ≈ 59 batches
            local avg_batches=59
            local avg_file_time=$(awk "BEGIN {printf \"%.0f\", $avg_speed * $avg_batches}")
            local remaining_files=$(awk "BEGIN {printf \"%.0f\", $total_files - $total_files_done}")
            # 4 workers 병렬이므로 /4
            local remaining_seconds=$(awk "BEGIN {printf \"%.0f\", ($remaining_files * $avg_file_time) / $NUM_WORKERS}")

            echo "Average Speed: ${avg_speed}s/batch (${TPU_CHIPS_PER_WORKER} chips pmap)"
            echo "Est. per File: ~$(format_time $avg_file_time) | Est. Remaining: ~$(format_time $remaining_seconds)"
        fi
    fi

    echo ""
}

# 한 번만 실행
print_table
