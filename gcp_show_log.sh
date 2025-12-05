#!/bin/bash
# TPU Worker 로그 출력 스크립트
# Usage: ./gcp_show_log.sh --worker=0

# 기본값
WORKER=all
TPU_NAME="ouroboros"
ZONE="europe-west4-b"
FOLLOW=false

# 인자 파싱
for arg in "$@"; do
    case $arg in
        --worker=*)
            WORKER="${arg#*=}"
            shift
            ;;
        --follow|-f)
            FOLLOW=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --worker=N    Worker 번호 (기본값: 0)"
            echo "  --follow, -f  실시간 로그 추적 (tail -f)"
            echo "  --help, -h    도움말 출력"
            exit 0
            ;;
        *)
            ;;
    esac
done

LOG_FILE="/tmp/train_worker_${WORKER}.log"

echo "=== TPU Worker ${WORKER} Log ==="
echo "TPU: ${TPU_NAME}, Zone: ${ZONE}"
echo "Log file: ${LOG_FILE}"
echo "================================"
echo ""

if [ "$FOLLOW" = true ]; then
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --worker=${WORKER} \
        --command="tail -f ${LOG_FILE}"
else
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --worker=${WORKER} \
        --command="cat ${LOG_FILE}"
fi
