#!/usr/bin/env bash
set -euo pipefail

# 1. 기본값 설정 (Hardcoded Defaults)
ZONE="europe-west4-b"
WORKER="all"
INSTANCE="ouroboros"
LOG_PREFIX="[gcp_run]"

log() {
  echo "${LOG_PREFIX} $*"
}

# 2. 인자 파싱 (옵션 처리)
# 옵션(-로 시작하는 것)이 나올 때까지 루프를 돌며 변수를 덮어씁니다.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --worker)
      WORKER="$2"
      shift 2
      ;;
    --instance)
      INSTANCE="$2"
      shift 2
      ;;
    --) # '--'를 만나면 옵션 파싱 강제 종료 (명령어 시작)
      shift
      break
      ;;
    -*) # 알 수 없는 옵션 처리
      echo "Error: Unknown option $1" >&2
      exit 1
      ;;
    *) # 옵션이 아닌 인자가 나오면 루프 종료 (여기서부터 명령어라고 간주)
      break
      ;;
  esac
done

# 3. 명령어 존재 여부 확인
# 옵션을 다 걷어내고 남은 인자가 없으면 에러
if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") [--zone Z] [--worker W] [--instance I] <command>" >&2
  exit 1
fi

# 4. 남은 인자들을 모두 합쳐서 실행할 명령어로 저장
COMMAND="$*"

log "zone=$ZONE worker=$WORKER instance=$INSTANCE"
log "command: $COMMAND"

# 5. gcloud 실행
gcloud compute tpus tpu-vm ssh "$INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="$COMMAND"

log "completed"