#!/usr/bin/env bash
set -euo pipefail

# Pull logs/results/checkpoints from the shared P5 via EC2 Instance Connect.
#
# Examples:
#   ./ops/pull_p5_artifacts.sh ./p5_artifacts --mode minimal
#   ./ops/pull_p5_artifacts.sh ./p5_artifacts --mode full
#   ./ops/pull_p5_artifacts.sh ./p5_artifacts --instance-id i-abc --ip 1.2.3.4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_out_dir> [--mode minimal|full] [--instance-id <id>] [--ip <ip>] [--remote-repo <path>]"
  exit 2
fi

LOCAL_OUT="$1"
shift

MODE="minimal"
INSTANCE_ID=""
IP=""
REMOTE_REPO="/opt/dlami/nvme/world-modality-vla"
REGION="us-east-2"
TAG_FILTER="rag-research-p5*"
OS_USER="ubuntu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --ip)
      IP="$2"
      shift 2
      ;;
    --remote-repo)
      REMOTE_REPO="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "minimal" && "${MODE}" != "full" ]]; then
  echo "Invalid --mode: ${MODE} (expected minimal|full)"
  exit 2
fi

if [[ -z "${INSTANCE_ID}" || -z "${IP}" ]]; then
  read -r INSTANCE_ID IP AZ <<<"$(aws ec2 describe-instances \
    --region "${REGION}" \
    --filters "Name=tag:Name,Values=${TAG_FILTER}" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[0].[InstanceId,PublicIpAddress,Placement.AvailabilityZone]' \
    --output text)"
else
  AZ="$(aws ec2 describe-instances \
    --region "${REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
    --output text)"
fi

if [[ -z "${INSTANCE_ID}" || -z "${IP}" || -z "${AZ}" || "${INSTANCE_ID}" == "None" || "${IP}" == "None" ]]; then
  echo "Could not resolve a running P5 instance in ${REGION}."
  exit 1
fi

KEY="$(mktemp /tmp/ec2-key-XXXX)"
trap 'rm -f "${KEY}" "${KEY}.pub"' EXIT
rm -f "${KEY}" "${KEY}.pub"
ssh-keygen -t ed25519 -f "${KEY}" -N "" -q >/dev/null

aws ec2-instance-connect send-ssh-public-key \
  --region "${REGION}" \
  --instance-id "${INSTANCE_ID}" \
  --instance-os-user "${OS_USER}" \
  --ssh-public-key "file://${KEY}.pub" \
  --availability-zone "${AZ}" >/dev/null

SSH_CMD=(ssh -i "${KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
REMOTE="${OS_USER}@${IP}"

ts="$(date +%Y%m%d_%H%M%S)"
dest="${LOCAL_OUT%/}/${ts}_${MODE}"
mkdir -p "${dest}"

echo "Instance: ${INSTANCE_ID}"
echo "IP: ${IP}"
echo "AZ: ${AZ}"
echo "Remote repo: ${REMOTE_REPO}"
echo "Local dest: ${dest}"
echo "Mode: ${MODE}"

echo
echo "== Pull logs =="
rsync -av --partial --progress -e "${SSH_CMD[*]}" \
  "${REMOTE}:${REMOTE_REPO%/}/logs/" "${dest}/logs/" \
  --include='*/' \
  --include='*.log' \
  --include='*.pid' \
  --include='*.json' \
  --exclude='*'

echo
echo "== Pull repo artifacts =="
rsync_args=(
  -av --partial --progress
  -e "${SSH_CMD[*]}"
  "${REMOTE}:${REMOTE_REPO%/}/" "${dest}/repo/"
  --include='*/'
  --include='outputs/**'
  --include='eval_libero_results/**'
  --include='RESEARCH_ANALYSIS.md'
  --include='README.md'
  --include='docs/**'
  --include='scripts/**'
  --include='lerobot_policy_world_modality/**'
  --include='world_modality/**'
  --include='pyproject.toml'
  --include='requirements.txt'
  --include='environment.yml'
  --exclude='*'
)

if [[ "${MODE}" == "minimal" ]]; then
  rsync "${rsync_args[@]}" \
    --exclude='**/*.safetensors' \
    --exclude='**/*.bin' \
    --exclude='**/*.pt' \
    --exclude='**/*.ckpt' \
    --exclude='**/*.pth' \
    --exclude='**/*.npy'
else
  rsync "${rsync_args[@]}" \
    --include='cache/HuggingFaceVLA/libero/*.npy'
fi

echo
echo "Done. Artifacts saved under: ${dest}"
