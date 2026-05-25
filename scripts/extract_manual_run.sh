#!/usr/bin/env bash
#
# iPhone 実機の Documents/Result/ を吸い出して
# audio/<デバイス>_manual_<日時>/ に退避する。
# auto-test の結果を上書きせず、手動操作の wav を比較用に保存するためのスクリプト。
#
# 使い方:
#   ./scripts/extract_manual_run.sh                      # 接続中の iPhone を自動検出
#   ./scripts/extract_manual_run.sh --device "iPhone (3)"
#   ./scripts/extract_manual_run.sh --label cpuAndGPU_v1 # audio/ ディレクトリ名に追加ラベル
#
# 前提:
#   - iPhone を USB 接続済み
#   - アプリで 1 回以上手動合成して Documents/Result/{mel,timing,*.wav} がある状態

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTRACT_SCRIPT="${REPO_ROOT}/scripts/extract_ui_test_results.sh"

DEVICE_ARG=""
LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE_ARG="$2"; shift 2 ;;
        --device=*) DEVICE_ARG="${1#--device=}"; shift ;;
        --label)    LABEL="$2"; shift 2 ;;
        --label=*)  LABEL="${1#--label=}"; shift ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail$/p' "$0" | grep -E '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "不明な引数: $1" >&2; exit 1 ;;
    esac
done

# デバイス名解決 (run_device_benchmark.sh と同じロジック)
resolve_device_name() {
    local query="$1"
    local tmp_json
    tmp_json="$(mktemp)"
    xcrun devicectl list devices --json-output "${tmp_json}" >/dev/null 2>&1 || true
    /usr/bin/env python3 - "${query}" "${tmp_json}" <<'PY'
import json, sys
query, path = sys.argv[1], sys.argv[2]
try:
    with open(path) as f: data = json.load(f)
except Exception:
    sys.exit(1)
devices = data.get("result", {}).get("devices", [])
def is_connected(d):
    return d.get("connectionProperties", {}).get("tunnelState") == "connected"
def fields(d):
    return {
        "name": d.get("deviceProperties", {}).get("name", ""),
        "udid": d.get("hardwareProperties", {}).get("udid", ""),
        "identifier": d.get("identifier", ""),
    }
if query:
    for d in devices:
        f = fields(d)
        if query in (f["name"], f["udid"], f["identifier"]):
            print(f["name"]); sys.exit(0)
    sys.exit(2)
connected = [d for d in devices if is_connected(d)]
if len(connected) == 1:
    print(fields(connected[0])["name"]); sys.exit(0)
sys.exit(3)
PY
}

DEVICE_NAME="$(resolve_device_name "${DEVICE_ARG}")" || {
    echo "デバイスの自動検出に失敗しました。--device で明示してください。" >&2
    exit 1
}
echo "device: ${DEVICE_NAME}"

echo ""
echo "===== 1/2: iPhone から Documents/Result/ を吸い出す ====="
"${EXTRACT_SCRIPT}" --device "${DEVICE_NAME}"

echo ""
echo "===== 2/2: wav と mel/timing を audio/ に退避 ====="
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEVICE_SLUG="$(printf '%s' "${DEVICE_NAME}" | tr ' ()' '___' | tr -s '_' | sed 's/_$//')"
SUFFIX="manual"
if [[ -n "${LABEL}" ]]; then
    SUFFIX="manual_${LABEL}"
fi
DEST_DIR="${REPO_ROOT}/audio/${DEVICE_SLUG}_${SUFFIX}_${TIMESTAMP}"
mkdir -p "${DEST_DIR}"
cp "${REPO_ROOT}"/result/output_*.wav "${DEST_DIR}/" 2>/dev/null || true
# mel/timing も一緒に退避して後で再集計できるようにする
if [[ -d "${REPO_ROOT}/result/mel" ]]; then
    cp -R "${REPO_ROOT}/result/mel" "${DEST_DIR}/"
fi
if [[ -d "${REPO_ROOT}/result/timing" ]]; then
    cp -R "${REPO_ROOT}/result/timing" "${DEST_DIR}/"
fi
# debug/ は CMLA_DEBUG_SNAPSHOT=1 のテスト用 (本番ランでは存在しないので skip OK)
if [[ -d "${REPO_ROOT}/result/debug" ]]; then
    cp -R "${REPO_ROOT}/result/debug" "${DEST_DIR}/"
fi
echo "保存先: ${DEST_DIR}"
ls -la "${DEST_DIR}"
