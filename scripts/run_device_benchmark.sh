#!/usr/bin/env bash
#
# iPhone 実機で XCUITest (testCaptureAllCombinations) を走らせ、
# 12 通りの timing/mel/wav をホストに吸い出してメトリクスを集計する。
#
# 使い方:
#   ./scripts/run_device_benchmark.sh                    # 接続中の iPhone を1台自動検出
#   ./scripts/run_device_benchmark.sh --device <名前|UDID|identifier>
#
# 前提:
#   - Xcode 15+ (xcrun devicectl)
#   - iPhone が USB 接続され Xcode と paired 済み
#   - PronounSE/venv が用意済み (集計に使用)
#
# 補足:
#   xcodebuild と devicectl は同じデバイスを別 ID 体系で扱う:
#     - xcodebuild の id=  : iPhone のハードウェア UDID (例: 00008110-0008396C26C3401E)
#     - devicectl の --device: CoreDevice identifier (例: 44D024A4-DF7F-...)
#   両方が受けつける「デバイス名」(例: "iPhone (3)") に統一して指定する。
#   --device に何を渡しても、内部で名前に正規化する。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${REPO_ROOT}/ios/CoreMLAudioApp/CoreMLAudioApp.xcodeproj"
SCHEME="CoreMLAudioApp"
TEST_TARGET="CoreMLAudioAppUITests/CoreMLAudioAppUITests/testCaptureAllCombinations"
PYTHON="${REPO_ROOT}/PronounSE/venv/bin/python"
EXTRACT_SCRIPT="${REPO_ROOT}/scripts/extract_ui_test_results.sh"
AGGREGATE_SCRIPT="${REPO_ROOT}/scripts/aggregate_metrics.py"
VIEW_MEL_SCRIPT="${REPO_ROOT}/scripts/view_mel.py"
VIEW_WAVEFORM_SCRIPT="${REPO_ROOT}/scripts/view_waveform.py"

DEVICE_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE_ARG="$2"
            shift 2
            ;;
        --device=*)
            DEVICE_ARG="${1#--device=}"
            shift
            ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail$/p' "$0" | grep -E '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "不明な引数: $1" >&2
            exit 1
            ;;
    esac
done

# devicectl の JSON 出力からデバイス名を解決する。
# 引数が空なら接続中 (tunnelState=connected) のデバイスを 1 台自動検出。
# 引数があればそれを name / udid / identifier のいずれかに照合してデバイス名に正規化。
resolve_device_name() {
    local query="$1"
    local tmp_json
    tmp_json="$(mktemp)"
    xcrun devicectl list devices --json-output "${tmp_json}" >/dev/null 2>&1 || true
    /usr/bin/env python3 - "${query}" "${tmp_json}" <<'PY'
import json
import sys

query = sys.argv[1]
path = sys.argv[2]
try:
    with open(path) as f:
        data = json.load(f)
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
            print(f["name"])
            sys.exit(0)
    sys.exit(2)
else:
    connected = [d for d in devices if is_connected(d)]
    if len(connected) == 1:
        print(fields(connected[0])["name"])
        sys.exit(0)
    sys.exit(3)
PY
    local rc=$?
    rm -f "${tmp_json}"
    return ${rc}
}

DEVICE_NAME=""
if [[ -z "${DEVICE_ARG}" ]]; then
    if ! DEVICE_NAME="$(resolve_device_name "")"; then
        echo "エラー: 接続中の実機が 1 台に絞れません" >&2
        echo "  xcrun devicectl list devices で connected 状態のデバイスを確認し、" >&2
        echo "  --device <名前|UDID|identifier> で明示指定してください" >&2
        exit 1
    fi
    echo "自動検出した device: ${DEVICE_NAME}"
else
    if ! DEVICE_NAME="$(resolve_device_name "${DEVICE_ARG}")"; then
        echo "エラー: --device '${DEVICE_ARG}' に該当するデバイスが見つかりません" >&2
        echo "  xcrun devicectl list devices で表示される Name / UDID / Identifier を渡してください" >&2
        exit 1
    fi
    echo "指定された device: ${DEVICE_ARG} -> ${DEVICE_NAME}"
fi

echo ""
echo "===== 1/4: xcodebuild test (実機) ====="
xcodebuild test \
    -project "${PROJECT}" \
    -scheme "${SCHEME}" \
    -destination "platform=iOS,name=${DEVICE_NAME}" \
    -only-testing:"${TEST_TARGET}"

echo ""
echo "===== 2/4: Documents/Result を吸い出し ====="
"${EXTRACT_SCRIPT}" --device "${DEVICE_NAME}"

echo ""
echo "===== 3/4: メトリクス集計 ====="
if [[ ! -x "${PYTHON}" ]]; then
    echo "警告: ${PYTHON} が見つかりません。集計と図生成はスキップします。" >&2
    echo "      手動で aggregate_metrics.py / view_mel.py / view_waveform.py を回してください。" >&2
    exit 0
fi
# CSV と PNG は result/ の外 (metrics/, figures/) にタイムスタンプ付きで保存。
# result/ は次回の extract で rsync --delete されるが metrics/ figures/ は残る。
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEVICE_SLUG="$(printf '%s' "${DEVICE_NAME}" | tr ' ()' '___' | tr -s '_' | sed 's/_$//')"
CSV_PATH="${REPO_ROOT}/metrics/metrics_${DEVICE_SLUG}_${TIMESTAMP}.csv"
FIGURE_DIR="${REPO_ROOT}/figures/${DEVICE_SLUG}_${TIMESTAMP}"
"${PYTHON}" "${AGGREGATE_SCRIPT}" --csv "${CSV_PATH}"

echo ""
echo "===== 4/4: 図 (mel スペクトログラム / 波形) を生成 ====="
mkdir -p "${FIGURE_DIR}"
"${PYTHON}" "${VIEW_MEL_SCRIPT}" --save "${FIGURE_DIR}/mel_grid.png" >/dev/null
"${PYTHON}" "${VIEW_WAVEFORM_SCRIPT}" --save "${FIGURE_DIR}/waveform_grid.png" >/dev/null
echo "保存先: ${FIGURE_DIR}"
ls -la "${FIGURE_DIR}"
