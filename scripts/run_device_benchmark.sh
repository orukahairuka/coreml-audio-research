#!/usr/bin/env bash
#
# iPhone 実機で XCUITest (testCaptureAllCombinations) を走らせ、
# 12 通りの timing/mel/wav をホストに吸い出してメトリクスを集計する。
#
# 使い方:
#   ./scripts/run_device_benchmark.sh                    # 接続中の iPhone を1台自動検出
#   ./scripts/run_device_benchmark.sh --device <UDID>    # 明示指定
#   ./scripts/run_device_benchmark.sh --device "iPhone (3)"
#
# 前提:
#   - Xcode 15+ (xcrun devicectl)
#   - iPhone が USB 接続され Xcode と paired 済み
#   - PronounSE/venv が用意済み (集計に使用)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${REPO_ROOT}/ios/CoreMLAudioApp/CoreMLAudioApp.xcodeproj"
SCHEME="CoreMLAudioApp"
TEST_TARGET="CoreMLAudioAppUITests/CoreMLAudioAppUITests/testCaptureAllCombinations"
PYTHON="${REPO_ROOT}/PronounSE/venv/bin/python"
EXTRACT_SCRIPT="${REPO_ROOT}/scripts/extract_ui_test_results.sh"
AGGREGATE_SCRIPT="${REPO_ROOT}/scripts/aggregate_metrics.py"

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

resolve_device() {
    # devicectl list devices の出力を行ごとに走査し、
    # 「connected」状態の最初の行を Identifier (UDID) と Name を返す。
    # 出力形式: Name | Hostname | Identifier | State | Model
    local output
    output=$(xcrun devicectl list devices 2>/dev/null || true)

    awk '
        /^[[:space:]]*$/ { next }
        /^Name/ { next }
        /^---/ { next }
        # State 列に "connected" を含む最初の行を採用
        /connected/ {
            # Identifier は UUID 形式 (8-4-4-4-12)
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$/) {
                    print $i
                    exit 0
                }
            }
        }
    ' <<<"${output}"
}

if [[ -z "${DEVICE_ARG}" ]]; then
    DEVICE_ARG="$(resolve_device)"
    if [[ -z "${DEVICE_ARG}" ]]; then
        echo "エラー: 接続中の実機が見つかりません" >&2
        echo "  USB 接続して xcrun devicectl list devices で connected と表示されることを確認してください" >&2
        echo "  もしくは --device <UDID|名前> で明示指定してください" >&2
        exit 1
    fi
    echo "自動検出した device: ${DEVICE_ARG}"
else
    echo "指定された device: ${DEVICE_ARG}"
fi

echo ""
echo "===== 1/3: xcodebuild test (実機) ====="
xcodebuild test \
    -project "${PROJECT}" \
    -scheme "${SCHEME}" \
    -destination "platform=iOS,id=${DEVICE_ARG}" \
    -only-testing:"${TEST_TARGET}"

echo ""
echo "===== 2/3: Documents/Result を吸い出し ====="
"${EXTRACT_SCRIPT}" --device "${DEVICE_ARG}"

echo ""
echo "===== 3/3: メトリクス集計 ====="
if [[ ! -x "${PYTHON}" ]]; then
    echo "警告: ${PYTHON} が見つかりません。集計はスキップします。" >&2
    echo "      手動で aggregate_metrics.py を回してください。" >&2
    exit 0
fi
"${PYTHON}" "${AGGREGATE_SCRIPT}"
