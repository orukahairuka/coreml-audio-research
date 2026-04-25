#!/usr/bin/env bash
#
# XCUITest (testCaptureAllCombinations) が書き出した
# Documents/Result/ 配下 (mel/ と timing/) を booted シミュレータから
# リポジトリの result/ にコピーする。
#
# 使い方:
#   ./scripts/extract_ui_test_results.sh
#
# 前提:
#   - シミュレータに CoreMLAudioApp がインストール済み
#   - そのシミュレータが booted 状態
#   - 事前に XCUITest を1回以上実行している

set -euo pipefail

BUNDLE_ID="erika.com.CoreMLAudioApp"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/result"

# 標準シミュレータと XCTestDevices (xcodebuild が UI テスト用に作るクローン)
# の両方を対象にアプリコンテナを探す
SIMCTL_SETS=(
    ""  # デフォルト (~/Library/Developer/CoreSimulator/Devices)
    "--set ${HOME}/Library/Developer/XCTestDevices"
)

CONTAINER=""
FOUND_SET=""
FOUND_DEVICE=""
# XCUITest が完走したコンテナだけを拾うため mel/ と timing/ の両方を要求する。
# ここを緩めると AudioPlayer の副産物 (output_*.wav) しか無いコンテナを掴んで
# rsync --delete でローカルの mel/ timing/ が消える。
for SET_ARG in "${SIMCTL_SETS[@]}"; do
    # 該当 set 配下のデバイスをリスト
    while IFS= read -r DEVICE_ID; do
        [[ -z "${DEVICE_ID}" ]] && continue
        # shellcheck disable=SC2086
        C=$(xcrun simctl ${SET_ARG} get_app_container "${DEVICE_ID}" "${BUNDLE_ID}" data 2>/dev/null || true)
        if [[ -n "${C}" && -d "${C}/Documents/Result/mel" && -d "${C}/Documents/Result/timing" ]]; then
            CONTAINER="${C}"
            FOUND_SET="${SET_ARG:-default}"
            FOUND_DEVICE="${DEVICE_ID}"
            break 2
        fi
    done < <(xcrun simctl ${SET_ARG} list devices 2>/dev/null | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}')
done

if [[ -z "${CONTAINER}" ]]; then
    echo "エラー: ${BUNDLE_ID} のデータコンテナで mel/ と timing/ が両方揃ったものが見つかりません" >&2
    echo "先に XCUITest (testCaptureAllCombinations) を完走させてから再実行してください" >&2
    exit 1
fi
echo "set: ${FOUND_SET}"
echo "device: ${FOUND_DEVICE}"
SRC_DIR="${CONTAINER}/Documents/Result"

# コピー実行 (既存ファイルは上書き、Documents/Result 配下のサブディレクトリを丸ごと複製)
# 上書き時に古い成果物が混ざらないよう --delete を付ける
mkdir -p "${DEST_DIR}"
rsync -av --delete --exclude='.gitkeep' "${SRC_DIR}/" "${DEST_DIR}/"

echo ""
echo "書き出し先: ${DEST_DIR}"
find "${DEST_DIR}" -mindepth 1 -maxdepth 2 -type f -o -type d | sed "s|${DEST_DIR}/|  |"
