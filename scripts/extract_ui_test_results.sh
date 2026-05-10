#!/usr/bin/env bash
#
# XCUITest (testCaptureAllCombinations) が書き出した
# Documents/Result/ 配下 (mel/ と timing/) をホストの result/ にコピーする。
#
# 使い方:
#   ./scripts/extract_ui_test_results.sh                    # シミュレータから取得
#   ./scripts/extract_ui_test_results.sh --device <名前|UDID>  # 実機から取得
#
# 前提:
#   - シミュレータ: アプリがインストール済みで booted、XCUITest を1回以上実行済み
#   - 実機: iPhone を USB 接続、XCUITest を1回以上実行済み、Xcode 15+ (xcrun devicectl 利用)
#
# --device の値は xcrun devicectl list devices で表示される Name か Identifier。

set -euo pipefail

BUNDLE_ID="erika.com.CoreMLAudioApp"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/result"

DEVICE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#--device=}"
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

extract_from_device() {
    # 実機 (xcrun devicectl) から /Documents/Result を一時ディレクトリに落とし、
    # その中身を DEST_DIR に rsync で複製する。
    local tmp_root
    tmp_root="$(mktemp -d)"
    # set -e や exit による早期終了でも tmp を残さないため EXIT トラップで掃除する
    # (RETURN だと function 内の exit / set -e で発火しない)
    trap "rm -rf '${tmp_root}'" EXIT

    echo "device: ${DEVICE}"
    echo "bundle: ${BUNDLE_ID}"
    echo "fetching /Documents/Result -> ${tmp_root}"

    # devicectl は --source ディレクトリの中身を --destination 直下に展開する
    # (Result/ ラッパーは作らない) ので tmp_root をそのまま src として扱う。
    xcrun devicectl device copy from \
        --device "${DEVICE}" \
        --domain-type appDataContainer \
        --domain-identifier "${BUNDLE_ID}" \
        --source /Documents/Result \
        --destination "${tmp_root}"

    local src_dir="${tmp_root}"
    if [[ ! -d "${src_dir}/mel" || ! -d "${src_dir}/timing" ]]; then
        echo "エラー: 取得したコンテナに mel/ と timing/ が揃っていません" >&2
        echo "  実機で XCUITest (testCaptureAllCombinations) を完走させてから再実行してください" >&2
        exit 1
    fi

    mkdir -p "${DEST_DIR}"
    rsync -av --delete --exclude='.gitkeep' "${src_dir}/" "${DEST_DIR}/"
}

extract_from_simulator() {
    # 標準シミュレータと XCTestDevices (xcodebuild が UI テスト用に作るクローン)
    # の両方を対象にアプリコンテナを探す
    local simctl_sets=(
        ""  # デフォルト (~/Library/Developer/CoreSimulator/Devices)
        "--set ${HOME}/Library/Developer/XCTestDevices"
    )

    local container=""
    local found_set=""
    local found_device=""
    # XCUITest が完走したコンテナだけを拾うため mel/ と timing/ の両方を要求する。
    # ここを緩めると AudioPlayer の副産物 (output_*.wav) しか無いコンテナを掴んで
    # rsync --delete でローカルの mel/ timing/ が消える。
    for set_arg in "${simctl_sets[@]}"; do
        # 該当 set 配下のデバイスをリスト
        while IFS= read -r device_id; do
            [[ -z "${device_id}" ]] && continue
            # shellcheck disable=SC2086
            local c
            c=$(xcrun simctl ${set_arg} get_app_container "${device_id}" "${BUNDLE_ID}" data 2>/dev/null || true)
            if [[ -n "${c}" && -d "${c}/Documents/Result/mel" && -d "${c}/Documents/Result/timing" ]]; then
                container="${c}"
                found_set="${set_arg:-default}"
                found_device="${device_id}"
                break 2
            fi
        # shellcheck disable=SC2086
        done < <(xcrun simctl ${set_arg} list devices 2>/dev/null | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}')
    done

    if [[ -z "${container}" ]]; then
        echo "エラー: ${BUNDLE_ID} のデータコンテナで mel/ と timing/ が両方揃ったものが見つかりません" >&2
        echo "先に XCUITest (testCaptureAllCombinations) を完走させてから再実行してください" >&2
        exit 1
    fi
    echo "set: ${found_set}"
    echo "device: ${found_device}"
    local src_dir="${container}/Documents/Result"

    # コピー実行 (既存ファイルは上書き、Documents/Result 配下のサブディレクトリを丸ごと複製)
    # 上書き時に古い成果物が混ざらないよう --delete を付ける
    mkdir -p "${DEST_DIR}"
    rsync -av --delete --exclude='.gitkeep' "${src_dir}/" "${DEST_DIR}/"
}

if [[ -n "${DEVICE}" ]]; then
    extract_from_device
else
    extract_from_simulator
fi

echo ""
echo "書き出し先: ${DEST_DIR}"
find "${DEST_DIR}" -mindepth 1 -maxdepth 2 -type f -o -type d | sed "s|${DEST_DIR}/|  |"
