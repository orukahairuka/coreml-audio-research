#!/usr/bin/env bash
#
# XCUITest (testCaptureAllCombinations) が書き出した
# Documents/Result/mel/ を booted シミュレータから
# リポジトリの result/mel/ にコピーする。
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
DEST_DIR="${REPO_ROOT}/result/mel"

# booted シミュレータの有無を確認
BOOTED=$(xcrun simctl list devices booted | awk '/\(Booted\)/ {print $NF}' | tr -d '()' | head -n 1)
if [[ -z "${BOOTED}" ]]; then
    echo "エラー: booted 状態のシミュレータが見つかりません" >&2
    echo "先に Xcode で XCUITest を走らせるか、シミュレータを起動してください" >&2
    exit 1
fi
echo "booted simulator: ${BOOTED}"

# アプリのデータコンテナ (Documents/ の親) を取得
CONTAINER=$(xcrun simctl get_app_container "${BOOTED}" "${BUNDLE_ID}" data 2>/dev/null || true)
if [[ -z "${CONTAINER}" || ! -d "${CONTAINER}" ]]; then
    echo "エラー: ${BUNDLE_ID} のデータコンテナが取得できません" >&2
    echo "シミュレータにアプリがインストールされているか確認してください" >&2
    exit 1
fi
SRC_DIR="${CONTAINER}/Documents/Result/mel"
if [[ ! -d "${SRC_DIR}" ]]; then
    echo "エラー: ${SRC_DIR} が存在しません" >&2
    echo "XCUITest を実行してメルファイルを生成してから再実行してください" >&2
    exit 1
fi

# コピー実行 (既存ファイルは上書き)
mkdir -p "${DEST_DIR}"
rsync -av --delete "${SRC_DIR}/" "${DEST_DIR}/"

echo ""
echo "書き出し先: ${DEST_DIR}"
ls -1 "${DEST_DIR}" | sed 's/^/  /'
