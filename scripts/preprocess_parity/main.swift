// 修正前と修正後の前処理の各段階を Mac で npy に保存する。
// scripts/compare_preprocess_parity.py --build-mac がアプリのソースと一緒にビルドして実行する。
//
// 使い方: preprocess_parity <出力先> <wav>...（wav のファイル名の先頭の `_` までを入力の名前にする）
import Foundation

let arguments = CommandLine.arguments
let outputDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

for path in arguments[2...] {
    let url = URL(fileURLWithPath: path)
    let name = url.lastPathComponent.components(separatedBy: "_")[0]
    let results = [("legacy", try PreprocessStages.legacy(url: url)), ("current", try PreprocessStages.current(url: url))]
    for (implementation, stages) in results {
        for array in stages.arrays {
            let file = outputDirectory.appendingPathComponent("preprocess_\(implementation)_in\(name)_\(array.name).npy")
            try NpyWriter.writeFloat32(array.values, rows: array.rows, cols: array.cols, to: file)
        }
    }
}
