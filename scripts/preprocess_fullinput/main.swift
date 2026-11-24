// iOS の前処理（AudioFeatureExtractor）を Mac で動かし、読み込み後の波形とメルを npy に保存する。
// scripts/verify_fullinput_preprocess.py がアプリのソースと一緒にビルドして実行する。
//
// 使い方: preprocess_fullinput <出力先> <名前>=<wav>...
// 出力: <名前>_load.npy（22050 Hz・モノラルにした波形）、<名前>_<trimAndFade|fullInput>_mel.npy [T × 256]
import Foundation

let arguments = CommandLine.arguments
let outputDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

for spec in arguments[2...] {
    let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
    let name = parts[0]
    let url = URL(fileURLWithPath: parts[1])
    let samples = try AudioFeatureExtractor.loadAudio(from: url)
    try NpyWriter.writeFloat32(samples, rows: 1, cols: samples.count,
                               to: outputDirectory.appendingPathComponent("\(name)_load.npy"))
    for mode in [AudioFeatureExtractor.WaveformPreprocess.trimAndFade, .fullInput] {
        let result = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples, waveformPreprocess: mode)
        try NpyWriter.writeFloat32(result.mel, rows: result.frameCount, cols: AudioFeatureExtractor.nMels,
                                   to: outputDirectory.appendingPathComponent("\(name)_\(mode.rawValue)_mel.npy"))
    }
}
