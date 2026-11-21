import CoreML
import Foundation

/// 計測モードで保存した配列（npy・wav）の説明を、実行ごとに `<prefix>_meta.json` として書く内容
///
/// 他の環境（NVIDIA GPU の PyTorch など）の出力と Python で比べるときに、
/// 実験条件と各配列の形状・データ型・サンプリング周波数をこのファイルだけで確かめられるようにする。
/// 書き出しは合成の計測が終わった後に行うので、計測時間には含まれない。
enum ThesisRunMetadata {

    /// meta.json の書式の版（キーを変えたら上げる）
    static let formatVersion = 1

    /// 保存した配列 1 つの説明
    struct ArrayEntry {
        let key: String
        let fileName: String
        /// npy の形状（行優先）。wav はサンプル数だけ
        let shape: [Int]
        /// npy は `float32`、wav は `int16`
        let dtype: String
        /// 各次元の意味（例: `["frame", "mel"]`）
        let axes: [String]
        let description: String
        var sha256: String?

        var dictionary: [String: Any] {
            var dict: [String: Any] = [
                "file": fileName,
                "format": fileName.hasSuffix(".wav") ? "wav" : "npy",
                "shape": shape,
                "dtype": dtype,
                "byte_order": "little",
                "axes": axes,
                "description": description,
            ]
            if let sha256 { dict["sha256"] = sha256 }
            return dict
        }
    }

    /// 合成に使った入力
    struct InputEntry {
        let name: String
        let file: String
        /// 前処理で得たフレーム数 T（vocoder モードは thesis_inputs.json の値）
        let totalFrames: Int
        /// 合成に使ったフレーム数 T′ = min(T, 262)
        let effectiveFrames: Int
    }

    // MARK: - モデルの入出力

    /// モデルの説明から、入出力の名前・形状・データ型を取り出す（推論はしない）
    static func modelEntry(resourceName: String, model: MLModel, computeUnit: ComputeUnitOption) -> [String: Any] {
        func features(_ descriptions: [String: MLFeatureDescription]) -> [String: Any] {
            var out = [String: Any]()
            for (name, desc) in descriptions {
                if let c = desc.multiArrayConstraint {
                    out[name] = ["shape": c.shape.map({ $0.intValue }), "dtype": dataTypeName(c.dataType)]
                } else {
                    out[name] = ["type": String(describing: desc.type)]
                }
            }
            return out
        }
        return [
            "resource": resourceName,
            "compute_units": computeUnit.rawValue,
            "inputs": features(model.modelDescription.inputDescriptionsByName),
            "outputs": features(model.modelDescription.outputDescriptionsByName),
        ]
    }

    static func dataTypeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "float16"
        case .float32: return "float32"
        case .double: return "float64"
        case .int32: return "int32"
        default: return "raw_\(type.rawValue)"
        }
    }

    // MARK: - 全体

    /// - Parameters:
    ///   - values: infer_runs.csv / vocoder_runs.csv の行（時間・状態の値をここから取る）
    ///   - models: `modelEntry` の結果（キーは encoder / decoder / hifigan）
    static func build(
        config: ThesisBenchConfig,
        values: [String: String],
        input: InputEntry,
        models: [String: Any],
        arrays: [ArrayEntry]
    ) -> [String: Any] {
        let fixedFrames: Int? = {
            if case .fixed(let t) = ThesisBenchConfig.shapeMode.inputPolicy { return t }
            return nil
        }()
        var inputDict: [String: Any] = [
            "name": input.name,
            "file": input.file,
            "T": input.totalFrames,
            "T_prime": input.effectiveFrames,
            "truncated_frames": max(0, input.totalFrames - input.effectiveFrames),
        ]
        if let fixedFrames {
            inputDict["zero_padded_frames"] = max(0, fixedFrames - input.effectiveFrames)
        }

        let timingKeys = [
            "preprocess_ms", "encoder_ms", "decoder_total_ms", "hifigan_ms", "synth_ms", "deemph_ms", "e2e_ms", "rtf",
        ]
        var timing = [String: Any]()
        for key in timingKeys {
            if let raw = values[key], let v = Double(raw) { timing[key] = v }
        }

        var arrayDict = [String: Any]()
        for entry in arrays { arrayDict[entry.key] = entry.dictionary }

        // 型の推論を軽くするため、入れ子の辞書は型を付けて別に作る
        let device: [String: Any] = [
            "model": ThesisRecorder.deviceModel,
            "os_version": ThesisRecorder.osVersion,
            "os_version_string": ThesisRecorder.osVersionString,
        ]
        // vocoder モードは Encoder・Decoder を使わない
        let encdecUnit: String = config.mode == .vocoder ? "" : (config.computeUnit?.rawValue ?? "")
        let computeUnits: [String: Any] = [
            "encoder": encdecUnit,
            "decoder": encdecUnit,
            "hifigan": config.hifiganComputeUnit?.rawValue ?? "",
        ]
        let audio: [String: Any] = [
            "sample_rate": Int(AudioFeatureExtractor.sampleRate),
            "hop_size": 256,
            "n_mels": AudioFeatureExtractor.nMels,
            "preemphasis": Double(AudioFeatureExtractor.preemphasisCoeff),
            "deemphasis": "y[n] = x[n] + 0.97 * y[n-1]（Float32 で計算）",
            "wav_conversion": "Float32 を 16 ビット PCM に変換（振幅の正規化なし。[-1, 1] を超えた値は上限・下限に置き換わる）",
        ]
        let run: [String: Any] = [
            "run_index": Int(values["run_index"] ?? "") ?? -1,
            "phase": values["phase"] ?? "",
            "status": values["status"] ?? "",
            "started_at": values["started_at"] ?? "",
            "thermal_before": values["thermal_before"] ?? "",
            "thermal_after": values["thermal_after"] ?? "",
        ]

        var meta = [String: Any]()
        meta["format_version"] = formatVersion
        meta["session_id"] = config.sessionID
        meta["launch_id"] = config.launchID
        meta["mode"] = config.mode.rawValue
        meta["bench_version"] = ThesisBenchConfig.benchVersion
        meta["condition_label"] = config.conditionLabel
        meta["device"] = device
        meta["precision"] = config.precision?.rawValue ?? ""
        meta["compute_units"] = computeUnits
        meta["shape_mode"] = ThesisBenchConfig.shapeMode.rawValue
        meta["model_frames"] = fixedFrames.map({ $0 as Any }) ?? NSNull()
        meta["input"] = inputDict
        meta["audio"] = audio
        meta["run"] = run
        meta["timing_ms"] = timing
        meta["timing_note"] = "配列・meta.json の保存は計測時間に含まない。rtf は合成処理時間 / 合成音の長さ"
        meta["models"] = models
        meta["arrays"] = arrayDict
        return meta
    }
}
