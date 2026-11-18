/// 合成処理の高速化の切り替え
///
/// どの組み合わせでも、モデルの入出力と合成した波形は変わらない（計測モードと単体テストで確認する）。
/// `adopted` が通常の合成で使う組み合わせで、計測モードは変更の前後を比べるために個別に指定する。
nonisolated struct SynthesisOptimizations: OptionSet, Sendable {
    let rawValue: Int

    /// HiFi-GAN の入力の転置をメモリへの直接のアクセスで行う
    static let vocoderTranspose = SynthesisOptimizations(rawValue: 1 << 0)
    /// Decoder の出力から次の入力のフレームをメモリから直接読む
    static let decoderFrameRead = SynthesisOptimizations(rawValue: 1 << 1)
    /// メルフィルタバンクの 0 でない範囲だけを計算し、行列を再利用する
    static let sparseMelFilterbank = SynthesisOptimizations(rawValue: 1 << 2)
    /// HiFi-GAN の出力波形をメモリから直接読む
    static let vocoderOutputRead = SynthesisOptimizations(rawValue: 1 << 3)
    /// Encoder の入力をまとめてコピーする
    static let encoderInputCopy = SynthesisOptimizations(rawValue: 1 << 4)
    /// 固定長のとき Decoder の pos の配列を再利用する
    static let reuseDecoderPos = SynthesisOptimizations(rawValue: 1 << 5)
    /// 固定長のとき Decoder の入力の配列を再利用する
    static let reuseDecoderInput = SynthesisOptimizations(rawValue: 1 << 6)
    /// モデルの呼び出しに同期版の `prediction(from:)` を使う。
    /// メインアクターの上で使うと呼び出しの間は画面の更新が止まるので、`offMainInference` と組み合わせる
    static let syncPrediction = SynthesisOptimizations(rawValue: 1 << 7)

    /// Encoder・Decoder・HiFi-GAN の実行をメインアクターの外で行う
    static let offMainInference = SynthesisOptimizations(rawValue: 1 << 8)

    /// 計測モードの指定とファイルに書く名前（適用を試す順）
    static let named: [(name: String, option: SynthesisOptimizations)] = [
        ("vocoderTranspose", .vocoderTranspose),
        ("decoderFrameRead", .decoderFrameRead),
        ("sparseMelFilterbank", .sparseMelFilterbank),
        ("vocoderOutputRead", .vocoderOutputRead),
        ("encoderInputCopy", .encoderInputCopy),
        ("reuseDecoderPos", .reuseDecoderPos),
        ("reuseDecoderInput", .reuseDecoderInput),
        ("syncPrediction", .syncPrediction),
        ("offMainInference", .offMainInference),
    ]

    /// 通常の合成で使う組み合わせ（実機の前後比較で採用したもの）
    static let adopted: SynthesisOptimizations = [
        .vocoderTranspose, .decoderFrameRead, .sparseMelFilterbank, .vocoderOutputRead,
        .encoderInputCopy, .reuseDecoderPos, .reuseDecoderInput,
        .syncPrediction, .offMainInference,
    ]

    /// `none` またはカンマ区切りの名前を読む。知らない名前があれば nil
    static func parse(_ text: String) -> SynthesisOptimizations? {
        var result: SynthesisOptimizations = []
        for name in text.split(separator: ",").map({ String($0) }) where name != "none" {
            guard let entry = named.first(where: { $0.name == name }) else { return nil }
            result.insert(entry.option)
        }
        return result
    }

    /// CSV に書く表記
    var label: String {
        let names = Self.named.filter({ contains($0.option) }).map({ $0.name })
        return names.isEmpty ? "none" : names.joined(separator: "+")
    }
}
