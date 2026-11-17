import Foundation

/// End-to-End 時間を処理ごとに切り分けて積算する（卒論第4章の追加計測用）
///
/// `start()` から最後の `lap(_:)` までを、`lap` を呼んだ位置で区切って隣り合う区間に分け、
/// 各区間の時間を `lap` に渡した名前ごとに足し込む。区間は隙間なく並ぶので、
/// 全区間の合計は `start()` から最後の `lap` までの経過時間と一致する。
///
/// Runner は `profiler: StageProfiler? = nil` を受け取り、nil のときは `profiler?.lap(...)` が
/// 何もしないので、従来の経路の動作は変わらない。
final class StageProfiler {

    /// 区間名 → 積算時間 (ms)
    private(set) var totals: [String: Double] = [:]
    /// 区間名 → 区間を通った回数（Decoder のステップごとの区間なら 262 など）
    private(set) var counts: [String: Int] = [:]
    private var last: CFAbsoluteTime = 0
    private var started: CFAbsoluteTime = 0

    /// 計測の起点を置く
    func start(at time: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()) {
        totals = [:]
        counts = [:]
        started = time
        last = time
    }

    /// 直前の区切りから今までを `stage` の時間として足し込み、ここを次の区間の起点にする。
    /// 区切った時刻を返す（呼び出し側が End-to-End の終点にそろえるため）
    @discardableResult
    func lap(_ stage: String) -> CFAbsoluteTime {
        let now = CFAbsoluteTimeGetCurrent()
        totals[stage, default: 0] += (now - last) * 1000
        counts[stage, default: 0] += 1
        last = now
        return now
    }

    /// 全区間の合計 (ms)
    var sumMs: Double { totals.values.reduce(0, +) }

    /// `start()` から最後の `lap` までの経過時間 (ms)
    var spanMs: Double { (last - started) * 1000 }

    func ms(_ stage: String) -> Double { totals[stage] ?? 0 }
}

/// 処理別計測の区間名。CSV の列名（`<名前>_ms`）にもそのまま使う
enum ProfileStage {
    // 前処理
    static let audioLoad = "pre_audio_load"
    static let melExtract = "pre_mel_extract"
    // Encoder
    static let encInputFill = "enc_input_fill"
    static let encInputValidate = "enc_input_validate"
    static let encPosFill = "enc_pos_fill"
    static let encFeatureProvider = "enc_feature_provider"
    static let encPredict = "enc_predict"
    static let encOutputExtract = "enc_output_extract"
    // Decoder（ステップごとに積算）
    /// 入力配列の確保と 0 埋め（ch4-v4 で dec_input_fill から分けた）
    static let decInputAlloc = "dec_input_alloc"
    /// 生成済みフレームの書き込み（ch4-v3 までは確保と 0 埋めも含んでいた）
    static let decInputFill = "dec_input_fill"
    static let decInputValidate = "dec_input_validate"
    static let decPosFill = "dec_pos_fill"
    static let decFeatureProvider = "dec_feature_provider"
    static let decPredict = "dec_predict"
    static let decOutputExtract = "dec_output_extract"
    static let decStepStats = "dec_step_stats"
    static let decAutoregressRead = "dec_autoregress_read"
    static let decProgressYield = "dec_progress_yield"
    // HiFi-GAN
    static let vocInputTranspose = "voc_input_transpose"
    static let vocInputValidate = "voc_input_validate"
    static let vocFeatureProvider = "voc_feature_provider"
    static let vocPredict = "voc_predict"
    static let vocOutputRead = "voc_output_read"
    // 後処理
    static let deemph = "post_deemph"

    /// CSV に出す順
    static let all: [String] = [
        audioLoad, melExtract,
        encInputFill, encInputValidate, encPosFill, encFeatureProvider, encPredict, encOutputExtract,
        decInputAlloc, decInputFill, decInputValidate, decPosFill, decFeatureProvider, decPredict, decOutputExtract,
        decStepStats, decAutoregressRead, decProgressYield,
        vocInputTranspose, vocInputValidate, vocFeatureProvider, vocPredict, vocOutputRead,
        deemph,
    ]

    /// モデルの呼び出し（prediction）の区間
    static let predictStages: Set<String> = [encPredict, decPredict, vocPredict]
}
