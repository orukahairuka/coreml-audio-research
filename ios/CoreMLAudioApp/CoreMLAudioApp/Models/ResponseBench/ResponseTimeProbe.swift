import Foundation

/// 通常の合成操作（`SynthesisViewModel.runSynthesis()`）の応答時間を測るための計測点
///
/// 時刻は単調増加時計（`DispatchTime.now().uptimeNanoseconds`、mach_absolute_time 由来）で取る。
/// 計測点を打つのは時刻を 1 つ配列に書くだけで、合成・保存・再生の処理には手を加えない。
/// `SynthesisViewModel.responseProbe` が nil（通常の起動）のときは何もしない。
///
/// 計測点の並びは実際の処理の順（`runSynthesis()` → `playOutputAndAwaitCompletion()` → `AudioPlayer.play()`）
final class ResponseTimeProbe {

    enum Point: Int, CaseIterable {
        /// `runSynthesis()` の先頭
        case entry
        /// A: `loadModels` から戻った直後（読み込み済みなら確認だけで戻る）。合成処理の開始とみなす
        case a
        /// B: `synthesizer.synthesize()` から戻った直後
        case b
        /// C: メルスペクトログラムと時間記録のファイル（`saveMelArtifacts` / `saveTimingArtifact`）の保存の完了
        case c
        /// D: `audioPlayer.play()` を呼ぶ直前
        case d
        /// E: `AudioPlayer.play()` の中で、合成音の WAV の保存が終わった直後
        case e
        /// F: `AudioPlayer.play()` の中で、`AVAudioPlayer.play()`（再生開始の API）を呼ぶ直前
        case f
        /// G: `AVAudioPlayer.play()` から戻った直後
        case g
        /// 再生終了の通知（didFinishPlaying）を MainActor で受けた時刻
        case end
    }

    private var marks = [UInt64](repeating: 0, count: Point.allCases.count)

    func reset() {
        for i in marks.indices { marks[i] = 0 }
    }

    func mark(_ point: Point) {
        marks[point.rawValue] = DispatchTime.now().uptimeNanoseconds
    }

    func has(_ point: Point) -> Bool {
        marks[point.rawValue] != 0
    }

    /// `from` から `to` までのミリ秒。どちらかが打たれていなければ nil
    func ms(_ from: Point, _ to: Point) -> Double? {
        let t0 = marks[from.rawValue]
        let t1 = marks[to.rawValue]
        guard t0 != 0, t1 != 0 else { return nil }
        return Double(Int64(bitPattern: t1 &- t0)) / 1_000_000
    }
}
