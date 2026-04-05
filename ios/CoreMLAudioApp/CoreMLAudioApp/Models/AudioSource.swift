import Foundation

/// 入力音声のソース
enum AudioSource: Equatable {
    /// バンドルされたサンプル音声
    case bundledSample
    /// 録音した音声ファイル
    case recording(URL)

    var displayName: String {
        switch self {
        case .bundledSample:
            return "サンプル音声"
        case .recording(let url):
            return url.deletingPathExtension().lastPathComponent
        }
    }
}
