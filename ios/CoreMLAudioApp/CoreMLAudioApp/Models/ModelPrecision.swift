/// CoreML モデルの精度 (変換時に決定される)
enum ModelPrecision: String, CaseIterable, Identifiable {
    case float32 = "Float32"
    case float16 = "Float16"
    case int8 = "Int8"

    var id: String { rawValue }

    /// Bundle 内のモデル名サフィックス
    var suffix: String { rawValue.lowercased() }
}
