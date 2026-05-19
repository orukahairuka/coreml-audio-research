import Testing
@testable import CoreMLAudioApp
import CoreML
import Foundation

/// Phase 4 — MLComputePlan で Decoder（および参考に Encoder / HiFi-GAN）の per-op
/// device assignment を取得して JSON に書き出すテスト。
///
/// 計画書: [`docs/2026-05-19/all-engine-precision-stability-plan.md`](../../../../docs/2026-05-19/all-engine-precision-stability-plan.md) §4
///
/// 各テストは 1 つの (precision, computeUnits) で Decoder の dispatch map を取得し、
/// `Documents/Result/debug/compute_plan/` に書き出す。
///
/// iOS 17.0+ 限定。シミュレータでも実機でも走るが、device assignment は実機の方が
/// 意味のある結果になるので Phase 1 と同じ実機で走らせるのが原則。
/// `@Test` / `@Suite` マクロは `@available` を許容しないので、struct 自体は availability
/// 注釈なしで、各テスト関数内で `#available` ガードする。
///
/// 集計は `scripts/aggregate_compute_plan.py` で。
@Suite(.serialized)
struct Phase4ComputePlanTests {

    // MARK: - F32

    @Test
    func decoder_f32_cpuOnly() async throws {
        try await inspectDecoder(precision: .float32, computeUnits: .cpuOnly)
    }

    @Test
    func decoder_f32_cpuAndGPU() async throws {
        try await inspectDecoder(precision: .float32, computeUnits: .cpuAndGPU)
    }

    @Test
    func decoder_f32_cpuAndNE() async throws {
        try await inspectDecoder(precision: .float32, computeUnits: .cpuAndNE)
    }

    @Test
    func decoder_f32_all() async throws {
        try await inspectDecoder(precision: .float32, computeUnits: .all)
    }

    // MARK: - F16

    @Test
    func decoder_f16_cpuAndGPU() async throws {
        try await inspectDecoder(precision: .float16, computeUnits: .cpuAndGPU)
    }

    @Test
    func decoder_f16_cpuAndNE() async throws {
        try await inspectDecoder(precision: .float16, computeUnits: .cpuAndNE)
    }

    @Test
    func decoder_f16_all() async throws {
        try await inspectDecoder(precision: .float16, computeUnits: .all)
    }

    // MARK: - Int8

    @Test
    func decoder_int8_cpuAndGPU() async throws {
        try await inspectDecoder(precision: .int8, computeUnits: .cpuAndGPU)
    }

    // MARK: - 参考: Encoder / HiFi-GAN

    @Test
    func encoder_f32_cpuAndGPU() async throws {
        try await inspectEncoder(precision: .float32, computeUnits: .cpuAndGPU)
    }

    @Test
    func encoder_f16_cpuAndNE() async throws {
        try await inspectEncoder(precision: .float16, computeUnits: .cpuAndNE)
    }

    @Test
    func hifigan_f32_cpuAndGPU() async throws {
        try await inspectHifigan(precision: .float32, computeUnits: .cpuAndGPU)
    }

    // ANE クリッピングが Decoder 由来か HiFi-GAN 由来かを切り分けるため、F16/Int8 × cpuAndNE/all の HiFi-GAN dispatch を取る
    @Test
    func hifigan_f16_cpuAndGPU() async throws {
        try await inspectHifigan(precision: .float16, computeUnits: .cpuAndGPU)
    }

    @Test
    func hifigan_f16_cpuAndNE() async throws {
        try await inspectHifigan(precision: .float16, computeUnits: .cpuAndNE)
    }

    @Test
    func hifigan_f16_all() async throws {
        try await inspectHifigan(precision: .float16, computeUnits: .all)
    }

    @Test
    func hifigan_int8_cpuAndNE() async throws {
        try await inspectHifigan(precision: .int8, computeUnits: .cpuAndNE)
    }

    @Test
    func hifigan_int8_all() async throws {
        try await inspectHifigan(precision: .int8, computeUnits: .all)
    }

    @Test
    func decoder_int8_cpuAndNE() async throws {
        try await inspectDecoder(precision: .int8, computeUnits: .cpuAndNE)
    }

    @Test
    func decoder_int8_all() async throws {
        try await inspectDecoder(precision: .int8, computeUnits: .all)
    }

    // MARK: - helper

    private func inspectDecoder(precision: ModelPrecision, computeUnits: ComputeUnitOption) async throws {
        let shapeMode = pickShapeMode(for: precision)
        let resourceName = shapeMode.transformerDecoderResourceName(for: precision)
        try await inspect(resourceName: resourceName, precision: precision, computeUnits: computeUnits)
    }

    private func inspectEncoder(precision: ModelPrecision, computeUnits: ComputeUnitOption) async throws {
        let shapeMode = pickShapeMode(for: precision)
        let resourceName = shapeMode.transformerEncoderResourceName(for: precision)
        try await inspect(resourceName: resourceName, precision: precision, computeUnits: computeUnits)
    }

    private func inspectHifigan(precision: ModelPrecision, computeUnits: ComputeUnitOption) async throws {
        let shapeMode = pickShapeMode(for: precision)
        guard let resourceName = shapeMode.hifiganResourceName(for: precision) else {
            #expect(Bool(false), "HiFi-GAN resource が \(precision.rawValue) で見つからない")
            return
        }
        try await inspect(resourceName: resourceName, precision: precision, computeUnits: computeUnits)
    }

    private func inspect(resourceName: String, precision: ModelPrecision, computeUnits: ComputeUnitOption) async throws {
        guard #available(iOS 17.0, *) else {
            print("[Phase4ComputePlanTests] skip: iOS 17+ が必要 (\(resourceName))")
            return
        }
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") else {
            #expect(Bool(false), "\(resourceName).mlmodelc が Bundle に見つからない")
            return
        }
        let inspector = ComputePlanInspector(
            modelURL: url,
            modelName: resourceName,
            precision: precision,
            computeUnits: computeUnits
        )
        let summary = try await inspector.inspectAndWrite()
        print("[Phase4ComputePlanTests] \(resourceName) precision=\(precision.rawValue) computeUnits=\(computeUnits.rawValue) -> \(summary)")
    }

    private func pickShapeMode(for precision: ModelPrecision) -> ShapeModeOption {
        if ShapeModeOption.fixed262.isAvailable(for: precision) { return .fixed262 }
        return .range1
    }
}
