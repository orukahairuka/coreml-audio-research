import CoreML
import Foundation

/// Phase 4 — MLComputePlan で Decoder（および他段）の per-op device assignment を取得して
/// JSON に書き出すユーティリティ。
///
/// 計画書: [`docs/2026-05-19/all-engine-precision-stability-plan.md`](../../../../../docs/2026-05-19/all-engine-precision-stability-plan.md) §4
///
/// 出力先:
///   `Documents/Result/debug/compute_plan/<precision>_<computeUnits>_<modelName>.json`
///
/// 含まれる項目:
///   - precision / computeUnits / modelName
///   - 各 op の `operatorName`, `outputNames`, `preferredDevice`, `supportedDevices`,
///     `estimatedCostWeight`
///   - dispatch サマリー（CPU / GPU / NE / 不明 の op 数）
///
/// 呼び出しは Phase 2/3 のテストから debug-only で行う想定。本番ビルドには影響しない
/// （build しても呼び出されなければ何もしない）。
///
/// iOS 17.0+ 限定。NeuralNetwork 形式の旧 .mlmodel は MLProgram でないので op 列挙不可。
/// 本プロジェクトの .mlpackage は MLProgram なので問題ない。
@available(iOS 17.0, *)
struct ComputePlanInspector {

    let modelURL: URL
    let modelName: String
    let precision: ModelPrecision
    let computeUnits: ComputeUnitOption

    /// Dispatch を JSON ファイルに書き出し、結果サマリーを返す。
    @discardableResult
    func inspectAndWrite() async throws -> String {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.mlComputeUnits

        let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: config)
        let entries = collectOperations(from: plan.modelStructure, plan: plan)
        let summary = makeSummary(entries: entries)

        let outURL = try outputURL()
        let payload: [String: Any] = [
            "precision": precision.rawValue,
            "computeUnits": computeUnits.rawValue,
            "modelName": modelName,
            "operationCount": entries.count,
            "dispatchSummary": summary.toDictionary(),
            "operations": entries.map { $0.toDictionary() },
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outURL, options: .atomic)
        print("[ComputePlanInspector] wrote \(outURL.lastPathComponent) ops=\(entries.count) " +
              "cpu=\(summary.cpu) gpu=\(summary.gpu) ne=\(summary.ne) unknown=\(summary.unknown)")
        return "ops=\(entries.count) cpu=\(summary.cpu) gpu=\(summary.gpu) ne=\(summary.ne)"
    }

    // MARK: - 内部実装

    private func collectOperations(from structure: MLModelStructure, plan: MLComputePlan) -> [OperationEntry] {
        // MLModelStructure は iOS 17 で enum 形式。switch で program ケースを取り出す。
        // pipeline / neuralNetwork は本プロジェクトでは使わないので空を返す。
        switch structure {
        case .program(let program):
            return collectFromProgram(program, plan: plan)
        case .neuralNetwork, .pipeline, .unsupported:
            return []
        @unknown default:
            return []
        }
    }

    private func collectFromProgram(_ program: MLModelStructure.Program, plan: MLComputePlan) -> [OperationEntry] {
        var entries: [OperationEntry] = []
        for (functionName, function) in program.functions {
            collectFromBlock(function.block, functionName: functionName, plan: plan, into: &entries)
        }
        return entries
    }

    private func collectFromBlock(
        _ block: MLModelStructure.Program.Block,
        functionName: String,
        plan: MLComputePlan,
        into entries: inout [OperationEntry]
    ) {
        for op in block.operations {
            let usage = plan.deviceUsage(for: op)
            let cost = plan.estimatedCost(of: op)
            entries.append(OperationEntry(
                functionName: functionName,
                operatorName: op.operatorName,
                outputNames: op.outputs.map { $0.name },
                preferredDevice: deviceLabel(usage?.preferred),
                supportedDevices: (usage?.supported ?? []).map { deviceLabel($0) },
                weight: cost?.weight ?? 0
            ))
            for nested in op.blocks {
                collectFromBlock(nested, functionName: functionName, plan: plan, into: &entries)
            }
        }
    }

    private func deviceLabel(_ device: MLComputeDevice?) -> String {
        guard let device else { return "unknown" }
        switch device {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neuralEngine"
        @unknown default: return "unknown"
        }
    }

    private func makeSummary(entries: [OperationEntry]) -> DispatchSummary {
        var s = DispatchSummary()
        for e in entries {
            switch e.preferredDevice {
            case "cpu": s.cpu += 1
            case "gpu": s.gpu += 1
            case "neuralEngine": s.ne += 1
            default: s.unknown += 1
            }
        }
        return s
    }

    private func outputURL() throws -> URL {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs
            .appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("debug", isDirectory: true)
            .appendingPathComponent("compute_plan", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let fileName = "\(precision.rawValue)_\(computeUnits.rawValue)_\(modelName).json"
        return dir.appendingPathComponent(fileName)
    }

    // MARK: - DTO

    struct OperationEntry {
        let functionName: String
        let operatorName: String
        let outputNames: [String]
        let preferredDevice: String
        let supportedDevices: [String]
        let weight: Double

        func toDictionary() -> [String: Any] {
            [
                "functionName": functionName,
                "operatorName": operatorName,
                "outputNames": outputNames,
                "preferredDevice": preferredDevice,
                "supportedDevices": supportedDevices,
                "estimatedCostWeight": weight,
            ]
        }
    }

    struct DispatchSummary {
        var cpu: Int = 0
        var gpu: Int = 0
        var ne: Int = 0
        var unknown: Int = 0

        func toDictionary() -> [String: Any] {
            ["cpu": cpu, "gpu": gpu, "neuralEngine": ne, "unknown": unknown]
        }
    }
}
