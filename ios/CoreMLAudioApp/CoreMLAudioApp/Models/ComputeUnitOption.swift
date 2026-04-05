import CoreML

enum ComputeUnitOption: String, CaseIterable {
    case all       = "all"
    case cpuAndGPU = "cpuAndGPU"
    case cpuAndNE  = "cpuAndNE"
    case cpuOnly   = "cpuOnly"

    var displayName: String {
        switch self {
        case .all:       return "すべて (CPU+GPU+ANE)"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNE:  return "CPU + ANE"
        case .cpuOnly:   return "CPU のみ"
        }
    }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all:       return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNE:  return .cpuAndNeuralEngine
        case .cpuOnly:   return .cpuOnly
        }
    }
}
