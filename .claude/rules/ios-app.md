---
paths: ios/**/*.swift
---

## iOS アプリ (CoreMLAudioApp)

### アーキテクチャ
- MVVM パターンを採用
- View → ViewModel → Model の依存方向を守る

### ファイル構成
```
ios/CoreMLAudioApp/CoreMLAudioApp/
├── App/           — アプリエントリーポイント
├── Views/         — SwiftUI View
├── ViewModels/    — ViewModel（UI ステート管理）
├── Models/        — CoreML 合成・音声処理ロジック
└── Result/        — 生成された音声ファイルの保存先（gitignore 対象）
```

### コーディングスタイル
- トレーリングクロージャは使わない。引数名を明示して可読性を優先する

### 運用
- 合成結果の音声ファイルは `Result/` に保存する
- 新しいファイルを追加する際は、既存の実装と重複がないか確認する
