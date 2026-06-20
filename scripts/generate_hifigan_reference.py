#!/usr/bin/env python3
"""Phase 3 — PyTorch HiFi-GAN Generator の中間 tensor 参照値を生成。

ANE 経路で振幅 4 倍に膨らむ HiFi-GAN の挙動を切り分けるため、
PyTorch CPU F32 で同じ入力を流して per-block の中間 tensor を `.npy` で保存する。

捕捉点（block レベル）:
- `conv_pre_out`: conv_pre 直後
- `upsample_0_*` 〜 `upsample_3_*`: 各 upsample stage 内部の境界
    - `leaky_relu`: leaky_relu(x) 直後
    - `ups`: ups[i](x) 直後（ConvTranspose 出力）
    - `resblock_j`: 各 resblock 出力（j = 0,1,2）
    - `stage_out`: xs / num_kernels の結果
- `pre_post_leaky`: 最終 leaky_relu 直後
- `conv_post_out`: conv_post 直後（tanh 前）
- `tanh_out`: 最終 tanh 直後（= wav サンプル）

入力: iOS の `postnet_output.npy` （shape [T, n_mels] = [262, 256]）
  → 内部で [1, n_mels, T] に転置して HiFi-GAN に渡す

実行例:
    cd PronounSE && venv/bin/python ../scripts/generate_hifigan_reference.py \\
        --postnet ../audio/iPhone_3_phase1_20260519/debug/20260519_212653_f16GpuRepeat1_Float16_cpuAndGPU/postnet_output.npy \\
        --out-dir ../data/2026-05-19/hifigan_reference/f16GpuRef
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "HiFiGAN"))  # utils_hifigan の絶対 import 用

from HiFiGAN.models import Generator  # noqa: E402
from HiFiGAN.env import AttrDict  # noqa: E402


def load_hifigan(checkpoint: Path, config: Path, device: str) -> Generator:
    with config.open() as f:
        param = AttrDict(json.load(f))
    t.manual_seed(param.seed)
    g = Generator(param).to(device)
    chkpt = t.load(checkpoint, map_location=device)
    g.load_state_dict(chkpt["generator"])
    g.eval()
    g.remove_weight_norm()
    return g


class HookedGenerator:
    """Generator の forward を再現しつつ、各段の中間 tensor を `state` に蓄積する。

    Generator.forward を直接走らせる代わりに forward の中身を再実装するので、
    upsample stage 内の per-resblock / per-leaky_relu の捕捉点まで取れる。
    """

    def __init__(self, generator: Generator):
        import torch.nn.functional as F
        self.g = generator
        self.F = F
        self.LRELU_SLOPE = 0.1  # HiFi-GAN 標準

    def forward(self, x: t.Tensor) -> tuple[t.Tensor, dict]:
        state: dict = {}
        g = self.g
        x = g.conv_pre(x)
        state["conv_pre_out"] = x.detach().cpu().numpy().copy()
        for i in range(g.num_upsamples):
            x_in = x
            x = self.F.leaky_relu(x_in, self.LRELU_SLOPE)
            state[f"upsample_{i}_leaky_relu"] = x.detach().cpu().numpy().copy()
            x = g.ups[i](x)
            state[f"upsample_{i}_ups"] = x.detach().cpu().numpy().copy()
            xs = None
            for j in range(g.num_kernels):
                res = g.resblocks[i * g.num_kernels + j](x)
                state[f"upsample_{i}_resblock_{j}"] = res.detach().cpu().numpy().copy()
                if xs is None:
                    xs = res
                else:
                    xs = xs + res
            x = xs / g.num_kernels
            state[f"upsample_{i}_stage_out"] = x.detach().cpu().numpy().copy()
        # conv_post 直前のこの活性化だけは、オリジナル HiFi-GAN (models.py Generator.forward)
        # が slope を明示せず default(0.01) を使う仕様。実モデルを忠実に再現するため
        # ここも LRELU_SLOPE(0.1) ではなく default のままにする（変更しないこと）。
        x = self.F.leaky_relu(x)
        state["pre_post_leaky"] = x.detach().cpu().numpy().copy()
        x = g.conv_post(x)
        state["conv_post_out"] = x.detach().cpu().numpy().copy()
        x = t.tanh(x)
        state["tanh_out"] = x.detach().cpu().numpy().copy()
        return x, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postnet", required=True, type=Path,
                        help="iOS の postnet_output.npy ([T, n_mels])")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "PronounSE/HiFiGAN/chkpt/g_00009000")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "PronounSE/HiFiGAN/chkpt/config.json")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data/2026-05-19/hifigan_reference")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.postnet.exists():
        print(f"# postnet not found: {args.postnet}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    g = load_hifigan(args.checkpoint, args.config, args.device)
    hooked = HookedGenerator(g)

    arr = np.load(args.postnet).astype(np.float32)
    if arr.ndim == 2:
        # [T, n_mels] -> [1, T, n_mels]
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        print(f"# unexpected postnet shape: {arr.shape}", file=sys.stderr)
        return 1
    # HiFi-GAN expects [B, n_mels, T]
    x = t.from_numpy(arr).to(args.device).transpose(1, 2)
    print(f"# input shape (after transpose): {tuple(x.shape)}")

    with t.no_grad():
        y, state = hooked.forward(x)

    # 統計サマリー
    print(f"# {'name':30s} {'shape':>20s} {'min':>12s} {'max':>12s} {'mean':>12s} {'rms':>10s}")
    summary = {}
    for name, val in state.items():
        v = val.astype(np.float64).flatten()
        rms = float(np.sqrt((v * v).mean()))
        summary[name] = {
            "shape": list(val.shape),
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "rms": rms,
        }
        np.save(args.out_dir / f"{name}.npy", val)
        print(f"  {name:30s} {str(val.shape):>20s} {v.min():>12.6f} {v.max():>12.6f} "
              f"{v.mean():>12.6f} {rms:>10.6f}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"# saved {len(state)} intermediate tensors to {args.out_dir}")
    print(f"# final tanh_out max={summary['tanh_out']['max']:.4f} rms={summary['tanh_out']['rms']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
