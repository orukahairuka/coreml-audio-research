#!/usr/bin/env python3
"""Phase 3 — PyTorch Decoder の中間 tensor 参照値を生成するスクリプト。

iOS で動いた `mel_normalized.npy` を入力に、PyTorch CPU F32 で同じ自己回帰ループを
走らせ、各境界（prenet / norm / selfattn×3 / dotattn×3 / ffn×3 / mel_linear / postnet）
の中間 tensor を `.npy` で書き出す。これを CoreML 各設定の最終出力と diff することで、
divergence が起きる layer を切り分ける。

計画書: `docs/2026-05-19/all-engine-precision-stability-plan.md` §5

実行例 (PronounSE の venv から):

    cd PronounSE && venv/bin/python ../scripts/generate_decoder_reference.py \\
        --mel-npy ../audio/<archive>/debug/<runId>/mel_normalized.npy \\
        --out-dir ../data/2026-05-19/decoder_reference/<tag>

出力:
    <out-dir>/encoder_output.npy
    <out-dir>/step_000/{prenet,norm,selfattn_0,...,postconvnet}.npy
    <out-dir>/step_001/...
    <out-dir>/step_002/...
    <out-dir>/decoder_step_stats.json  各 step の mel_min/max/mean
    <out-dir>/postnet_output_final.npy  最終 step の postnet 出力

`--capture-steps` でキャプチャ対象 step を指定可（既定 `0,1,2`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))

from Transformer.network import Model  # noqa: E402
from Transformer import hyperparams as hp  # noqa: E402


def load_model(checkpoint_path: Path, device: str) -> Model:
    m = Model(hp.prenet_type).to(device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    m.load_state_dict(checkpoint["model"])
    m.eval()
    return m


def load_mel_normalized(npy_path: Path) -> t.Tensor:
    """iOS の NpyWriter が書いた `mel_normalized.npy` を読む。

    iOS 側は [T, n_mels] の 2D 形式で書く（DebugRunSnapshot.writeFloat2D）。
    Encoder が期待するのは [1, T, n_mels] なので必要なら次元を足す。
    """
    arr = np.load(npy_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"unexpected mel shape: {arr.shape}")
    return t.from_numpy(arr.astype(np.float32))


def register_capture_hooks(
    decoder: nn.Module,
    target_steps: list[int],
    state: dict,
) -> list:
    """各 layer に forward hook を仕込んで、`target_steps` に該当する step の出力を保存する。"""
    handles = []

    def make_hook(name: str):
        def _hook(_module, _input, output):
            cur = state.get("current_step")
            if cur in target_steps:
                if isinstance(output, tuple):
                    # Attention は (out, attn) を返す
                    out = output[0]
                else:
                    out = output
                step_caps = state.setdefault("captures", {}).setdefault(cur, {})
                step_caps[name] = out.detach().cpu().numpy().copy()
        return _hook

    handles.append(decoder.decoder_prenet.register_forward_hook(make_hook("prenet")))
    handles.append(decoder.norm.register_forward_hook(make_hook("norm")))
    for i, layer in enumerate(decoder.selfattn_layers):
        handles.append(layer.register_forward_hook(make_hook(f"selfattn_{i}")))
    for i, layer in enumerate(decoder.dotattn_layers):
        handles.append(layer.register_forward_hook(make_hook(f"dotattn_{i}")))
    for i, layer in enumerate(decoder.ffns):
        handles.append(layer.register_forward_hook(make_hook(f"ffn_{i}")))
    handles.append(decoder.mel_linear.register_forward_hook(make_hook("mel_linear")))
    handles.append(decoder.postconvnet.register_forward_hook(make_hook("postconvnet")))
    return handles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-npy", required=True, help="iOS が書いた mel_normalized.npy のパス")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "PronounSE/Transformer/chkpt/chkpt__20000.pth.tar"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data/2026-05-19/decoder_reference"),
    )
    parser.add_argument(
        "--capture-steps",
        default="0,1,2",
        help="カンマ区切りで capture 対象 step を指定（既定 0,1,2）",
    )
    parser.add_argument("--device", default="cpu", help="cpu / mps")
    parser.add_argument("--max-steps", type=int, default=0, help="0=全 step, それ以外は途中で打ち切り")
    args = parser.parse_args()

    capture_steps = [int(s) for s in args.capture_steps.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.checkpoint), args.device)
    mel_normalized = load_mel_normalized(Path(args.mel_npy)).to(args.device)
    total_T = mel_normalized.shape[1]
    print(f"# mel input shape: {tuple(mel_normalized.shape)}")
    pos_src = t.arange(1, total_T + 1).unsqueeze(0).to(args.device)

    state: dict = {"current_step": -1, "captures": {}}
    hooks = register_capture_hooks(model.decoder, capture_steps, state)

    with t.no_grad():
        memory, c_mask = model.procEncoder(mel_normalized, pos_src)
    np.save(out_dir / "encoder_output.npy", memory.detach().cpu().numpy())
    print(f"# saved encoder_output.npy shape={tuple(memory.shape)}")

    n_steps = total_T if args.max_steps <= 0 else min(total_T, args.max_steps)
    decoder_input = t.zeros(1, 1, hp.n_mels).to(args.device)
    mel_step_stats: list[dict] = []
    final_postnet = None
    final_mel = None

    with t.no_grad():
        for step in range(n_steps):
            state["current_step"] = step
            pos_trg = t.arange(1, decoder_input.size(1) + 1).unsqueeze(0).to(args.device)
            mel_pred, postnet_pred = model.procDecoder(memory, decoder_input, pos_trg, c_mask)
            mel_last = mel_pred[:, -1:, :].detach().cpu().numpy()
            mel_step_stats.append({
                "step": step,
                "mel_min": float(mel_last.min()),
                "mel_max": float(mel_last.max()),
                "mel_mean": float(mel_last.mean()),
            })
            decoder_input = t.cat([decoder_input, mel_pred[:, -1:, :]], dim=1)
            if step == n_steps - 1:
                final_postnet = postnet_pred.detach().cpu().numpy()
                final_mel = mel_pred.detach().cpu().numpy()

    for h in hooks:
        h.remove()

    for step in sorted(state.get("captures", {}).keys()):
        step_dir = out_dir / f"step_{step:03d}"
        step_dir.mkdir(exist_ok=True)
        captures = state["captures"][step]
        for name, arr in captures.items():
            np.save(step_dir / f"{name}.npy", arr)
        print(f"# saved {len(captures)} captures for step {step} -> {step_dir}")

    with (out_dir / "decoder_step_stats.json").open("w") as f:
        json.dump(mel_step_stats, f, indent=2)
    print(f"# saved decoder_step_stats.json ({len(mel_step_stats)} steps)")

    if final_postnet is not None:
        np.save(out_dir / "postnet_output_final.npy", final_postnet)
        print(f"# saved postnet_output_final.npy shape={final_postnet.shape}")
    if final_mel is not None:
        np.save(out_dir / "mel_out_final.npy", final_mel)
        print(f"# saved mel_out_final.npy shape={final_mel.shape}")

    print(f"# done. results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
