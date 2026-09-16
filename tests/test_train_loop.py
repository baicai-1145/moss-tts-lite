"""End-to-end training loop smoke tests (CPU): full + lora + merge round-trip."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from moss_tts_lite.st_loader import read_safetensors
from moss_tts_lite import train as train_mod
from tests._train_fixtures import make_records, small_weights, write_small_model_dir

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    write_small_model_dir(str(model_dir))
    train_jsonl = tmp_path_factory.mktemp("data") / "train.jsonl"
    records = make_records(n=3, frames=8, seed=21)
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_root = tmp_path_factory.mktemp("out")
    return {"model_dir": str(model_dir), "train_jsonl": str(train_jsonl),
            "out_root": str(out_root), "base_weights": small_weights()}


def _run(env, mode, extra=()):
    out = os.path.join(env["out_root"], f"{mode}-run")
    argv = ["train",
            "--model-dir", env["model_dir"],
            "--train-jsonl", env["train_jsonl"],
            "--output-dir", out,
            "--mode", mode,
            "--max-steps", "2",
            "--per-device-batch-size", "1",
            "--gradient-accumulation-steps", "1",
            "--logging-steps", "1",
            "--device", "cpu",
            *extra]
    summary = train_mod.main(argv)
    return summary, out


def test_full_mode_two_steps(env):
    summary, out = _run(env, "full")
    assert summary["steps"] == 2
    assert len(summary["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(x)) for x in summary["losses"])
    # full model written with original key names + assets
    weights = read_safetensors(os.path.join(out))
    base = env["base_weights"]
    assert set(weights.keys()) == set(base.keys())
    assert os.path.exists(os.path.join(out, "vocab.json"))
    assert os.path.exists(os.path.join(out, "train_args.json"))
    # training actually moved the weights
    moved = [k for k in base
             if not torch.equal(weights[k].float(), base[k].float())]
    assert moved, "full SFT did not change any weights"


def test_lora_mode_and_merge_roundtrip(env):
    peft = pytest.importorskip("peft")
    summary, out = _run(env, "lora", ["--lora-rank", "4", "--lora-alpha", "8",
                                      "--merge-and-export"])
    assert summary["steps"] == 2
    assert os.path.exists(os.path.join(out, "adapter_model.safetensors"))
    assert os.path.exists(os.path.join(out, "adapter_config.json"))

    merged_path = os.path.join(out, "merged")
    assert os.path.exists(os.path.join(merged_path, "model.safetensors"))
    merged = read_safetensors(merged_path)
    base = env["base_weights"]
    assert set(merged.keys()) == set(base.keys())
    # LoRA delta actually applied to targeted projections
    changed = [k for k in base
               if not torch.equal(merged[k].float(), base[k].float())]
    assert any("q_proj" in k for k in changed)
    assert any(k not in changed for k in base)  # untouched params exist

    # independent merge entry point round-trips the same adapter
    out2 = os.path.join(env["out_root"], "merge2")
    train_mod.main(["merge", "--adapter-dir", out, "--model-dir", env["model_dir"],
                    "--output-dir", out2, "--device", "cpu"])
    merged2 = read_safetensors(out2)
    assert set(merged2.keys()) == set(base.keys())
    for k in base:
        assert torch.allclose(merged2[k].float(), merged[k].float(), atol=1e-5), k


def test_lora_only_trainables_move(env):
    pytest.importorskip("peft")
    from peft import PeftModel

    summary, out = _run(env, "lora", ["--lora-rank", "4", "--lora-alpha", "8"])
    adapter_sd = read_safetensors(os.path.join(out, "adapter_model.safetensors"))
    # adapter tensors are lora_A/lora_B on the target modules only
    assert adapter_sd, "empty adapter state dict"
    assert all("lora_" in k for k in adapter_sd)
    assert any("q_proj" in k for k in adapter_sd)


def test_qlora_two_steps_on_gpu(env):
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes QLoRA needs CUDA")
    summary, out = _run(env, "qlora", ["--bf16"], )
    assert summary["steps"] == 2
    assert all(torch.isfinite(torch.tensor(x)) for x in summary["losses"])
    assert os.path.exists(os.path.join(out, "adapter_model.safetensors"))


def test_max_audio_frames_filters_all(env):
    with pytest.raises(SystemExit):
        train_mod.main(["train",
                        "--model-dir", env["model_dir"],
                        "--train-jsonl", env["train_jsonl"],
                        "--output-dir", os.path.join(env["out_root"], "filter"),
                        "--mode", "full", "--max-steps", "1",
                        "--max-audio-frames", "4",
                        "--device", "cpu"])


def test_cli_help():
    import subprocess
    exe = sys.executable
    for args in (["-m", "moss_tts_lite.train", "--help"],
                 ["-m", "moss_tts_lite.train", "merge", "--help"],
                 ["-m", "moss_tts_lite.prepare_data", "--help"]):
        proc = subprocess.run([exe, *args], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert proc.returncode == 0, (args, proc.stderr)
        assert "usage" in proc.stdout.lower()


