"""Release gates:"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import gc
import json
import shutil
import tempfile
from pathlib import Path

import torch

from moss_tts_lite.cli import (
    DEFAULT_MAX_SEQ_LEN,
    KV_RAMP_MARGIN,
    _kv_mib,
    _resolve_max_seq_len,
)
from moss_tts_lite.export import (FORMAT_TAG, PRESET_METRICS, Q_FILE, export_standalone,
                      is_standalone_dir, read_standalone, render_model_card,
                      standalone_presets, write_safetensors)
from moss_tts_lite.fast import generate_fast
from moss_tts_lite.gptq import load_gptq_fast
from moss_tts_lite.gptq import pack_fast, rtn_quantize
from moss_tts_lite.model import MossTTSModel
from moss_tts_lite.prompt import build_tts_prompt
from moss_tts_lite.st_loader import read_safetensors, safetensors_header

ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
LITE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = LITE_ROOT
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
STATE = os.path.join(MODEL_DIR, "gptq", "w1p.pt")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
GIB = 2 ** 30
SEED = 1234
MAXNEW = 4096
OUT = os.path.join(ROOT, ".tmp", "kvfit_agent", "wav")
TEMPLATE = os.path.join(REPO, "README_hf.md")

PKG_ROOT = Path(__file__).resolve().parents[1] / "moss_tts_lite"
ALLOWED_THIRD_PARTY = {"torch", "numpy", "soundfile", "yaml"}
# Training-only modules (optional deps, lazy imports -- never imported by the
# inference core). They may import peft/bitsandbytes/transformers/torchaudio.
TRAINING_MODULES = {"nn.py", "data.py", "train.py", "prepare_data.py"}
TRAIN_ONLY_THIRD_PARTY = {"peft", "bitsandbytes", "transformers", "torchaudio",
                          "accelerate", "safetensors"}

def _collect_imports(path: Path) -> list[tuple[str, int, str]]:
    """Return (top_level_module, lineno, shown_name) for every Import / ImportFrom node anywhere in the file (module body, f..."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append(("moss_tts_lite", node.lineno, "." * node.level + (node.module or "")))
            else:
                mod = node.module or ""
                found.append((mod.split(".")[0], node.lineno, mod))
    return found

def test_dep_purity():
    stdlib = sys.stdlib_module_names
    files = sorted(PKG_ROOT.rglob("*.py"))
    assert files, f"no .py files found under {PKG_ROOT}"

    violations: list[str] = []
    seen: dict[str, set[str]] = {}

    for f in files:
        rel = f.relative_to(PKG_ROOT.parent).as_posix()
        is_training_module = f.name in TRAINING_MODULES
        try:
            imports = _collect_imports(f)
        except SyntaxError as e:
            raise AssertionError(f"{rel}: unparsable: {e}") from e
        for top, lineno, name in imports:
            seen.setdefault(top, set()).add(rel)
            allowed = ALLOWED_THIRD_PARTY
            if is_training_module:
                allowed = allowed | TRAIN_ONLY_THIRD_PARTY
            if top not in stdlib and top not in allowed and top != "moss_tts_lite":
                violations.append(f"{rel}:{lineno}: import {name!r} (top={top!r})")

    assert not violations, (
        "moss_tts_lite imports outside torch/numpy/soundfile/yaml + stdlib:\n"
        + "\n".join(violations))

    third = sorted(t for t in seen if t in ALLOWED_THIRD_PARTY)
    own = sorted(t for t in seen if t == "moss_tts_lite")
    all_allowed = ALLOWED_THIRD_PARTY | TRAIN_ONLY_THIRD_PARTY
    other = sorted(t for t in seen if t not in stdlib
                   and t not in all_allowed and t != "moss_tts_lite")
    # the inference core must never import the training extras or the
    # training-only modules
    core_files = [f for f in files if f.name not in TRAINING_MODULES]
    for f in core_files:
        rel = f.relative_to(PKG_ROOT.parent).as_posix()
        for top, lineno, name in _collect_imports(f):
            assert top not in TRAIN_ONLY_THIRD_PARTY, \
                f"{rel}:{lineno}: inference core imports training dep {name!r}"
            if top == "moss_tts_lite":
                assert ".nn" not in name and not name.endswith(".train") \
                    and ".data" not in name and ".prepare_data" not in name, \
                    f"{rel}:{lineno}: inference core imports training module {name!r}"
    std_used = sorted(t for t in seen if t in stdlib)
    print(f"  scanned {len(files)} files under {PKG_ROOT.name}/")
    print(f"  third-party: {third}")
    print(f"  own package: {own}")
    print(f"  stdlib used ({len(std_used)}): {std_used}")
    assert not other, f"unexpected modules: {other}"

def main_dep_purity() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_dep_purity()
    print("ALL TESTS PASSED")
    return 0

N_LAYERS = 2
HIDDEN = 256
HEAD_DIM = 32
N_HEADS = 4
N_KV = 2
TEXT_VOCAB = 128
PROJS = ("q", "k", "v", "o", "gate", "up", "down")
_SRC = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
        "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
        "down": "mlp.down_proj"}
_LIN_SHAPE = {"q": (N_HEADS * HEAD_DIM, HIDDEN), "k": (N_KV * HEAD_DIM, HIDDEN),
              "v": (N_KV * HEAD_DIM, HIDDEN), "o": (HIDDEN, N_HEADS * HEAD_DIM),
              "gate": (2 * HIDDEN, HIDDEN), "up": (2 * HIDDEN, HIDDEN),
              "down": (HIDDEN, 2 * HIDDEN)}

def _bf16(shape, seed, scale=0.05):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

def _synthetic_base(dirpath: str) -> dict:
    """A miniature MOSS-TTS checkpoint with the exact key naming scheme."""
    tensors: dict[str, torch.Tensor] = {}
    tensors["language_model.embed_tokens.weight"] = _bf16((TEXT_VOCAB, HIDDEN), 1)
    tensors["language_model.norm.weight"] = _bf16((HIDDEN,), 2)
    for li in range(N_LAYERS):
        p = f"language_model.layers.{li}"
        tensors[f"{p}.input_layernorm.weight"] = _bf16((HIDDEN,), 10 + li)
        for proj in PROJS:
            tensors[f"{p}.{_SRC[proj]}.weight"] = _bf16(_LIN_SHAPE[proj], 20 + li * 7 + len(proj))
        tensors[f"{p}.self_attn.q_norm.weight"] = _bf16((HEAD_DIM,), 40 + li)
        tensors[f"{p}.self_attn.k_norm.weight"] = _bf16((HEAD_DIM,), 50 + li)
        tensors[f"{p}.post_attention_layernorm.weight"] = _bf16((HIDDEN,), 60 + li)
    for i in range(32):
        tensors[f"emb_ext.{i}.weight"] = _bf16((1025, HIDDEN), 100 + i)
    for i in range(33):
        tensors[f"lm_heads.{i}.weight"] = _bf16((TEXT_VOCAB if i == 0 else 1025, HIDDEN),
                                                 200 + i)
    os.makedirs(dirpath, exist_ok=True)

    write_safetensors(os.path.join(dirpath, "model.safetensors"), sorted(tensors.items()))
    with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {k: "model.safetensors" for k in sorted(tensors)}}, f)
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump({"torch_dtype": "bfloat16"}, f)
    for name in ("vocab.json", "merges.txt", "tokenizer.json"):
        with open(os.path.join(dirpath, name), "w") as f:
            f.write(f"# synthetic {name}\n")
    return tensors

def _pack(q, s, mn, kt=8, dev=None):
    """`pack_fast`, but shape-faithful on a machine without the CUDA kernel."""
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        qp = q[:, 1::2] | (q[:, 0::2] << 4)
        shp = torch.ops.aten._convert_weight_to_int4pack(
            qp.contiguous().to("meta"), kt).shape
        gen = torch.Generator().manual_seed(int(q.float().abs().sum().item()))
        packed = torch.randint(0, 2**31 - 1, tuple(shp), generator=gen,
                               dtype=torch.int32)
        kk = q.shape[1] // s.shape[1]    # noqa: F841 (documents the group layout)
        qsz = torch.stack([s, mn + 8.0 * s], -1).bfloat16().transpose(0, 1).contiguous()
        return packed, qsz
    return pack_fast(q.to(dev), s.to(dev), mn.to(dev), kt)

def _synthetic_state(base: dict, group=32, keep_v=True) -> tuple[dict, dict]:
    """RTN-pack every projection except v_proj (bf16 keep), like w1."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state, gmap, keep = {}, {}, []
    for li in range(N_LAYERS):
        state[li] = {}
        for proj in PROJS:
            if keep_v and proj == "v":
                keep.append(f"{li}:v")
                continue
            key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
            q, s, mn = rtn_quantize(base[key], group)
            packed, qsz = _pack(q, s, mn, 8, dev)
            state[li][proj] = {"packed": packed.cpu(), "qsz": qsz.cpu()}
            gmap[f"{li}:{proj}"] = group
    return state, {"group_size_map": gmap, "bf16_linears": keep,
                   "bf16_layers": []}

def _write_state(path: str, state: dict, meta: dict) -> None:
    torch.save(state, path)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)

def phase_e1_writer_roundtrip() -> bool:
    print("E1 writer/reader round-trip (all dtypes, metadata, 3-D payload)")
    tmp = tempfile.mkdtemp(prefix="export_e1_")
    try:
        tensors = {
            "f32": torch.linspace(-3, 3, 1000, dtype=torch.float32).reshape(10, 100),
            "f64": torch.randn(7, dtype=torch.float64),
            "i32_packed": torch.randint(-2**31, 2**31 - 1, (6, 8), dtype=torch.int32),
            "u8": torch.randint(0, 255, (5, 3), dtype=torch.uint8),
            "i8": torch.randint(-127, 127, (4,), dtype=torch.int8),
            "bf16": _bf16((3, 5), 7),
            "bool": torch.tensor([True, False, True]),
            "empty": torch.zeros(0, dtype=torch.bfloat16),
            "three_d": torch.arange(24, dtype=torch.int32).reshape(2, 3, 4),
        }
        path = os.path.join(tmp, "t.safetensors")
        size, sha = write_safetensors(path, sorted(tensors.items()),
                                     metadata={"format": FORMAT_TAG, "preset": "w1"})
        ok = True
        hdr = safetensors_header(path)
        ok &= hdr.get("__metadata__", {}).get("preset") == "w1"
        ok &= len([k for k in hdr if k != "__metadata__"]) == len(tensors)
        with open(path, "rb") as f:
            import struct
            (n,) = struct.unpack("<Q", f.read(8))
        ok &= (8 + n) % 8 == 0 and size == os.path.getsize(path)
        back = read_safetensors(path)
        for name, t in sorted(tensors.items()):
            r = back[name]
            same = (r.dtype == t.dtype and tuple(r.shape) == tuple(t.shape)
                    and torch.equal(r.view(torch.uint8), t.contiguous().view(torch.uint8)))
            print(f"  {name:12s} {str(t.dtype):14s} {tuple(t.shape)} bytes-equal={same}")
            ok &= same

        ok &= "__metadata__" not in back
        print(f"  E1: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def phase_e2_export_roundtrip() -> bool:
    print("E2 export end-to-end on a synthetic checkpoint")
    tmp = tempfile.mkdtemp(prefix="export_e2_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        emeta = export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                                  hash_base=True, template_path=TEMPLATE,
                                  quiet=True)
        checks: dict[str, bool] = {}
        checks["files_present"] = all(os.path.isfile(os.path.join(out, f))
                                      for f in (Q_FILE, "meta.json", "README.md",
                                                "LICENSE", "vocab.json"))
        checks["format_tag"] = emeta["format"] == FORMAT_TAG
        checks["preset_name"] = emeta["preset_name"] == "w1"
        checks["base_n_keys"] = emeta["base"]["n_keys"] == len(base)
        checks["base_sha256"] = all("sha256" in s for s in emeta["base"]["shards"])
        p = emeta["presets"]["w1"]
        checks["bf16_keeps"] = p["bf16_linears"] == [f"{li}:v" for li in range(N_LAYERS)]
        checks["group_map"] = set(p["group_size_map"]) == {
            f"{li}:{pr}" for li in range(N_LAYERS) for pr in PROJS if pr != "v"}
        checks["heads"] = (emeta["model_config"]["n_heads"] == N_HEADS
                           and emeta["model_config"]["n_kv_heads"] == N_KV
                           and emeta["model_config"]["hidden_size"] == HIDDEN)
        print(f"  meta: preset={emeta['preset_name']} tensors={emeta['n_tensors']} "
              f"bf16_keeps={len(p['bf16_linears'])} "
              f"quantized={p['n_quantized_linears']}")

        rmeta, weights, rstate, gmap, keep = read_standalone(out)
        n_q = n_bad = 0
        for li in sorted(state):
            for proj, rec in state[li].items():
                n_q += 2
                n_bad += int(not torch.equal(rstate[li][proj]["packed"], rec["packed"]))
                n_bad += int(not torch.equal(rstate[li][proj]["qsz"], rec["qsz"]))
        print(f"  int4 payloads: {n_q} tensors compared, {n_bad} mismatch")
        checks["int4_roundtrip"] = n_bad == 0
        n_b = n_bad_b = 0
        for key, t in base.items():
            if key in weights:
                n_b += 1
                n_bad_b += int(not torch.equal(weights[key], t))
        n_kept = sum(1 for k in base
                     if k.endswith(tuple(f".{s}.weight" for s in _SRC.values()))
                     and k in weights)
        print(f"  bf16 tensors: {n_b} compared ({n_kept} of them backbone "
              f"projections by original key), {n_bad_b} mismatch")
        checks["bf16_bytes"] = n_bad_b == 0
        checks["quantized_count"] = emeta["presets"]["w1"]["n_quantized_linears"] \
            == len(state) * (len(PROJS) - 1)
        checks["keeps_reread"] = keep == [(li, "v") for li in range(N_LAYERS)]
        checks["gmap_reread"] = gmap == {(li, pr): 32 for li in range(N_LAYERS)
                                         for pr in PROJS if pr != "v"}
        failed = sorted(k for k, v in checks.items() if not v)
        if failed:
            print(f"  failed checks: {failed}")
        print(f"  E2: {'PASS' if not failed else 'FAIL'}")
        return not failed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def phase_e3_patch_path_equivalence() -> bool:
    """The export must be *the same bytes* the patch path uses, and they must still be the bytes an independent re-quantizat..."""
    print("E3 equivalence with the patch path (byte-level)")
    tmp = tempfile.mkdtemp(prefix="export_e3_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                          hash_base=False, quiet=True)
        patch = torch.load(spath, map_location="cpu", weights_only=False)
        _m, sweights, sstate, sgmap, skeep = read_standalone(out)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ok = True
        n = 0
        for li in sorted(patch):
            for proj in sorted(patch[li]):
                a, b = patch[li][proj], sstate[li][proj]
                g = sgmap[(li, proj)]

                ok &= torch.equal(a["packed"], b["packed"])
                ok &= torch.equal(a["qsz"], b["qsz"])

                key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
                q, s, mn = rtn_quantize(base[key], g)
                packed, qsz = _pack(q, s, mn, 8, dev)
                ok &= torch.equal(b["packed"], packed.cpu())
                ok &= torch.equal(b["qsz"], qsz.cpu())
                n += 1

        for li, proj in skeep:
            key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
            ok &= torch.equal(sweights[key], base[key])
        print(f"  {n} projections: patch-state == standalone == fresh RTN "
              f"re-quantization (bit-exact, payload device={dev}), "
              f"bf16 keeps untouched")
        print(f"  E3: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def phase_e4_loader_guards() -> bool:
    print("E4 loader guards")
    tmp = tempfile.mkdtemp(prefix="export_e4_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                          hash_base=False, quiet=True)
        ok = is_standalone_dir(out) and not is_standalone_dir(tmp)
        ok &= not is_standalone_dir(os.path.join(tmp, "base"))
        ok &= standalone_presets(out) == ["w1"]

        try:
            read_standalone(out, expected_preset="w2")
            ok = False
        except SystemExit:
            pass

        moved = os.path.join(tmp, "quantized.moved")
        os.rename(os.path.join(out, Q_FILE), moved)
        try:
            read_standalone(out)
            ok = False
        except FileNotFoundError:
            pass
        os.rename(moved, os.path.join(out, Q_FILE))

        raw = read_safetensors(os.path.join(out, Q_FILE))
        trimmed = {k: v for k, v in raw.items() if k != "layers.0.q.qsz"}
        write_safetensors(os.path.join(out, "trimmed.safetensors"),
                          sorted(trimmed.items()))
        mpath = os.path.join(out, "meta.json")
        m = json.load(open(mpath))
        m["weight_file"] = dict(m["weight_file"], name="trimmed.safetensors")
        with open(mpath, "w") as f:
            json.dump(m, f)
        del raw, trimmed
        try:
            read_standalone(out)
            ok = False
        except ValueError as e:
            ok &= "incomplete" in str(e)

        out2 = os.path.join(tmp, "export2")
        export_standalone(os.path.join(tmp, "base"), spath, out2, "w1",
                          hash_base=False, quiet=True)
        _m, _w, st, _g, _k = read_standalone(out2)
        keys = {k for rec in st.values() for r in rec.values() for k in r}
        ok &= keys == {"packed", "qsz"}
        print(f"  is_standalone / presets / mismatch / missing file / "
              f"incomplete record / record keys={'PASS' if ok else 'FAIL'}")
        print(f"  E4: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

_M5_META = {
    "created_utc": "2025-09-11T00:00:00Z",
    "n_tensors": 679,
    "weight_file": {"bytes": 7636941152, "sha256": "ab" * 32},
    "quantization": {"q_dtype": "I32"},
    "model_config": {"hidden_size": 4096},
    "base": {"local_dir": "models/MOSS-TTS-v1.5", "shards": [
        {"name": "model-00001-of-00004.safetensors", "bytes": 4932667368,
         "sha256": "cd" * 32}]},
    "base_repo": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "tool": {"torch": "2.9.1"},

    "state": {"source_file": "w1.pt", "source_bytes": 4246863871,
              "sha256": "ef" * 32},
}

def phase_e5_model_card() -> bool:
    print("E5 model card rendering")
    ok = True
    tpl = open(TEMPLATE, encoding="utf-8").read()
    for preset in sorted(PRESET_METRICS):
        meta = dict(_M5_META)
        meta["presets"] = {preset: {
            "label": PRESET_METRICS[preset]["label"],
            "group_size_default": PRESET_METRICS[preset]["group_size"],
            "bf16_linears": ["0:v"], "bf16_layers": [],
            "n_quantized_linears": 216,
            "metrics": PRESET_METRICS[preset]}}
        card = render_model_card(tpl, meta, preset)
        ok &= "{{" not in card and "}}" not in card
        ok &= f"{PRESET_METRICS[preset]['gate2_audio_top25_pct']:.2f}" in card
        ok &= f"{PRESET_METRICS[preset]['steps_per_s_steady']:.1f}" in card
        ok &= "Apache-2.0" in card
        ok &= "跨语言验证" in card

        ok &= "n/a" not in card

        ok &= "`w1.pt`" in card and "ef" * 16 in card
        print(f"  {preset}: {len(card)} chars, no leftover placeholder, "
              f"metrics + cross-language section present, no n/a, "
              f"state file named")

    try:
        render_model_card("{{NOT_A_PLACEHOLDER}}", meta, preset)
        ok = False
    except ValueError:
        pass

    meta_tie = dict(_M5_META)
    meta_tie["presets"] = {"w2": {
        "label": PRESET_METRICS["w2"]["label"], "group_size_default": 32,
        "bf16_linears": ["0:v"], "n_quantized_linears": 216,
        "metrics": {**PRESET_METRICS["w2"], "tie_robust_cover_pct": 99.3,
                    "audio_mean_abs_logit_delta": 0.2641,
                    "tie_robust_cover_gain_lang_pt": 0.545,
                    "tie_robust_cover_gain_emo_pt": 1.02}}}
    card_tie = render_model_card(tpl, meta_tie, "w2")
    ok &= "tie-robust `cover`" in card_tie
    ok &= "### 3.1 关于 top-25 数字的口径" in card_tie
    ok &= "跨语言验证" in card_tie
    ok &= "n/a" not in card_tie
    print(f"  w2+tie metrics: {len(card_tie)} chars, tie rows + §3.1 note + "
          f"cross-language section, no n/a")
    print(f"  E5: {'PASS' if ok else 'FAIL'}")
    return bool(ok)

def phase_e6_purity() -> bool:
    print("E6 export.py dependency purity")
    import ast
    path = os.path.join(REPO, "moss_tts_lite", "export.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    allowed = {"torch", "numpy", "json", "os", "re", "shutil", "struct", "sys",
               "time", "hashlib", "argparse", "__future__"}
    extra = sorted(mods - allowed)
    top = sorted(mods)
    ok = not extra

    ok &= "safetensors" not in mods
    print(f"  top-level imports: {top}")
    print(f"  disallowed: {extra}")
    print(f"  E6: {'PASS' if ok else 'FAIL'}")
    return bool(ok)

def main_export() -> int:
    results = {
        "E1_writer": phase_e1_writer_roundtrip(),
        "E2_export": phase_e2_export_roundtrip(),
        "E3_equiv": phase_e3_patch_path_equivalence(),
        "E4_guards": phase_e4_loader_guards(),
        "E5_card": phase_e5_model_card(),
        "E6_purity": phase_e6_purity(),
    }
    ok = all(results.values())
    print("test_export: " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                     for k, v in results.items())
          + f" -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

STANDALONE = os.environ.get("MOSS_TTS_8GB_DIR",
    "/root/MOSS-TTS/models_export/MOSS-TTS-v1.5-W4GPTQ-w1")

_PEAK_GIB = 7.90
_PEAK_CODEC_GIB = 3.80
_PEAK_BASE_GIB = 8.45

EIGHT_GB_CARD = 8.0

CASES = [

    ("zh12", "你好，欢迎收听这段试音。", None, MAXNEW),
    ("en", "Hello, this is a short test of the text to speech system.",
     "English", MAXNEW),
    ("pause", "我今天学习了一首中国的古诗，它的名字是[pause 8s]静夜思！", None, MAXNEW),
    ("long150",
     "人工智能正在改变我们的生活方式。从智能手机到自动驾驶，从医疗诊断到金融风控，"
     "机器学习算法已经渗透到各行各业。与此同时，人们也开始关注数据隐私、算法公平和"
     "就业结构等社会问题。算法推荐系统在提升信息获取效率的同时，也可能造成信息茧房"
     "效应。如何在个性化与公共性之间取得平衡，是这个时代需要认真思考的问题。",
     None, MAXNEW),
]

EIGHT_GB_BUDGET = 2048

def _smi_total_mib() -> int:
    """Total nvidia-smi footprint of EVERY process on the device."""
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    total = 0
    for line in out.strip().splitlines():
        if line:
            total += int(line.split(",")[1].strip().split()[0])
    return total

_SMI_BASELINE_MIB = 0

def _smi_mib() -> int:
    """This process's own nvidia-smi footprint (includes the graph pool)."""
    return _smi_total_mib() - _SMI_BASELINE_MIB

def _start_ctx_gib() -> float:
    """Create the CUDA context and return its cost in GiB."""
    global _SMI_BASELINE_MIB
    _SMI_BASELINE_MIB = _smi_total_mib()
    torch.cuda.init()
    probe = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    return (_smi_total_mib() - _SMI_BASELINE_MIB) / 1024

def phase_0_sizing() -> bool:
    """The on-demand size arithmetic, over the combination matrix."""
    print("=== Phase 0: KV-size derivation (unit, no GPU) ===")
    ok = True
    rows = [

        (None, 66, 4096, 66 + 4096 + 64, "CLI default, short zh"),
        (None, 143, 4096, 143 + 4096 + 64, "CLI default, long zh"),
        (None, 71, 4096, 71 + 4096 + 64, "CLI default, en"),
        (None, 185, 4096, 185 + 4096 + 64, "continuation (120-frame prefix)"),
        (None, 66, 1024, 66 + 1024 + 64, "reduced budget"),
        (None, 66, 0, 66 + 64, "no generation at all"),
        (1024, 66, 4096, 66 + 4096 + 64, "floor BELOW the need: need wins"),
        (4226, 66, 4096, 4226, "floor EQUAL to the need: exact"),
        (8192, 66, 4096, 8192, "floor ABOVE the need: user wins (legacy size)"),
        (16384, 143, 4096, 16384, "floor far above: user wins"),
        (None, 1, 1, 1 + 1 + 64, "degenerate but sane"),
    ]
    for requested, l0, mnt, want, why in rows:
        got = _resolve_max_seq_len(requested, l0, mnt)
        good = got == want
        ok &= good
        print(f"  floor={str(requested):6s} L0={l0:4d} max_new={mnt:5d} -> "
              f"{got:6d} (want {want:6d}) {'ok' if good else 'FAIL'}  # {why}")

    n_vq = 32
    print(f"  KV_RAMP_MARGIN={KV_RAMP_MARGIN} >= n_vq(={n_vq}) + audio_end rows: "
          f"{KV_RAMP_MARGIN >= n_vq + 2}")
    ok &= KV_RAMP_MARGIN >= n_vq + 2

    print(f"  library default (synthesize/MossTTSModel) stays {DEFAULT_MAX_SEQ_LEN}: "
          f"{DEFAULT_MAX_SEQ_LEN == 8192}")
    ok &= DEFAULT_MAX_SEQ_LEN == 8192

    for n, want in ((8192, 1152.0), (1024, 144.0)):
        got = _kv_mib(n)
        good = abs(got - want) < 1e-6
        ok &= good
        print(f"  _kv_mib({n}) = {got:.1f} MiB (want {want:.1f}) "
              f"{'ok' if good else 'FAIL'}")

    w = _tiny_weights()
    m_small = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=300)
    m_big = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=8192)
    rope_ok = bool(torch.equal(m_small.rope_cos, m_big.rope_cos[:300])
                   and torch.equal(m_small.rope_sin, m_big.rope_sin[:300]))
    m_grow = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=300)
    m_grow.ensure_seq_len(8192)
    grow_ok = bool(torch.equal(m_grow.rope_cos, m_big.rope_cos)
                   and torch.equal(m_grow.rope_sin, m_big.rope_sin)
                   and m_grow.k_cache.shape == m_big.k_cache.shape)

    noop_ok = m_grow.ensure_seq_len(1024) == 8192

    m_grow.reset()
    m_grow._seq = 4
    m_grow.k_cache[:, :, :4] = 7.0
    m_grow.ensure_seq_len(16384)
    live_ok = bool((m_grow.k_cache[:, :, :4] == 7.0).all()) \
        and bool((m_grow.k_cache[:, :, 8192:] == 0).all())
    for label, good in (("rope[:n] == rope built at n", rope_ok),
                        ("ensure_seq_len reproduces a bigger model", grow_ok),
                        ("no-op resize changes nothing", noop_ok),
                        ("growth keeps live KV, zeroes the tail", live_ok)):
        ok &= good
        print(f"  resize identity: {label:44s} {'ok' if good else 'FAIL'}")
    del m_small, m_big, m_grow, w
    gc.collect()
    print(f"  phase 0 gate: {'PASS' if ok else 'FAIL'}")
    return ok

def _tiny_weights() -> dict:
    """Minimal weights dict:"""
    return {
        "language_model.embed_tokens.weight": torch.zeros(155648, 4096),
        "language_model.norm.weight": torch.zeros(4096),
        "language_model.layers.0.self_attn.q_norm.weight": torch.zeros(128),
        "language_model.layers.0.self_attn.k_norm.weight": torch.zeros(128),
        "language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4096, 4096),
        "language_model.layers.0.self_attn.k_proj.weight": torch.zeros(1024, 4096),
        "language_model.layers.0.self_attn.v_proj.weight": torch.zeros(1024, 4096),
        "language_model.layers.0.self_attn.o_proj.weight": torch.zeros(4096, 4096),
        "language_model.layers.0.input_layernorm.weight": torch.zeros(4096),
        "language_model.layers.0.post_attention_layernorm.weight": torch.zeros(4096),
        "language_model.layers.0.mlp.gate_proj.weight": torch.zeros(12288, 4096),
        "language_model.layers.0.mlp.up_proj.weight": torch.zeros(12288, 4096),
        "language_model.layers.0.mlp.down_proj.weight": torch.zeros(4096, 12288),
        "lm_heads.0.weight": torch.zeros(155648, 4096),
        **{f"lm_heads.{i + 1}.weight": torch.zeros(1025, 4096) for i in range(32)},
        **{f"emb_ext.{i}.weight": torch.zeros(1025, 4096) for i in range(32)},
    }

def _fresh(max_seq_len: int):
    """Standalone w1p export -> (model, fast) at an explicit cache size."""
    if os.path.isdir(STANDALONE):
        from moss_tts_lite.export import load_standalone_model
        return load_standalone_model(STANDALONE, device="cuda",
                                     max_seq_len=max_seq_len)
    from moss_tts_lite.st_loader import read_safetensors
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=max_seq_len)
    del weights
    fast = load_gptq_fast(model, STATE)
    torch.cuda.empty_cache()
    return model, fast

def phase_1_peaks() -> bool:
    """Measured peaks for typical inputs at the on-demand cache size."""
    print("=== Phase 1: measured peaks (on-demand KV, w1p fast path) ===")
    _start_ctx_gib()
    os.makedirs(OUT, exist_ok=True)
    ok = True
    worst = 0.0
    standalone = os.path.isdir(STANDALONE)
    ceiling = _PEAK_GIB if standalone else _PEAK_BASE_GIB
    print(f"  assembly path: {'standalone export' if standalone else 'base + w1p.pt'}; "
          f"ceiling {ceiling} GiB", flush=True)
    for name, text, language, budget in CASES:
        prompt = build_tts_prompt(text, language=language)
        l0 = int(prompt["input_ids"].shape[1])
        want = _resolve_max_seq_len(None, l0, budget)
        model, fast = _fresh(want)
        try:
            torch.cuda.reset_peak_memory_stats()
            res = generate_fast(fast, prompt, max_new_tokens=budget, seed=SEED)
            peak = torch.cuda.max_memory_allocated() / GIB
            kv = _kv_mib(int(model.max_seq_len)) / 1024
            good = peak <= ceiling and res.n_steps > 0
            ok &= good
            worst = max(worst, peak)
            print(f"  [{name:8s}] L0={l0:4d} max_seq_len={want} ({kv:.3f} GiB KV) "
                  f"steps={res.n_steps} peak={peak:.3f} GiB (ceiling {ceiling}) "
                  f"{'ok' if good else 'FAIL'}")
        finally:
            del fast, model
            gc.collect()
            torch.cuda.empty_cache()

    model, fast = _fresh(DEFAULT_MAX_SEQ_LEN)
    prompt = build_tts_prompt(CASES[0][1])
    torch.cuda.reset_peak_memory_stats()
    generate_fast(fast, prompt, max_new_tokens=MAXNEW, seed=SEED)
    peak_8192 = torch.cuda.max_memory_allocated() / GIB
    del fast, model
    gc.collect()
    torch.cuda.empty_cache()
    saved = peak_8192 - worst

    kv_saved_mib = _kv_mib(8192) - _kv_mib(_resolve_max_seq_len(None, 66, MAXNEW))
    good = saved > 0.4 and abs(kv_saved_mib - 558) < 2
    ok &= good
    print(f"  fixed-8192 peak={peak_8192:.3f} GiB; on-demand saves {saved:.3f} GiB "
          f"({kv_saved_mib:.0f} MiB of KV) {'ok' if good else 'FAIL'}")
    print(f"  phase 1 gate: {'PASS' if ok else 'FAIL'} (worst {worst:.3f} GiB)")
    return ok

def _phase2_one(case: str) -> dict:
    """One 8 GB case, in its own process (see `_phase2_child` for why)."""
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    ctx = _start_ctx_gib()
    cap = EIGHT_GB_CARD - ctx
    torch.cuda.set_per_process_memory_fraction(cap / total)
    name, text, language, _ = next(c for c in CASES if c[0] == case)
    prompt = build_tts_prompt(text, language=language)
    l0 = int(prompt["input_ids"].shape[1])
    want = _resolve_max_seq_len(None, l0, EIGHT_GB_BUDGET)
    rec = dict(case=case, l0=l0, ml=want, ctx=ctx, cap=cap, ok=False)
    try:
        model, fast = _fresh(want)
        torch.cuda.reset_peak_memory_stats()
        res = generate_fast(fast, prompt, max_new_tokens=EIGHT_GB_BUDGET, seed=SEED)
        rec["steps"] = res.n_steps
        rec["peak_alloc"] = torch.cuda.max_memory_allocated() / GIB
        rec["ok"] = res.n_steps > 0
    except torch.cuda.OutOfMemoryError:
        rec["err"] = "OOM"
    return rec

def _phase2_child() -> bool:
    """Run phase 2 in fresh processes and return the verdict."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = True
    ctx = None
    print(f"=== Phase 2: 8 GB card simulation ===")
    for case, _, _, _ in CASES:
        proc = subprocess.run([sys.executable, "-m", "tests.test_release"],
                              cwd=root, env=dict(os.environ, KVFIT_PHASE2=case),
                              capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("KVFITREC ")]
        if not line:
            print(f"  [{case:8s}] child failed:\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}")
            ok = False
            continue
        import json
        rec = json.loads(line[0][len("KVFITREC "):])
        ctx = rec["ctx"]
        smi = rec["smi_peak"]
        good = rec["ok"] and smi <= EIGHT_GB_CARD
        ok &= good
        print(f"  [{case:8s}] L0={rec['l0']:4d} max_seq_len={rec['ml']:5d} "
              f"steps={rec.get('steps', 0):4d} torch peak={rec.get('peak_alloc', 0):.3f} "
              f"smi peak={smi:.3f} (card {EIGHT_GB_CARD}, spare "
              f"{EIGHT_GB_CARD - smi:+.3f}) {'ok' if good else 'FAIL'}"
              + (f" [{rec.get('err', '')}]" if rec.get("err") else ""), flush=True)
    print(f"  CUDA context {ctx if ctx else float('nan'):.3f} GiB; torch capped at "
          f"{EIGHT_GB_CARD - (ctx or 0):.3f} GiB; pass criterion: "
          f"nvidia-smi peak <= {EIGHT_GB_CARD} GiB")

    for label, env in (("codec", "KVFIT_C1=c"), ("boundary", "KVFIT_C1=b"),
                       ("advice", "KVFIT_C1=a")):
        proc = subprocess.run([sys.executable, "-m", "tests.test_release"],
                              cwd=root, env=dict(os.environ, KVFIT_PHASE2="", **{env.split("=")[0]: env.split("=")[1]}),
                              capture_output=True, text=True)
        out = [l for l in proc.stdout.splitlines() if l.startswith("KVFITC1 ")]
        good = proc.returncode == 0 and bool(out) and out[0].endswith("PASS")
        ok &= good
        print("  " + (out[0][len("KVFITC1 "):] if out else f"{label} child failed"))
    print(f"  phase 2 gate: {'PASS' if ok else 'FAIL'}")
    return ok

def _phase2_rest(which: str) -> bool:
    """Codec / boundary-control / advice checks, each in its own capped process."""
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    ctx = _start_ctx_gib()
    cap = EIGHT_GB_CARD - ctx
    torch.cuda.set_per_process_memory_fraction(cap / total)
    ok = True
    if which == "c":
        from moss_tts_lite.codec import MossCodecDecoder
        codec_dir = os.environ.get("MOSS_AUDIO_MODEL_DIR") or os.path.join(
            ROOT, "models", "MOSS-Audio-Tokenizer")
        torch.cuda.reset_peak_memory_stats()
        codec = MossCodecDecoder(codec_dir, device=torch.device("cuda"))
        peak = torch.cuda.max_memory_allocated() / GIB
        del codec
        ok = peak <= _PEAK_CODEC_GIB and peak <= cap
        print(f"KVFITC1 codec alone: peak={peak:.3f} GiB (ceiling {_PEAK_CODEC_GIB}, "
              f"cap {cap:.3f}) -> {'PASS' if ok else 'FAIL'}")
    elif which == "b":

        over = 4 * EIGHT_GB_BUDGET
        prompt = build_tts_prompt(CASES[0][1])
        l0 = int(prompt["input_ids"].shape[1])
        try:
            model, fast = _fresh(_resolve_max_seq_len(None, l0, over))
            try:
                generate_fast(fast, prompt, max_new_tokens=over, seed=SEED)
            finally:
                del fast, model
        except torch.cuda.OutOfMemoryError:
            ok = True
            print(f"KVFITC1 boundary control: --max-new-tokens {over} OOMs as "
                  f"measured -> PASS")
        else:
            ok = False
            print(f"KVFITC1 boundary control: --max-new-tokens {over} unexpectedly "
                  f"FIT (cap not binding; boundary numbers stale) -> FAIL")
    else:
        from moss_tts_lite.cli import _kv_advice, _oom_advice, synthesize
        msg_ok = False
        try:
            synthesize("你好。", os.path.join(OUT, "never.wav"), max_new_tokens=60000,
                       seed=SEED, fast=True, w4_group_size=32, gptq_state=STATE,
                       model_dir=MODEL_DIR)
        except (torch.cuda.OutOfMemoryError, ValueError) as exc:
            msg_ok = "max-new-tokens" in str(exc) and "segment" in str(exc)
        kv_msg = str(_kv_advice(ValueError("prompt 66 + max_new_tokens 4096 "
                                           "exceeds KV cache 512"), 66, 4096))
        oom_msg = str(_oom_advice(torch.cuda.OutOfMemoryError("CUDA out of memory."),
                                  "你好。", 60000))
        passthrough = str(_kv_advice(
            ValueError("w4_group_size must be 32/64/128/256"), 1, 1)) \
            == "w4_group_size must be 32/64/128/256"
        both = all("--max-new-tokens" in m and "segment" in m
                   for m in (kv_msg, oom_msg))
        ok = msg_ok and both and passthrough
        print(f"KVFITC1 advice: over-budget raised with advice={msg_ok}, "
              f"builders name both remedies={both}, unrelated ValueError "
              f"passes through={passthrough} -> {'PASS' if ok else 'FAIL'}")
    return ok

def main_vram_budget() -> int:
    if os.environ.get("KVFIT_PHASE2"):
        import json
        rec = _phase2_one(os.environ["KVFIT_PHASE2"])
        rec["smi_peak"] = _smi_mib() / 1024
        print("KVFITREC " + json.dumps(rec))
        return 0 if rec["ok"] else 1
    if os.environ.get("KVFIT_C1"):
        return 0 if _phase2_rest(os.environ["KVFIT_C1"]) else 1
    ok0 = phase_0_sizing()
    if not torch.cuda.is_available():
        print("test_vram_budget: no CUDA -- phase 0 only")
        return 0 if ok0 else 1
    if not os.path.exists(STATE):
        print(f"SKIP phase 1/2: GPTQ state {STATE} not present")
        return 0 if ok0 else 1
    ok1 = phase_1_peaks()
    ok2 = _phase2_child()
    ok = ok0 and ok1 and ok2
    print(f"test_vram_budget: sizing={'PASS' if ok0 else 'FAIL'} "
          f"peaks={'PASS' if ok1 else 'FAIL'} "
          f"8gb={('PASS' if ok2 else 'FAIL')} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def main() -> int:
    rc = 0

    if os.environ.get("KVFIT_PHASE2") or os.environ.get("KVFIT_C1"):
        return main_vram_budget()
    rc |= main_dep_purity() or 0
    rc |= main_export() or 0
    rc |= main_vram_budget() or 0

    return rc

if __name__ == "__main__":
    sys.exit(main())
