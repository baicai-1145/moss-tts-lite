# moss-tts-lite 训练指南（SFT / LoRA / QLoRA）

在保持推理核心零新增硬依赖（torch/numpy/soundfile/pyyaml）的前提下，为
MOSS-TTS-v1.5 补全训练能力：自研可训练模型（`moss_tts_lite/nn.py`）、
与官方 processor **逐 token 对齐**的数据管道（`moss_tts_lite/data.py`）、
单卡训练 CLI（`moss_tts_lite/train.py`）与官方 CAT 编码 CLI
（`moss_tts_lite/prepare_data.py`）。

训练走可选依赖组，推理核心不受影响：

```bash
pip install '.[train]'          # transformers + peft + bitsandbytes(Linux) + safetensors
```

| 模块 | 作用 | 依赖 |
|---|---|---|
| `moss_tts_lite/nn.py` | `TrainableMossTTS`：参数名与 safetensors 键严格一致，SDPA/左 padding/grad checkpointing，loss 与官方 `modeling_moss_tts.py` 同式 | torch |
| `moss_tts_lite/data.py` | JSONL → computing_loss 序列 + loss_mask + collate（对齐官方 `dataset.py`/`processing_moss_tts.py`） | torch |
| `moss_tts_lite/train.py` | `python -m moss_tts_lite.train`：full / lora / qlora 训练 + merge 导出 | torch（peft/bnb 懒加载） |
| `moss_tts_lite/prepare_data.py` | `python -m moss_tts_lite.prepare_data`：官方 MOSS-Audio-Tokenizer 编码 audio → audio_codes | transformers（懒加载） |

## 1. 数据格式

**原始 JSONL**（`audio` 必填，其余可选）：

```json
{"audio": "wavs/nahida_001.wav", "text": "初次见面，我是纳西妲。", "language": "Chinese",
 "ref_audio": "wavs/ref.wav",
 "instruction": null, "tokens": null, "quality": null, "sound_event": null, "ambient_sound": null}
```

多参考音频用 `reference`（列表，可含 null 占位，S1/S2...）或 `reference_audio`。

**编码后 JSONL**（prepare_data 产物，官方兼容）：

```json
{"audio": "...", "text": "...", "language": "Chinese",
 "audio_codes": [[c0..c31], [c0..c31], ...],
 "ref_audio_codes": [[...]]}
```

`audio_codes` 是 `[T, n_vq=32]` 的嵌套 int list（24kHz 下 1 帧 ≈ 1/12 秒…实为
CAT 帧率，25 秒 ≈ 3000 帧，超出可用 `--max-audio-frames` 过滤）。

## 2. 编码：prepare_data

用官方 MOSS-Audio-Tokenizer（CAT）把 wav 编成 codes。音频读取用 soundfile；
重采样是 torchaudio 默认 sinc 重采样的 numpy 移植（装了 torchaudio 则直接用官方实现）；
响度归一化（-20 dBFS，增益 ±3dB）与官方 processor 一致。

```bash
python -m moss_tts_lite.prepare_data \
    --codec-dir /path/to/MOSS-Audio-Tokenizer \
    --input-jsonl data/nahida_raw.jsonl \
    --output-jsonl data/nahida_train.jsonl \
    --device cuda --batch-size 8
```

- `--n-vq` 默认 None（用 codec 默认 32）。
- `--skip-reference-audio-codes` 跳过 ref 编码（只用 text+audio 也能训）。
- CAT 需要从 HF 下载（首次运行自动拉取 `OpenMOSS-Team/MOSS-Audio-Tokenizer`，
  或传本地目录）。

## 3. 训练：train

```bash
# QLoRA（推荐，24GB 单卡）
python -m moss_tts_lite.train \
    --model-dir /path/to/MOSS-TTS-v1.5 \
    --train-jsonl data/nahida_train.jsonl \
    --output-dir runs/nahida-qlora \
    --mode qlora --bf16 \
    --lora-rank 32 --lora-alpha 64 \
    --per-device-batch-size 8 --max-batch-tokens 1200 \
    --num-workers 4 --fused-optimizer \
    --learning-rate 1e-4 --num-epochs 3 \
    --channelwise-loss-weight 1,32 \
    --max-audio-frames 3000 --seed 42 \
    --merge-and-export
```

- `--mode full`：全参 SFT（默认 lr 1e-5；显存 ≥ 2×模型 bf16 + 优化器态，4090 24G 只够小改）。
- `--mode lora`：纯 peft LoRA 不量化（CPU 也能跑，适合冒烟）。
- `--mode qlora`：NF4 4-bit + LoRA（bitsandbytes，需 CUDA/Linux；lr 默认 1e-4）。
- `--channelwise-loss-weight 1,32`：text 头权重 1、32 个 audio 头共 32（每头 1），
  也可给 33 个逗号值。
- `--merge-and-export`：训练完直接 merge 并导出到 `<output-dir>/merged/`。
- 日志行含 step/loss/lr/steps_per_sec/eta/peak_mem。
- 结束保存：lora/qlora → `adapter_model.safetensors` + `adapter_config.json` +
  `train_args.json`；full → 原键名分片 safetensors（>4GB 自动分片 + index）+ tokenizer/config 拷贝。

**QLoRA 说明**：语言主干线性层量化为 NF4（bnb_4bit_use_double_quant），
embeddings/lm_heads 保持 bf16（与 transformers QLoRA 默认跳过 lm_head 的布局一致）。
模型先以 bf16 加载到 CPU，再逐层量化上卡（24GB 卡友好）。

### 3.1 吞吐调优（变长音频数据必读）

语音样本长度差异大（本数据集 p50≈180 / p95≈264 / max≈457 token）。**固定
batch 随机组批**会造成：padding 计算浪费 + 最坏 batch 显存失控（24GB 卡上
bs4 随机组批会 OOM）。三个正交手段：

1. **--max-batch-tokens（token-budget batching）**：按长度分桶组批，
   保证 padded max_len × bs ≤ 预算——短句自动满批、长句自动小批，
   消灭 padding 浪费与 OOM。--per-device-batch-size 作为批大小上限。
2. **--gradient-checkpointing**：激活显存大幅下降，换来反向重算 ~30% 开销，
   但允许把 token 预算开得更大（gc+mbt2400 峰值实测仅 8.9 GB）。
3. --num-workers 4 --fused-optimizer：免费小补。

4090-24G 实测（纳西妲 1705 条 / 3.04h，40 步取后 30 步均值）：

| 配置 | samples/s | 相对基线 | 峰值显存 | SM | 功耗 |
|---|---|---|---|---|---|
| bs1×ga4（旧默认） | 3.0 | 1.0× | 13.0 GB | ~42% | ~92 W |
| bs4 + mbt6000 | 10.2 | 3.4× | 20.5 GB | ~38% | ~76 W |
| bs8 + mbt1200 | 13.0 | 4.3× | 21.9 GB | ~45% | ~150 W |
| gc + bs8 + mbt2400 | 待独占卡复测 | — | 8.9 GB | — | — |

推荐起步：--per-device-batch-size 8 --max-batch-tokens 1200 --num-workers 4
--fused-optimizer；显存富余优先加预算、再加 bs 上限，紧张则加 gc。
瓶颈是 per-step Python/launch 开销（peft 包装层 ~400 次调用/步），
batch 越大摊得越薄。

### 3.2 MOSS-TTS-Local-Transformer-v1.5(48kHz v2 codec)

同一套 CLI 直接支持 HF 版 OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
(config.json 的 model_type == "moss_tts_local"，自动切换模型类与数据布局):

```bash
python -m moss_tts_lite.prepare_data \
    --codec-dir /path/to/MOSS-Audio-Tokenizer-v2 \
    --input-jsonl data/nahida_raw.jsonl \
    --output-jsonl data/nahida_train_v2.jsonl \
    --device cuda --batch-size 4        # 自动 48kHz 立体声、n_vq=12

python -m moss_tts_lite.train \
    --model-dir /path/to/MOSS-TTS-Local-Transformer-v1.5 \
    --train-jsonl data/nahida_train_v2.jsonl \
    --output-dir runs/nahida-v2-qlora \
    --mode qlora --bf16 --per-device-batch-size 4 \
    --max-batch-tokens 900 --num-workers 4 --fused-optimizer \
    --num-epochs 3 --seed 42
```

与 v1(ModelScope 版)的关键差异(均已自动处理):

| | v1(delay) | v2(local-transformer) |
|---|---|---|
| 采样率/声道 | 24kHz 单声道 | **48kHz 立体声** |
| n_vq | 32 | **12** |
| audio_start/end id | 151652/151653 | **151669/151670**(从 tokenizer 解析) |
| slot token | 专用字符串 | **复用 vision_pad/video_pad 的 id**(按 config 注入) |
| 音频帧展开 | delay 模式,每帧 n_vq 个位置 | **每帧 1 个位置**(序列短 ~8x) |
| loss | 逐样本逐通道归一 | batch 级加权均值 + binary local text head |

v2 序列短(1705 条数据 p50=149 / p95=233 / max=426 token),token 预算可以
开得比 v1 更激进。共享 4090(~14.7GB 可用)实测:

| 配置 | samples/s | 峰值显存 |
|---|---|---|
| bs4 + mbt900 (rank32) | 11.2 | 12.6 GB |
| bs8 + mbt900 (rank64) | ~16 | 13.3 GB |
| bs8 + mbt1200 (rank64) | OOM(差 1.7GB) | - |
| bs8/bs16 + mbt2400 独占卡 | OOM | >24GB |
| bs16 + mbt1200 独占卡 | 5.1 | 15.9 GB |

结论: v2 短序列(p50=149)下 **bs8+mbt900 是甜点**; 更大 token 预算反被
激活/带宽压力反噬(mbt1200 独占反而慢 3x)。
注: torch.compile(dynamic=True) 在 NF4+peft 下实测负优化(1.9 vs 2.8 steps/s,
bnb4bit graph break + 动态 shape 重编译),保持默认关闭。merge 子命令同样
自动识别 v2 底座。

## 4. 合并导出：merge（独立入口）

```bash
python -m moss_tts_lite.train merge \
    --adapter-dir runs/nahida-qlora \
    --model-dir /path/to/MOSS-TTS-v1.5 \
    --output-dir models/nahida-merged \
    --device cuda --bf16
```

- 输出目录 = 原版格式（`model.safetensors` + tokenizer/config 文件），
  **可直接给 moss-tts-lite CLI `--model-dir` 用**：
  ```bash
  moss-tts-lite "风雨踩着云朵来了。" -o nahida.wav \
      --model-dir models/nahida-merged --device cuda
  ```
- qlora adapter 默认先量化基座再 merge（`--qlora-exact-base`，与训练时前向一致，
  需 CUDA+bnb；不可用时自动回退 bf16 基座 merge 并打日志）。
- `--fp32` 可导出 fp32（默认 bf16）。

## 5. 纳西妲 QLoRA 实战（本次 4090 实例）

```bash
# 1) 编码
python -m moss_tts_lite.prepare_data \
    --codec-dir ~/models/MOSS-Audio-Tokenizer \
    --input-jsonl ~/data/nahida/raw.jsonl \
    --output-jsonl ~/data/nahida/train.jsonl \
    --device cuda --batch-size 8

# 2) QLoRA 训练（~25s 样本，3000 帧上限）
python -m moss_tts_lite.train \
    --model-dir ~/models/MOSS-TTS-v1.5 \
    --train-jsonl ~/data/nahida/train.jsonl \
    --output-dir ~/runs/nahida-qlora \
    --mode qlora --bf16 --gradient-checkpointing \
    --lora-rank 32 --lora-alpha 64 \
    --per-device-batch-size 1 --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 --num-epochs 3 --warmup-ratio 0.03 \
    --channelwise-loss-weight 1,32 --max-audio-frames 3000 \
    --seed 42 --merge-and-export

# 3) 合成（merged 导出目录直接喂给推理 CLI）
moss-tts-lite "智慧之城的知识，终究要靠自己书写。" -o nahida.wav \
    --model-dir ~/runs/nahida-qlora/merged
```

## 6. 显存建议（参考）

| 模式 | 配置 | 显存（24GB 卡） |
|---|---|---|
| qlora r=32 | bs1 + grad ckpt + bf16 + NF4 | ≈ 8–12 GB（T≤3000 帧样本） |
| lora r=32 | bf16 基座 + grad ckpt | ≈ 10–16 GB |
| full | bf16 + AdamW | ≥ 24 GB（1.6B 全参，优化器态 fp32 为主）超限请降 lr/步数或用 qlora |

序列长度 = prompt(≈100 token) + 音频帧 × 1 + delay 斜坡 32 + 模板；
`--max-audio-frames 3000`（≈25 s）是 24GB 单样本安全线。

## 7. 与官方实现的对齐验证

- **序列构造**：`tests/test_train_data.py::test_official_processor_parity`
  用真实 MOSS tokenizer（HF AutoTokenizer）+ 官方
  `MossTTSDelayProcessor(mode="computing_loss"/"generation")` 逐 token 比对
  `input_ids`（含 ref audio codes 嵌入 user 块的用例）与 `_pad` 布局；
  无 transformers/MOSS-TTS checkout 时自动 skip。
- **模型数值**：`tests/test_train_nn.py` 用小权重（真实键名）同时喂
  `MossTTSModel`（推理）与 `TrainableMossTTS`，fp32 logits allclose、
  bf16 容差内一致；loss/backward/梯度存在；左 padding 因果性。
- **训练循环**：`tests/test_train_loop.py` CPU 跑 full/lora 各 2 步、
  adapter 保存/加载/merge round-trip、CLI --help。
- 官方 loss 公式（逐头 CE → channel 归一 → 加权平均）、loss_mask
  （prompt_len-1 起）、labels 掩码（非 loss 区/padding/audio_pad_code=1024→-100）
  均按 `modeling_moss_tts.py` / `finetuning/dataset.py` 移植。

## 8. 常见问题

- **qlora 报 ImportError**：bitsandbytes 仅 Linux；Windows 本地用 `--mode lora` 冒烟，
  正式 qlora 放 Linux/远程卡。
- **想要官方 HF 格式的 merged 模型**：merge 产出即 HF 兼容 safetensors
  （键名同原版），也可直接被 transformers 加载（需官方 modeling 文件）。
- **多卡/DDP/W&B**：YAGNI，不做；单卡循环足够（与设计文档一致）。
