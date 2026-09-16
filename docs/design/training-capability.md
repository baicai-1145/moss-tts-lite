# moss-tts-lite 训练能力补全设计（SFT + QLoRA）

日期：2026-09-15　状态：已确认（用户批准 lite 自研栈 / 纯 text-speech pair / 训练+合成闭环验证）

## 目标

为 moss-tts-lite 补全 SFT 与 QLoRA 训练能力，保持极简哲学（推理核心仍 4 依赖；训练走可选依赖组）。
验收：远程 4090（单卡 GPU6 24GB）上用纳西妲数据集跑通 QLoRA（loss 下降、adapter 保存、merge 后 lite CLI 合成）。

## 架构决策

1. **自研训练栈**：可训练 nn.Module（权重键与 safetensors 一致）+ 轻量单卡训练循环；QLoRA 用 peft+bitsandbytes（可选依赖）。
2. **数据编码用官方 CAT encoder**（transformers trust_remote_code 加载 MOSS-Audio-Tokenizer）——lite 只带 decoder；编码是一次性离线步骤。
3. **序列构造复用 lite tokenizer**（QwenBPE + prompt.py 模板），补 computing_loss 模式，与官方 processor 逐 token 对齐。
4. **数据格式**：`{"audio": path, "text": str, "language": "Chinese"}`（可选 ref_audio）；单说话人音色从数据学习。

## 模块设计

| 文件 | 职责 |
|---|---|
| `moss_tts_lite/nn.py` | `TrainableMossTTS(nn.Module)`：Qwen3 风格 backbone + emb_ext(32) + lm_heads(33)；参数名=权重键；forward(input_ids[B,T,33], attention_mask, labels, channelwise_loss_weight)→(text_logits, audio_logits, loss)；SDPA、左 padding、grad checkpointing |
| `moss_tts_lite/data.py` | JSONL Dataset（audio_codes/ref_audio_codes/text/language）→ computing_loss 序列 + loss_mask；collate（左 pad、labels=-100 掩码：prompt 外/pad/audio_pad_code）|
| `moss_tts_lite/train.py` | CLI：`--mode {full,lora,qlora}`、LoRA(r/alpha/targets)、AdamW+cosine+warmup+clip、bf16、日志、checkpoint、`--merge-and-export`（merge 后按原版权重目录格式导出，供 lite CLI 推理）|
| `moss_tts_lite/prepare_data.py` | 官方 CAT encoder 编码 audio→audio_codes、ref_audio→ref_audio_codes，产出官方兼容 JSONL |
| `tests/test_train_*.py` | CPU 小权重：nn forward 与推理模型 logits 一致；序列构造与官方 processor 对齐；训练循环 2 步 smoke |
| `pyproject.toml` | `[train]` 可选依赖组：transformers/peft/bitsandbytes/safetensors |

## 数据流

raw 7z → 解压 → train_raw.jsonl → prepare_data（CAT encode）→ train.jsonl → train.py --mode qlora → adapter → merge → lite CLI 合成。

## 验证计划

1. 本地 CPU：pytest 全绿（新测试 + 旧测试不回归）；小权重 logits 对齐。
2. 远程 GPU6：prepare 纳西妲 → QLoRA 数十步 loss 下降 → adapter 保存 → merge → lite CLI 合成纳西妲语音。

## 分工

- subagent A：上述代码模块 + 测试。
- subagent B：7z 上传解压、数据集整理 train_raw.jsonl、远程 venv（torch/peft/bnb/transformers）。
- lead：权重下载（已启动）、最终 QLoRA 验证与合成闭环。
