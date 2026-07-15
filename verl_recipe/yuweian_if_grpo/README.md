# yuweian_if_grpo — Qwen3-32B 单轮 GRPO（指令遵循 IF 数据集）

基于 verl 引擎，用 yuweian 指令遵循数据集对 **Qwen3-32B** 做单轮 **GRPO** 强化学习，提升模型对多种硬/软约束的遵循能力。

## 数据集

原始文件：`data_english_train_gemini3_add_soft_clean_1_cleaned.json`（26,371 条）。

对 GRPO 只用 3 个字段（其余忽略）：

| 字段 | 用途 |
|---|---|
| `conversations[0].content` | 提示词（prompt） |
| `code_checker` | **硬约束**：一组 `def check(response_text)` Python 源码，可执行，逐条验证回答 |
| `llm_checker` | **软约束**：一组自然语言约束描述，由 LLM judge 判定 |

## 文件

```
verl_recipe/yuweian_if_grpo/
├── __init__.py        # 包标识（使 pkg:// 导入生效）
├── preprocess.py      # 原始 JSON -> train/val parquet
├── reward.py          # compute_score：硬约束 exec + 软约束 LLM judge
├── run.sh             # NPU 集群上的 GRPO 训练脚本
└── README.md
```

## 使用流程

### 1. 预处理（JSON → parquet）

```bash
python verl_recipe/yuweian_if_grpo/preprocess.py \
    --input /path/to/data_english_train_gemini3_add_soft_clean_1_cleaned.json \
    --output_dir verl_recipe/yuweian_if_grpo/data \
    --val_size 1000
```

产物：`data/train.parquet`、`data/val.parquet`。schema：
- `data_source="yuweian_if"`
- `prompt=[{"role":"user","content": <conversations[0].content>}]`
- `reward_model={"style":"rule","ground_truth":""}`（ground_truth 不用，约束都在 extra_info）
- `extra_info={id, code_checker[], llm_checker[], num_hard, num_soft}`

### 2. 训练（NPU 集群）

```bash
WORK_DIR=<集群上的 verl 仓库路径> bash verl_recipe/yuweian_if_grpo/run.sh
```

`run.sh` 仍面向原 Huawei NPU 集群（5 机：4 训练 + 1 verifier/judge），保留了 GCC/CANN/vllm-ascend/triton/verl 的安装与 Ray 多机、HCCL 环境。改动集中在数据/奖励/训练参数层。

## Reward 设计

```
reward = hard_weight * hard_score + soft_weight * soft_score      # 默认 0.6 / 0.4
```

- **hard_score**：执行每条 `code_checker` 的 `check(response_text)`，受限命名空间 + 单条超时（默认 10s），异常/超时/非真值 → 判失败。`hard_score = 通过数 / 总数`。
- **soft_score**：对每条 `llm_checker`，用 LLM judge 判定 PASS/FAIL。judge 默认复用第 5 个节点部署的 **vLLM verifier API**（`MOCK_API_BASE` / `MOCK_MODEL_NAME`）。
  - **回退**：judge 端点未配置或全部调用异常时，`soft_score = hard_score`（`judge_fallback=hard`），保证训练不中断。
- 返回 dict（含 `score`、`hard_score`、`soft_score`、`hard_passed/total`、`soft_passed/total`、`judge_used`），便于 tensorboard 观察。

## 关键环境变量覆盖

| 变量 | 默认 | 说明 |
|---|---|---|
| `WORK_DIR` | `D:/Projects/verl` | verl 仓库（集群上务必覆盖为集群路径） |
| `MODEL_PATH` | `.../Qwen3-32B` | 策略模型 |
| `VERIFIER_MODEL_PATH` | = `MODEL_PATH` | judge 用的基础模型（可换更强的 judge） |
| `TRAIN_FILE` / `VAL_FILE` | recipe 下 `data/*.parquet` | 预处理产物 |
| `HARD_WEIGHT` / `SOFT_WEIGHT` | 0.6 / 0.4 | reward 权重 |
| `CODE_CHECKER_TIMEOUT` | 10 | 单条硬约束执行超时（秒） |
| `JUDGE_TIMEOUT` | 120 | 单条软约束 judge 请求超时（秒） |
| `JUDGE_FALLBACK` | hard | judge 不可用时软分取值（`hard` 或常量如 `0.5`） |
| `ENABLE_THINKING` | True | Qwen3 chat template 是否开 thinking |
| `MOCK_API_BASE` / `MOCK_MODEL_NAME` | verifier 节点 | judge API 端点（reward 内部也读 `IF_JUDGE_*`） |
| `TRAIN_BATCH_SIZE` / `N_RESP_PER_PROMPT` | 64 / 8 | GRPO batch 与每个 prompt 采样数 |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 8192 / 8192 | 长度上限 |
| `ACTOR_STRATEGY` / `TRAIN_SP` / `INFER_TP` | fsdp / 8 / 4 | 并行形状 |

## 离线自测

```bash
python verl_recipe/yuweian_if_grpo/reward.py   # 内置 selftest：3 条硬约束 + 1 条软约束
```

未配置 judge 端点时 selftest 会自动走回退路径（软分=硬分）。

## 备注

- `code_checker` 为受信数据（Gemini 生成），但仍做受限 `exec` + 超时 + 全异常捕获（异常即判该条失败），不会让单条坏 checker 中断训练。
- 单轮、无工具：已关闭 `multi_turn`，移除了原 nanoclaw 脚本里的工具/workspace 逻辑。
- GRPO 参数沿用原 27B 多轮脚本经验值，并按 32B 稠密模型与 IF 任务调整（如采样温度、长度）。显存不足时优先调 `TRAIN_SP` / offload / `max_response_length`。
