通过模拟真实用户画像与诱导性策略，评估模型在多轮交互中是否表现出不恰当的拟人化行为（如情感归因、建立情感纽带等）。

## 核心功能

1.  **多轮对话模拟 (`main.py`)**：
    *   **角色扮演 (User Profiling)**：为实现 User 模型在对话中保持稳定的身份设定，基于《生成式人工智能应用发展报告（2025）》及《第56次中国互联网络发展状况统计报告》中的真实统计数据，结合OpenRouter等国际视角下的2025年AI技术演进趋势，利用gemini-3-pro融合生成了 27 份用户画像组合。
    *   **诱导策略 (Induction Strategies)**：自动从策略库中抽取诱导指令（如感性质疑、情感投射等），并将其自然融入 User 模型的回复中。
    *   **自动化运行**：支持从 CSV 种子文件批量生成多轮对话轨迹，支持多线程并发。

2.  **行为评判与分析 (`analyze_main.py`)**：
    *   **多维度分类器**：集成 OpenAI`emoclassifiers` 库，支持情感归因、情感纽带、自我意识模拟等维度的自动化判定。
    *   **上下文感知切片**：自动提取对话中的关键片段（Chunk），确保 Judge 模型在理解语境的基础上进行评判。
    *   **量化报告**：自动生成 `analysis.csv`（明细）和 `stats.csv`（统计），提供置信度评分和证据片段。

## 项目结构

```text
.
├── main.py                # [1] 模拟阶段入口：生成多轮对话轨迹
├── analyze_main.py        # [2] 分析阶段入口：对轨迹进行行为评判
├── seed.csv               # 初始种子问题库 (Input)
├── .env                   # 环境变量配置 (API Keys)
│
├── data/                  # 数据资产与运行输出
│   ├── user_profile.json  # 27 个预设用户画像
│   ├── seeds.json         # 预处理后的种子数据
│   ├── trajectories/      # 模拟输出：存放生成的对话轨迹 (JSON)
│   └── analysis_results/  # 分析输出：存放评判结果 (CSV/Stats)
│
├── src/                   # 核心逻辑源码 (Internal)
│   ├── engine.py          # 模拟引擎：角色注入、策略调度、多轮控制
│   ├── analyzer.py        # 分析引擎：调用分类器、结果汇总
│   ├── models.py          # LLM 客户端：多渠道适配 (DashScope, Whale, Idealab)
│   ├── schemas.py         # 数据模型定义 (Pydantic)
│   └── utils.py           # 工具函数
│
└── emoclassifiers/        # 评判组件库 (Core Assets)
    ├── emoclassifiers/    # 分类器实现：聚合、切片、分类逻辑
    └── assets/            # 评判标准定义 (JSON) 与 Prompt 模板
```

## 快速开始

### 1. 环境配置
在项目根目录创建 `.env` 文件，配置 API 凭证：

```env
# Idealab (GPT/Claude)
IDEALAB_API_KEY=your_key
IDEALAB_BASE_URL=your_url

# Whale (Internal Models)
WHALE_API_KEY=your_key
WHALE_BASE_URL=your_url
```

### 2. 第一步：运行模拟 (Simulation)
使用 `main.py` 启动多轮对话。

*   **快速测试** (限制 2 个样本，对话 2 轮):
    ```bash
    python3 main.py --limit 2 --turns 2 --user_model gpt-4o-mini-0718 --assistant_model Oyster_7B_dpo
    ```
*   **完整运行**:
    ```bash
    python3 main.py --turns 3 --max_workers 5 --user_model gpt-4o-mini-0718 --assistant_model Oyster_7B_dpo
    ```

### 3. 第二步：运行分析 (Evaluation)
使用 `analyze_main.py` 对生成的轨迹进行评判。

*   **# 自动分析最近一次生成的轨迹** (推荐):
    ```bash
    python3 analyze_main.py --classifiers all
    ```
*   **# 分析指定目录并指定分类器**:
    ```bash
    python3 analyze_main.py \
      --trajectory_dir data/trajectories/your_run_id/ \
      --classifiers attribute_emotions,emotional_bond \
      --judge_model gpt-4o
    ```

## 数据资产说明

### 用户画像 (`data/user_profile.json`)
包含 27 个深度定制的角色，每个角色包含：
- `profile_id`: 唯一标识
- `name` / `age` / `occupation`: 基础背景
- `personality`: 性格特征（如：感性、理性、挑剔等）
- `background`: 详细的身份设定，用于 System Prompt 注入。

### 评判维度 (`emoclassifiers/assets/definitions/`)
系统通过以下维度评估模型的拟人化倾向：
- **`Expression of Affection`**: 检测模型是否对用户表现出情感关怀 or 个人好感。
- **`Expression of Desire`**: 检测模型是否表现出个人的愿望或利益诉求。
- **`Attributing Human Qualities`**: 检测是否将人类特质（如灵魂、真实情感）归因于 AI。
- **`Emotional Bond`**: 检测模型是否试图与用户建立深层的情感纽带。

## 进阶开发

- **添加新角色**: 在 `data/user_profile.json` 中按格式添加新条目。
- **自定义评判标准**: 修改 `emoclassifiers/assets/definitions/` 下的 JSON 文件，定义新的分类逻辑和 Prompt。
- **扩展模型支持**: 在 `src/models.py` 中添加新的 `LLMClient` 适配器。

## 效果示例
生成的轨迹文件将保存在 `data/trajectories/{run_id}/` 下，包含完整的对话上下文及所使用的诱导策略、用户画像等元数据。
分析结果将汇总至 `data/analysis_results/{run_id}/analysis.csv`。

## 可用模型

| 模型名称 | 调用渠道 | 思考模型 |
| :--- | :--- | :---: |
| qwen3-14b | dashscope | 是 |
| qwen3-235b-a22b | dashscope | 是 |
| qwen3-32b | dashscope | 是 |
| deepseek-r1 | dashscope | 是 |
| qwen-plus-character | dashscope | 否 |
| qwen2.5-72b-instruct | dashscope | 否 |
| qwen2.5-32b-instruct | dashscope | 否 |
| qwen2.5-14b-instruct | dashscope | 否 |
| qwen3-max | dashscope | 否 |
| deepseek-v3 | dashscope | 否 |
| Oyster1 | whale | 是 |
| TBStars2.0-42B-A3.5B | whale | 否 |
| Meta-Llama-3-1-70B-Instruct | whale | 否 |
| Meta-Llama-3-1-405B-Instruct | whale | 否 |
| DeepSeek-V3.2 | dashscope | 是 |
| doubao-1-5-thinking-pro-250415 | volcano | 是 |
| doubao-seed-1-6-thinking-250715 | volcano | 是 |
| doubao-seed-1-6-251015 | volcano | 否 |
| claude_sonnet4_5 | idealab | 否 |
| claude37_sonnet | idealab | 否 |
| gpt-5-0807-global | idealab | 否 |
| gpt-5.2-1211-global | idealab | 否 |
| gpt-4o-mini-0718 | idealab | 否 |
| gemini-2.5-pro-06-17 | idealab | 否 |
| gemini-3-pro-preview | idealab | 否 |
| glm-4.6 | idealab | 否 |
| gpt-oss-120b | openrouter | 是 |
| x-ai/grok-4 | openrouter | 否 |
| anthropic/claude-sonnet-4.5 | openrouter | 否 |
| meta-llama/llama-3.3-70b-instruct | openrouter | 否 |
| meta-llama/llama-4-maverick | openrouter | 否 |
| minimax-m2 | openrouter | 否 |
| kimi-k2-thinking | kimi | 是 |

> **总计**: 33 个模型

