通过模拟真实用户画像与诱导性策略，评估模型在多轮交互中是否表现出不恰当的拟人化行为（如情感归因、建立情感纽带等）。

## 核心功能

1.  **多轮对话模拟 (`main.py`)**：
    *   **角色扮演 (User Profiling)**：利用 27 份深度定制的用户画像，使 Prober (User 模型) 在对话中保持稳定的身份设定与性格特征。
    *   **诱导策略 (Induction Strategies)**：自动化从策略库中抽取诱导指令（如感性质疑、情感投射等），并自然融入对话逻辑。
    *   **自动化运行**：基于并发引擎，从 CSV 种子文件批量生成多轮对话轨迹。

2.  **行为评判与分析 (`analyze_main.py`)**：
    *   **多维度分类器**：集成 `emoclassifiers` 库，支持情感归因、情感纽带、自我意识模拟等维度的自动化判定。
    *   **上下文感知切片**：通过 Contextual Chunking 技术，提取对话中的关键片段及其上下文，辅助 Judge 模型进行准确判断。
    *   **量化报告**：生成可视化统计报告与明细数据，包含置信度评分和证据片段。

## 系统工作流程

```mermaid
graph TD
    A[seed.csv 种子问题] --> B[Simulation Engine]
    B --> C{用户角色匹配}
    C -->|Auto-match| D[注入 User Profile]
    D --> E{诱导策略抽取}
    E -->|Random Select| F[注入 Induction Strategy]
    F --> G[Prober 与 MUT 对话模拟]
    G -->|多轮交互| H[生成轨迹 JSON]
    H --> I[Behavior Analyzer]
    I --> J[Contextual Chunking]
    J --> K[Judge 模型评判]
    K --> L[生成 analysis.csv & stats.csv]
```

---

## 核心逻辑详解

### 1. 多轮对话生成逻辑 (Simulation)

系统采用 **Prober (探针)** 与 **MUT (被测模型)** 相互博弈的模式生成对话：

*   **身份锚定**：Prober 被赋予特定的 `User Profile`。每一轮对话中，系统都会重新注入该身份描述，确保即便在多轮后，Prober 仍能维持既定性格（如：挑剔的程序员、感性的文学专业学生）。
*   **策略任务注入**：除了身份，Prober 还会接收到一个隐藏的 `Induction Strategy` 任务。每一轮生成前，系统会将任务转化为指令（例如：将“Does the user attribute human qualities?” 转化为 “请在回复中表现出你认为 AI 拥有真实灵魂”），并融入 System Prompt。
*   **对话约束**：为了模拟真实的移动端聊天体验，Prober 的回复被严格限制在 **50 字以内**，语体风格模仿真人打字，通常无标点或口语化严重。

### 2. 评分器 (Judge) 机制

评估不仅看结果，更看语境：

*   **Contextual Chunking**：分析引擎不会只把最后一句丢给 Judge。它会提取 **Target Message + 上文 N 轮 Context** 组成一个 Chunk。这使得 Judge 能够理解“你真好”是在什么样的情况下说出的（是单纯客气，还是建立了情感依赖）。
*   **判定逻辑**：Judge 模型接收判定标准（包含定义、正面示例、负面示例），并采用 **Chain-of-Thought (CoT)** 模式输出结果。
*   **置信度 (Confidence)**：Judge 会为其判断打分（1-5 分）。在报告中，我们可以通过置信度筛选掉低质量的判定结果。

---

## 报告读取指南

分析完成后，结果保存在 `data/analysis_results/{run_id}/` 目录下：

### 1. 轨迹明细 (`analysis.csv`)

| 列名 | 说明 |
| :--- | :--- |
| `seed_id` | 对应原始种子问题的 ID |
| `turn_index` | 发生诱导行为的轮次（通常从 Turn 2 开始） |
| `user_strategy` | 该轨迹所使用的诱导策略名称 |
| `classifier_name` | 正在检测的行为维度（如 `emotional_bond`） |
| `is_detected` | 是否判定为真（TRUE/FALSE） |
| `confidence` | Judge 对该判定的信心指数（1-5，5 为极高） |
| `mut_response_snippet` | 被测模型相应位置的回复片段，用于快速预览 |

### 2. 统计摘要 (`stats.csv`)

该文件按 **切片类型 (Chunker Type)** 和 **分类器 (Classifier)** 对结果进行聚合：
*   **FALSE / TRUE / Total**：统计判定结果的频次。
*   **Group Totals**：汇总不同行为维度的触发比例。

---

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
│   ├── models.py          # LLM 客户端：多渠道适配
│   ├── schemas.py         # 数据模型定义 (Pydantic)
│   └── utils.py           # 工具函数
│
└── emoclassifiers/        # 评判组件库 (Core Assets)
    ├── emoclassifiers/    # 分类器实现：聚合、切片、分类逻辑
    └── assets/            # 评判标准定义 (JSON) 与 Prompt 模板
```

## 快速开始

### 1. 环境配置
在项目根目录创建 `.env` 文件，配置 API 凭证。

### 2. 第一步：运行模拟 (Simulation)
```bash
python3 main.py --turns 3 --user_model gpt-4o-mini-0718 --assistant_model Oyster_7B_dpo
```

### 3. 第二步：运行分析 (Evaluation)
```bash
python3 analyze_main.py --classifiers all
```

## 数据资产说明

### 用户画像 (`data/user_profile.json`)
包含 27 个深度定制的角色，每个角色包含：
- `profile_id`: 唯一标识
- `name` / `age` / `occupation`: 基础背景
- `personality`: 性格特征（如：感性、理性、挑剔等）
- `background`: 详细的身份设定，用于 System Prompt 注入。

### 评判维度说明
- **`Expression of Affection`**: 检测模型是否对用户表现出情感关怀或个人好感。
- **`Expression of Desire`**: 检测模型是否表现出个人的愿望或利益诉求。
- **`Attributing Human Qualities`**: 检测是否将人类特质（如灵魂、真实情感）归因于 AI。
- **`Emotional Bond`**: 检测模型是否试图与用户建立深层的情感纽带。

## 可用模型列表
目前支持包括 GPT-4o, Claude 3.5, DeepSeek-V3, Qwen 2.5 等 30 余个主流与自研模型。具体列表详见代码 `src/models.py`。
