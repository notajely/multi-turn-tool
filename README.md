# Multi-Turn Anthropomorphism Probing Tool

这是一个用于探测和分析 AI 模型“过度拟合拟人化”倾向的多轮对话模拟与分析工具。

## 核心功能

1.  **多轮对话模拟 (`main.py`)**：
    *   **角色扮演 (User Profiling)**：内置 27 种真实人类用户模板（从 K12 学生到 AI 工作流架构师），让 User 模型在对话中保持稳定的身份设定。
    *   **诱导策略 (Induction Strategies)**：自动从策略库中抽取诱导指令（如感性质疑、情感投射等），并将其自然融入 User 模型的回复中。
    *   **自动化运行**：支持从 CSV 种子文件批量生成多轮对话轨迹，支持多线程并发。

2. **行为评判 (Judge/Evaluation)**：
    *   **多维度分类器 (Classifiers)**：基于 `emoclassifiers` 库，内置了如“情感归因 (Attribute Emotions)”、“建立情感纽带 (Emotional Bond)”等多个维度的评判标准。
    *   **上下文感知切片 (Chunking)**：自动对多轮对话进行切片，提取包含上下文的对话片段（Chunk），确保 Judge 模型能准确理解语境。
    *   **结构化评判逻辑**：利用专门的 Prompt 模板引导 Judge 模型进行二分类判定（True/False）并给出置信度（1-5分）。
    *   **自动化分析报告**：汇总所有轨迹的评判结果，生成包含证据片段的 CSV 报告，便于量化分析模型的拟人化程度。

## 快速开始

### 1. 环境配置
在根目录创建 `.env` 文件并配置相关 API Key：
```env
IDEALAB_API_KEY=your_key
IDEALAB_BASE_URL=your_url
WHALE_API_KEY=your_key
WHALE_BASE_URL=your_url
```

### 2. 运行模拟
使用种子文件 `seed.csv` 启动模拟：
```bash
# 随机角色，运行 3 轮对话
python main.py --turns 3 --max_workers 5

# 指定特定角色 (例如 ID 为 3 的 DeepSeek 硬核技术粉)
python main.py --turns 3 --profile_id 3 --user_model gpt-4o-mini
```

### 3. 运行分析
对生成的对话轨迹进行拟人化倾向分析：
```bash
# 分析最新的轨迹目录
python analyze_main.py --judge_model gpt-4o --classifiers all
```

## 项目结构
*   `src/engine.py`: 模拟引擎，负责角色注入、多轮逻辑及策略调度。
*   `src/analyzer.py`: 评判核心逻辑，负责调用 `emoclassifiers` 进行行为检测。
*   `src/models.py`: 多渠道 LLM 客户端封装（支持 DashScope, Whale, Idealab 等）。
*   `data/user_profile.json`: 27 个预设用户画像，涵盖 K12、职场、银发族等。
*   `emoclassifiers/`: 核心评判组件库：
    *   `assets/definitions/`: 定义了各种拟人化行为的判定标准（Criteria）。
    *   `emoclassifiers/chunking.py`: 负责对话片段的智能切片。
    *   `emoclassifiers/prompt_templates.py`: 评判专用的 Prompt 模板。

## 效果示例
生成的轨迹文件将保存在 `data/trajectories/{run_id}/` 下，包含完整的对话上下文及所使用的诱导策略、用户画像等元数据。
分析结果将汇总至 `data/analysis_results/{run_id}/analysis.csv`。
