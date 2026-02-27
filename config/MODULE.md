# Config Module — 提示词 (Prompts) 使用文档

## 提示词加载机制

- **配置文件**: `config/prompts_zh.yaml` (中文) / `config/prompts_en.yaml` (英文)
- **加载器**: `PromptManager` (`opencontext/config/prompt_manager.py`)
  - `get_prompt(name)` — 用点号分隔的 key 获取单个提示词字符串
  - `get_prompt_group(name)` — 获取提示词组 (通常包含 `system` 和 `user` 两个 key)
- **全局访问**: 通过 `GlobalConfig` (`opencontext/config/global_config.py`) 的 `get_prompt()` / `get_prompt_group()` 函数

---

## 提示词使用全景

### 1. `chat_workflow` — 聊天工作流 (10 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `chat_workflow.intent_analysis` | Intent 节点 | `context_consumption/context_agent/nodes/intent.py:161` | 分析并增强复杂用户查询 |
| `chat_workflow.query_classification` | Intent 节点 | `context_consumption/context_agent/nodes/intent.py:74` | 将查询分类为 `simple_chat` 或 `qa_analysis` |
| `chat_workflow.social_interaction` | Intent 节点 | `context_consumption/context_agent/nodes/intent.py:110` | 为社交/闲聊类问题生成友好回复 |
| `chat_workflow.executor.generate` | Executor 节点 | `context_consumption/context_agent/nodes/executor.py:122` | 基于上下文生成新内容 |
| `chat_workflow.executor.edit` | Executor 节点 | `context_consumption/context_agent/nodes/executor.py:166` | 编辑/改写已有内容 |
| `chat_workflow.executor.answer` | Executor 节点 | `context_consumption/context_agent/nodes/executor.py:239` | 回答问题、总结分析 |
| `chat_workflow.context_collection.tool_analysis` | LLM 上下文策略 | `context_consumption/context_agent/core/llm_context_strategy.py:52` | 分析信息缺口，规划工具调用 |
| `chat_workflow.context_collection.tool_result_validation` | LLM 上下文策略 | `context_consumption/context_agent/core/llm_context_strategy.py:356` | 过滤验证工具返回结果的相关性 |
| `chat_workflow.context_collection.sufficiency_evaluation` | LLM 上下文策略 | `context_consumption/context_agent/core/llm_context_strategy.py:142` | 评估收集的上下文是否足够回答问题 |
| `chat_workflow.context_collection.context_filter` | **未被使用** | — | 判断上下文条目与问题的相关性 |

### 2. `processing` — 处理管道 (1 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `processing.extraction.chat_analyze` | TextChatProcessor | `context_processing/processor/text_chat_processor.py:77` | 从聊天记录中提取结构化记忆 (profile/entity/event/knowledge/document) |

### 3. `merging` — 合并 (3 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `merging.context_merging_multiple` | ContextMerger (fallback) | `context_processing/merger/context_merger.py:285` | 通用多上下文合并 (后备方案) |
| `merging.knowledge_merging` | ContextMerger | `context_processing/merger/context_merger.py:278` | knowledge 类型专用合并 (通过动态 key `merging.{type}_merging` 调用) |
| `merging.overwrite_merge` | ProfileProcessor | `context_processing/processor/profile_processor.py:120` | profile 新旧信息 LLM 智能合并 |

### 4. `entity_processing` — 实体处理 (3 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `entity_processing.entity_extraction` | Intent 节点 | `context_consumption/context_agent/nodes/intent.py:231` | 从用户查询中提取命名实体 (person/project/team/organization) |
| `entity_processing.entity_meta_merging` | **未被使用** | — | 实体元数据合并 (新旧实体数据融合) |
| `entity_processing.entity_matching` | **未被使用** | — | 实体名称匹配 (判断提取的实体是否匹配已存储实体) |

### 5. `document_processing` — 文档处理 (3 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `document_processing.vlm_analysis` | DocumentProcessor | `context_processing/processor/document_processor.py:609` | 用 VLM 从图片/文档中提取文字内容 |
| `document_processing.text_chunking` | DocumentTextChunker | `context_processing/chunker/document_text_chunker.py:194` | LLM 智能文本分块 |
| `document_processing.global_semantic_chunking` | DocumentTextChunker | `context_processing/chunker/document_text_chunker.py:272` | 全局语义分块 (保留原文，添加上下文前缀) |

### 6. `hierarchy_summary` — 层级摘要 (10 个 key)

| 提示词 Key | 使用位置 | 文件路径:行号 | 作用 |
|---|---|---|---|
| `hierarchy_summary.system` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 所有摘要的基础系统提示 |
| `hierarchy_summary.daily_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 生成日度活动摘要 (L1) |
| `hierarchy_summary.weekly_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 生成周度活动摘要 (L2) |
| `hierarchy_summary.monthly_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 生成月度活动摘要 (L3) |
| `hierarchy_summary.daily_partial_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 日度分批摘要 (内容超 token 限制时) |
| `hierarchy_summary.weekly_partial_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 周度分批摘要 |
| `hierarchy_summary.monthly_partial_summary` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1101` | 月度分批摘要 |
| `hierarchy_summary.daily_merge` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1179` | 合并多个日度分批摘要为完整日摘要 |
| `hierarchy_summary.weekly_merge` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1179` | 合并多个周度分批摘要为完整周摘要 |
| `hierarchy_summary.monthly_merge` | HierarchySummaryTask | `periodic_task/hierarchy_summary.py:1179` | 合并多个月度分批摘要为完整月摘要 |

---

## 问题清单

### YAML 中定义但代码中未使用的提示词 (3 个)

| 提示词 Key | 设计意图 | 备注 |
|---|---|---|
| `chat_workflow.context_collection.context_filter` | 判断上下文与问题的相关性，返回相关 ID | 可能被 `tool_result_validation` 取代 |
| `entity_processing.entity_meta_merging` | 合并新旧实体元数据 | 实体处理流程可能尚未完整实现 |
| `entity_processing.entity_matching` | 判断提取实体是否匹配已存储实体 | 同上 |

### 代码中引用但 YAML 中未定义的提示词 (3 个)

| 提示词 Key | 引用位置 | 文件路径:行号 | 风险 |
|---|---|---|---|
| `processing.extraction.screenshot_analyze` | ScreenshotProcessor | `context_processing/processor/screenshot_processor.py:416` | `get_prompt_group()` 返回 None，运行时会抛 ValueError |
| `merging.screenshot_batch_merging` | ScreenshotProcessor | `context_processing/processor/screenshot_processor.py:544` | 同上 |
| `completion_service.semantic_continuation` | CompletionService | `context_consumption/completion/completion_service.py:248` | `get_prompt_group()` 返回 None，运行时出错 |

---

## 统计

- **YAML 中定义的提示词总数**: 30 个
- **实际被代码使用**: 27 个
- **未使用**: 3 个
- **代码引用但 YAML 缺失**: 3 个
