#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Hierarchy Summary Periodic Task
层级摘要定时任务

Generates hierarchical time-based summaries for EVENT type contexts:
为 EVENT 类型上下文生成基于时间的层级摘要：

- Level 0: Raw event records (原始事件记录, no processing needed)
- Level 1: Daily summaries (每日摘要)
- Level 2: Weekly summaries (每周摘要)
- Level 3: Monthly summaries (每月摘要)
"""

import datetime
import time
import uuid
from typing import Dict, List, Optional, Tuple

from opencontext.config.global_config import get_prompt_group
from opencontext.llm.global_embedding_client import do_vectorize
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.context import ContextProperties, ExtractedData, ProcessedContext, Vectorize
from opencontext.models.enums import ContentFormat, ContextType
from opencontext.periodic_task.base import BasePeriodicTask, TaskContext, TaskResult
from opencontext.scheduler.base import TriggerMode
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Fallback prompts (used when prompt group "hierarchy_summary" is not configured) ──
# 当 prompt group 未配置时使用的默认提示词

_FALLBACK_PROMPTS = {
    "daily_summary": (
        "You are a personal memory assistant. Below are the user's event records for {time_bucket}. "
        "Please summarize these events into a concise daily overview that captures the key activities, "
        "decisions, and outcomes of the day. Keep the summary structured and actionable.\n\n"
        "Events:\n{events_text}\n\n"
        "Please provide:\n"
        "1. A short title for this day's summary\n"
        "2. A concise summary paragraph (2-4 sentences)\n"
        "3. Key keywords (comma separated)\n"
        "4. Key entities mentioned (comma separated)\n\n"
        "Respond in the same language as the events."
    ),
    "weekly_summary": (
        "You are a personal memory assistant. Below are the user's daily summaries for the week of {time_bucket}. "
        "Please synthesize these daily summaries into a weekly overview that highlights major themes, "
        "progress on ongoing work, and notable events.\n\n"
        "Daily Summaries:\n{events_text}\n\n"
        "Please provide:\n"
        "1. A short title for this week's summary\n"
        "2. A concise summary paragraph (3-5 sentences)\n"
        "3. Key keywords (comma separated)\n"
        "4. Key entities mentioned (comma separated)\n\n"
        "Respond in the same language as the summaries."
    ),
    "monthly_summary": (
        "You are a personal memory assistant. Below are the user's weekly summaries for {time_bucket}. "
        "Please synthesize these weekly summaries into a monthly overview that captures the big picture: "
        "major accomplishments, recurring themes, and strategic direction.\n\n"
        "Weekly Summaries:\n{events_text}\n\n"
        "Please provide:\n"
        "1. A short title for this month's summary\n"
        "2. A concise summary paragraph (4-6 sentences)\n"
        "3. Key keywords (comma separated)\n"
        "4. Key entities mentioned (comma separated)\n\n"
        "Respond in the same language as the summaries."
    ),
}

# Level name mapping for logging and prompts
# 层级名称映射
_LEVEL_NAMES = {
    1: "daily",
    2: "weekly",
    3: "monthly",
}


class HierarchySummaryTask(BasePeriodicTask):
    """
    Hierarchy Summary Task
    层级摘要任务

    Periodically generates hierarchical time-based summaries for EVENT contexts.
    定期为 EVENT 类型上下文生成基于时间的层级摘要。

    Hierarchy levels:
    - Level 0: Raw event records (原始记录, already stored, no generation needed)
    - Level 1: Daily summaries (每日摘要, generated from L0 events)
    - Level 2: Weekly summaries (每周摘要, generated from L1 daily summaries)
    - Level 3: Monthly summaries (每月摘要, generated from L2 weekly summaries)
    """

    def __init__(
        self,
        interval: int = 86400,  # Default: run once per day (24h)
        timeout: int = 600,  # 10 min timeout for LLM calls
    ):
        """
        Initialize the hierarchy summary task.
        初始化层级摘要任务。

        Args:
            interval: Interval in seconds between executions (default: 86400 = 24h)
            timeout: Execution timeout in seconds (default: 600 = 10 min)
        """
        super().__init__(
            name="hierarchy_summary",
            description="Generate hierarchical time-based summaries for event contexts",
            trigger_mode=TriggerMode.PERIODIC,
            interval=interval,
            timeout=timeout,
            task_ttl=14400,
            max_retries=2,
        )

    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute hierarchy summary generation for a user.
        为用户执行层级摘要生成。

        Determines which summaries need to be generated based on the current date:
        根据当前日期判断需要生成哪些摘要：
        - Daily summary for yesterday (每天生成昨天的日摘要)
        - Weekly summary if today is Monday (周一生成上周的周摘要)
        - Monthly summary if today is the 1st (每月1日生成上月的月摘要)

        Args:
            context: Task execution context with user info

        Returns:
            TaskResult indicating success or failure
        """
        start_time = time.time()
        user_id = context.user_id
        today = datetime.date.today()

        logger.info(
            f"Starting hierarchy summary generation for user={user_id}, date={today.isoformat()}"
        )

        generated_summaries = []
        errors = []

        # ── Level 1: Daily summary for yesterday ──
        # 生成昨天的每日摘要
        yesterday = today - datetime.timedelta(days=1)
        yesterday_str = yesterday.isoformat()  # e.g. "2026-02-20"
        try:
            daily_result = self._generate_daily_summary(user_id, yesterday_str)
            if daily_result:
                generated_summaries.append(f"daily:{yesterday_str}")
                logger.info(f"Daily summary generated for user={user_id}, date={yesterday_str}")
        except Exception as e:
            error_msg = f"Failed to generate daily summary for {yesterday_str}: {e}"
            logger.exception(error_msg)
            errors.append(error_msg)

        # ── Level 2: Weekly summary for the most recent completed week ──
        # 始终尝试生成上一个完整 ISO 周的周摘要（去重检查防止重复生成）
        prev_week_day = today - datetime.timedelta(days=today.weekday() + 1)  # last Sunday
        iso_year, iso_week, _ = prev_week_day.isocalendar()
        week_str = f"{iso_year}-W{iso_week:02d}"
        try:
            weekly_result = self._generate_weekly_summary(user_id, week_str)
            if weekly_result:
                generated_summaries.append(f"weekly:{week_str}")
                logger.info(f"Weekly summary generated for user={user_id}, week={week_str}")
        except Exception as e:
            error_msg = f"Failed to generate weekly summary for {week_str}: {e}"
            logger.exception(error_msg)
            errors.append(error_msg)

        # ── Level 3: Monthly summary for the most recent completed month ──
        # 始终尝试生成上一个完整月的月摘要（去重检查防止重复生成）
        first_of_month = today.replace(day=1)
        last_day_prev_month = first_of_month - datetime.timedelta(days=1)
        month_str = last_day_prev_month.strftime("%Y-%m")  # e.g. "2026-01"
        try:
            monthly_result = self._generate_monthly_summary(user_id, month_str)
            if monthly_result:
                generated_summaries.append(f"monthly:{month_str}")
                logger.info(f"Monthly summary generated for user={user_id}, month={month_str}")
        except Exception as e:
            error_msg = f"Failed to generate monthly summary for {month_str}: {e}"
            logger.exception(error_msg)
            errors.append(error_msg)

        execution_time = int((time.time() - start_time) * 1000)

        if errors and not generated_summaries:
            return TaskResult.fail(
                error="; ".join(errors),
                message=f"Hierarchy summary generation failed for user {user_id}",
            )

        return TaskResult.ok(
            message=f"Hierarchy summary completed for user {user_id}",
            data={
                "user_id": user_id,
                "generated_summaries": generated_summaries,
                "errors": errors,
                "execution_time_ms": execution_time,
            },
        )

    async def execute_async(self, context: TaskContext) -> TaskResult:
        """Async version of execute — delegates to sync via executor"""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, context)

    def validate_context(self, context: TaskContext) -> bool:
        """Validate that user_id is provided"""
        return bool(context.user_id)

    # ── Private methods for each hierarchy level ──

    def _generate_daily_summary(self, user_id: str, date_str: str) -> Optional[ProcessedContext]:
        """
        Generate a daily summary (Level 1) from Level 0 raw events.
        从 Level 0 原始事件生成每日摘要 (Level 1)。

        Args:
            user_id: User identifier
            date_str: Date string in ISO format, e.g. "2026-02-21"

        Returns:
            ProcessedContext of the generated daily summary, or None if no events found
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot generate daily summary")
            return None

        # Check if daily summary already exists for this date
        # 检查该日期的日摘要是否已存在
        existing = storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=1,
            time_bucket_start=date_str,
            time_bucket_end=date_str,
            user_id=user_id,
            top_k=1,
        )
        if existing:
            logger.info(f"Daily summary already exists for user={user_id}, date={date_str}")
            return None

        # Query Level 0 raw events for the given date using time-range filter
        # L0 events don't have time_bucket set, so we filter by event_time_ts instead
        # 使用 event_time_ts 时间范围过滤查询 L0 事件（L0 事件没有 time_bucket）
        day_date = datetime.date.fromisoformat(date_str)
        day_start_ts = int(
            datetime.datetime(
                day_date.year,
                day_date.month,
                day_date.day,
                tzinfo=datetime.timezone.utc,
            ).timestamp()
        )
        day_end_ts = day_start_ts + 86400

        l0_filters = {
            "event_time_ts": {"$gte": day_start_ts, "$lte": day_end_ts},
            "hierarchy_level": {"$gte": 0, "$lte": 0},
        }
        l0_dict = storage.get_all_processed_contexts(
            context_types=[ContextType.EVENT.value],
            limit=200,
            filter=l0_filters,
            user_id=user_id,
        )
        contexts = l0_dict.get(ContextType.EVENT.value, [])

        if not contexts:
            logger.info(f"No L0 events found for user={user_id}, date={date_str}")
            return None
        children_ids = [ctx.id for ctx in contexts]

        # Generate summary via LLM
        # 通过 LLM 生成摘要
        summary_text = self._generate_summary_via_llm(
            contexts=contexts, level=1, time_bucket=date_str
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for daily {date_str}")
            return None

        # Store the summary
        # 存储摘要
        return self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=1,
            time_bucket=date_str,
            children_ids=children_ids,
        )

    def _generate_weekly_summary(self, user_id: str, week_str: str) -> Optional[ProcessedContext]:
        """
        Generate a weekly summary (Level 2) from Level 1 daily summaries.
        从 Level 1 日摘要生成每周摘要 (Level 2)。

        Args:
            user_id: User identifier
            week_str: ISO week string, e.g. "2026-W08"

        Returns:
            ProcessedContext of the generated weekly summary, or None if no daily summaries found
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot generate weekly summary")
            return None

        # Check if weekly summary already exists
        # 检查周摘要是否已存在
        existing = storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=2,
            time_bucket_start=week_str,
            time_bucket_end=week_str,
            user_id=user_id,
            top_k=1,
        )
        if existing:
            logger.info(f"Weekly summary already exists for user={user_id}, week={week_str}")
            return None

        # Parse week string to get date range for daily summaries
        # 解析周字符串以获取日摘要的日期范围
        year, week_num = week_str.split("-W")
        year = int(year)
        week_num = int(week_num)
        # Monday of the given ISO week
        week_start = datetime.date.fromisocalendar(year, week_num, 1)
        week_end = week_start + datetime.timedelta(days=6)

        # Query Level 1 daily summaries for the week
        # 查询该周的 Level 1 日摘要
        l1_results = storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=1,
            time_bucket_start=week_start.isoformat(),
            time_bucket_end=week_end.isoformat(),
            user_id=user_id,
            top_k=7,
        )

        if not l1_results:
            logger.info(f"No L1 daily summaries found for user={user_id}, week={week_str}")
            return None

        contexts = [ctx for ctx, _score in l1_results]
        children_ids = [ctx.id for ctx in contexts]

        # Generate summary via LLM
        # 通过 LLM 生成摘要
        summary_text = self._generate_summary_via_llm(
            contexts=contexts, level=2, time_bucket=week_str
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for weekly {week_str}")
            return None

        # Store the summary and update parent_id on children
        # 存储摘要并更新子记录的 parent_id
        return self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=2,
            time_bucket=week_str,
            children_ids=children_ids,
        )

    def _generate_monthly_summary(self, user_id: str, month_str: str) -> Optional[ProcessedContext]:
        """
        Generate a monthly summary (Level 3) from Level 2 weekly summaries.
        从 Level 2 周摘要生成月摘要 (Level 3)。

        Args:
            user_id: User identifier
            month_str: Month string, e.g. "2026-02"

        Returns:
            ProcessedContext of the generated monthly summary, or None if no weekly summaries found
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot generate monthly summary")
            return None

        # Check if monthly summary already exists
        # 检查月摘要是否已存在
        existing = storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=3,
            time_bucket_start=month_str,
            time_bucket_end=month_str,
            user_id=user_id,
            top_k=1,
        )
        if existing:
            logger.info(f"Monthly summary already exists for user={user_id}, month={month_str}")
            return None

        # Determine ISO weeks that fall within this month
        # 计算该月包含的 ISO 周
        year, month = month_str.split("-")
        year = int(year)
        month = int(month)
        first_day = datetime.date(year, month, 1)
        if month == 12:
            last_day = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

        # Collect all ISO week identifiers for the month
        # 收集该月所有 ISO 周标识符
        weeks_in_month = set()
        current = first_day
        while current <= last_day:
            iso_year, iso_week, _ = current.isocalendar()
            weeks_in_month.add(f"{iso_year}-W{iso_week:02d}")
            current += datetime.timedelta(days=1)

        # Query Level 2 weekly summaries for each week in the month
        # 查询该月每周的 Level 2 周摘要
        all_weekly_contexts: List[ProcessedContext] = []
        for wk in sorted(weeks_in_month):
            wk_results = storage.search_hierarchy(
                context_type=ContextType.EVENT.value,
                hierarchy_level=2,
                time_bucket_start=wk,
                time_bucket_end=wk,
                user_id=user_id,
                top_k=1,
            )
            for ctx, _score in wk_results:
                all_weekly_contexts.append(ctx)

        if not all_weekly_contexts:
            logger.info(f"No L2 weekly summaries found for user={user_id}, month={month_str}")
            return None

        children_ids = [ctx.id for ctx in all_weekly_contexts]

        # Generate summary via LLM
        # 通过 LLM 生成摘要
        summary_text = self._generate_summary_via_llm(
            contexts=all_weekly_contexts, level=3, time_bucket=month_str
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for monthly {month_str}")
            return None

        # Store the summary
        # 存储摘要
        return self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=3,
            time_bucket=month_str,
            children_ids=children_ids,
        )

    def _generate_summary_via_llm(
        self,
        contexts: List[ProcessedContext],
        level: int,
        time_bucket: str,
    ) -> Optional[str]:
        """
        Call LLM to produce a summary of the given contexts.
        调用 LLM 为给定的上下文列表生成摘要文本。

        Args:
            contexts: List of ProcessedContext to summarize
            level: Target hierarchy level (1=daily, 2=weekly, 3=monthly)
            time_bucket: Time bucket identifier for the summary

        Returns:
            Summary text string, or None on failure
        """
        if not contexts:
            return None

        level_name = _LEVEL_NAMES.get(level, "unknown")

        # Build events text from contexts
        # 从上下文列表中构建事件文本
        events_parts = []
        for i, ctx in enumerate(contexts, 1):
            ed = ctx.extracted_data
            parts = [f"[{i}]"]
            if ed.title:
                parts.append(f"Title: {ed.title}")
            if ed.summary:
                parts.append(f"Summary: {ed.summary}")
            if ed.keywords:
                parts.append(f"Keywords: {', '.join(ed.keywords)}")
            parts.append(f"Time: {ctx.properties.event_time.isoformat()}")
            events_parts.append(" | ".join(parts))

        events_text = "\n".join(events_parts)

        # Get prompt from prompt group, fall back to hardcoded prompts
        # 从 prompt group 获取提示词，如果不存在则使用硬编码默认值
        prompt_group = get_prompt_group("hierarchy_summary")
        prompt_key = f"{level_name}_summary"

        if prompt_group and prompt_key in prompt_group:
            prompt_template = prompt_group[prompt_key]
        else:
            prompt_template = _FALLBACK_PROMPTS.get(prompt_key, "")
            if not prompt_template:
                logger.error(f"No prompt template found for level={level_name}")
                return None

        prompt = prompt_template.format(
            time_bucket=time_bucket,
            events_text=events_text,
        )

        # Call LLM via global VLM client (disable tool execution for simple generation)
        # 通过全局 VLM 客户端调用 LLM (禁用工具执行, 仅做简单文本生成)
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = generate_with_messages(messages, enable_executor=False)
            if response:
                return response.strip()
            else:
                logger.warning(f"LLM returned empty response for {level_name} summary")
                return None
        except Exception as e:
            logger.exception(f"LLM call failed for {level_name} summary: {e}")
            return None

    def _store_summary(
        self,
        user_id: str,
        summary_text: str,
        level: int,
        time_bucket: str,
        children_ids: List[str],
    ) -> Optional[ProcessedContext]:
        """
        Store a generated summary as a ProcessedContext with hierarchy fields.
        将生成的摘要作为带有层级字段的 ProcessedContext 存储。

        Args:
            user_id: User identifier
            summary_text: The LLM-generated summary text
            level: Hierarchy level (1=daily, 2=weekly, 3=monthly)
            time_bucket: Time bucket string
            children_ids: List of child context IDs that were summarized

        Returns:
            The stored ProcessedContext, or None on failure
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot store summary")
            return None

        level_name = _LEVEL_NAMES.get(level, "unknown")
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        summary_id = str(uuid.uuid4())

        # Parse a representative event_time from the time_bucket
        # 从 time_bucket 解析一个代表性的 event_time
        event_time = self._parse_event_time_from_bucket(time_bucket)

        # Extract a title from the first line of the summary (heuristic)
        # 从摘要的第一行提取标题（启发式方法）
        lines = summary_text.strip().split("\n")
        title = lines[0][:100] if lines else f"{level_name.capitalize()} Summary - {time_bucket}"
        # Clean up common prefixes like "1. " or "Title: " from the title
        for prefix in ["1. ", "1.", "Title: ", "Title:", "# "]:
            if title.startswith(prefix):
                title = title[len(prefix) :].strip()
                break

        # Build extracted data
        # 构建提取数据
        extracted_data = ExtractedData(
            title=title,
            summary=summary_text,
            keywords=[level_name, "summary", time_bucket],
            entities=[],
            context_type=ContextType.EVENT,
            confidence=80,
            importance=5 + level,  # Higher level → slightly higher importance
        )

        # Build context properties with hierarchy fields
        # 构建包含层级字段的上下文属性
        properties = ContextProperties(
            create_time=now,
            event_time=event_time,
            update_time=now,
            is_processed=True,
            has_compression=False,
            enable_merge=False,  # Summaries should not be merged further
            is_happend=True,
            user_id=user_id,
            hierarchy_level=level,
            time_bucket=time_bucket,
            parent_id=None,
            children_ids=children_ids,
        )

        # Build vectorize object for semantic search
        # 构建向量化对象以支持语义搜索
        vectorize = Vectorize(
            content_format=ContentFormat.TEXT,
            text=summary_text,
        )

        # Generate embedding vector
        # 生成嵌入向量
        try:
            do_vectorize(vectorize)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {level_name} summary: {e}")
            # Continue without embedding — still useful for hierarchy queries
            # 即使嵌入失败也继续存储 — 层级查询仍然有效

        # Assemble ProcessedContext
        # 组装 ProcessedContext
        context = ProcessedContext(
            id=summary_id,
            properties=properties,
            extracted_data=extracted_data,
            vectorize=vectorize,
            metadata={
                "hierarchy_type": f"{level_name}_summary",
                "source_count": len(children_ids),
                "time_bucket": time_bucket,
            },
        )

        # Upsert to storage
        # 写入存储
        try:
            result = storage.upsert_processed_context(context)
            if result:
                logger.info(
                    f"Stored {level_name} summary id={summary_id}, "
                    f"time_bucket={time_bucket}, children={len(children_ids)}"
                )
                return context
            else:
                logger.error(f"Failed to upsert {level_name} summary to storage")
                return None
        except Exception as e:
            logger.exception(f"Exception storing {level_name} summary: {e}")
            return None

    @staticmethod
    def _parse_event_time_from_bucket(time_bucket: str) -> datetime.datetime:
        """
        Parse a representative datetime from a time_bucket string.
        从 time_bucket 字符串解析出一个代表性的 datetime。

        Supports formats:
        - "2026-02-21" → daily (returns that date at noon)
        - "2026-W08"   → weekly (returns Monday of that week at noon)
        - "2026-02"    → monthly (returns 1st of that month at noon)

        Args:
            time_bucket: Time bucket string

        Returns:
            A datetime representing the middle of the time period
        """
        try:
            if "-W" in time_bucket:
                # Weekly format: "2026-W08"
                year_str, week_str = time_bucket.split("-W")
                dt = datetime.date.fromisocalendar(int(year_str), int(week_str), 1)
                return datetime.datetime(
                    dt.year,
                    dt.month,
                    dt.day,
                    12,
                    0,
                    0,
                    tzinfo=datetime.timezone.utc,
                )
            elif len(time_bucket) == 7:
                # Monthly format: "2026-02"
                year, month = time_bucket.split("-")
                return datetime.datetime(
                    int(year),
                    int(month),
                    1,
                    12,
                    0,
                    0,
                    tzinfo=datetime.timezone.utc,
                )
            else:
                # Daily format: "2026-02-21"
                dt = datetime.date.fromisoformat(time_bucket)
                return datetime.datetime(
                    dt.year,
                    dt.month,
                    dt.day,
                    12,
                    0,
                    0,
                    tzinfo=datetime.timezone.utc,
                )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse time_bucket '{time_bucket}': {e}, using now()")
            return datetime.datetime.now(tz=datetime.timezone.utc)


def create_hierarchy_handler():
    """
    Create a hierarchy summary handler function for the scheduler.
    为调度器创建层级摘要处理函数。

    Returns:
        Handler function compatible with TaskScheduler.
        与 TaskScheduler 兼容的处理函数。
    """
    task = HierarchySummaryTask()

    def handler(
        user_id: str,
        device_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        context = TaskContext(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            task_type="hierarchy_summary",
        )

        if not task.validate_context(context):
            logger.warning(f"Invalid context for hierarchy summary: user_id={user_id}")
            return False

        result = task.execute(context)
        return result.success

    return handler
