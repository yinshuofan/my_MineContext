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
import json
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

# ── Token estimation constants ──
# Token 估算常量

_MAX_INPUT_TOKENS = 60000  # Conservative limit (model supports ~128K, but quality degrades)
_BATCH_TOKEN_TARGET = 25000  # Per-batch limit for overflow splitting
_PROMPT_OVERHEAD_TOKENS = 800  # System prompt + formatting overhead


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for mixed Chinese/English text.
    估算中英文混合文本的 token 数。

    Chinese characters ≈ 1.5 tokens/char, ASCII ≈ 0.25 tokens/char.
    Conservative for mixed content to avoid overflowing context windows.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    chinese_chars = 0
    ascii_chars = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            chinese_chars += 1
        else:
            ascii_chars += 1
    return int(chinese_chars * 1.5 + ascii_chars * 0.25)


# ── Fallback prompts (used when prompt group "hierarchy_summary" is not configured) ──
# 当 prompt group 未配置时使用的默认提示词

_FALLBACK_PROMPTS = {
    # ── Normal summary prompts ──
    "daily_summary": (
        "Please summarize the following user activities for the time period: {time_period}\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "weekly_summary": (
        "Please summarize the following user activities for the time period: {time_period}\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "monthly_summary": (
        "Please summarize the following user activities for the time period: {time_period}\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    # ── Partial (batch sub-summary) prompts ──
    "daily_partial_summary": (
        "Please summarize this PARTIAL batch of user activities for {time_period}. "
        "This is batch {batch_info} — summarize only these events, preserving key details.\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "weekly_partial_summary": (
        "Please summarize this PARTIAL batch of user activities for {time_period}. "
        "This is batch {batch_info} — summarize only these days, preserving key details.\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "monthly_partial_summary": (
        "Please summarize this PARTIAL batch of user activities for {time_period}. "
        "This is batch {batch_info} — summarize only these weeks, preserving key details.\n\n"
        "**Activity Records**:\n{activity_records}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    # ── Merge prompts ──
    "daily_merge": (
        "Please merge the following partial daily summaries into a single cohesive daily summary "
        "for {time_period}.\n\n"
        "**Partial Summaries**:\n{partial_summaries}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "weekly_merge": (
        "Please merge the following partial weekly summaries into a single cohesive weekly summary "
        "for {time_period}.\n\n"
        "**Partial Summaries**:\n{partial_summaries}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
    ),
    "monthly_merge": (
        "Please merge the following partial monthly summaries into a single cohesive monthly summary "
        "for {time_period}.\n\n"
        "**Partial Summaries**:\n{partial_summaries}\n\n"
        "Please generate a JSON summary object with: title, summary, keywords, entities, importance."
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
            trigger_mode=TriggerMode.USER_ACTIVITY,
            interval=interval,
            timeout=timeout,
            task_ttl=14400,
            max_retries=2,
        )

    async def execute(self, context: TaskContext) -> TaskResult:
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
            daily_result = await self._generate_daily_summary(user_id, yesterday_str)
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
            weekly_result = await self._generate_weekly_summary(user_id, week_str)
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
            monthly_result = await self._generate_monthly_summary(user_id, month_str)
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

    def validate_context(self, context: TaskContext) -> bool:
        """Validate that user_id is provided"""
        return bool(context.user_id)

    # ── Content formatting methods ──

    @staticmethod
    def _format_l0_events(contexts: List[ProcessedContext]) -> str:
        """
        Format L0 raw events for L1 (daily) summary generation.
        将 L0 原始事件格式化为 L1（每日）摘要生成的输入。

        Events are sorted by time, numbered, with key fields shown.

        Args:
            contexts: List of L0 ProcessedContext events

        Returns:
            Formatted text string
        """
        sorted_ctxs = sorted(
            contexts,
            key=lambda c: (c.properties.event_time.isoformat() if c.properties.event_time else ""),
        )
        parts = []
        for i, ctx in enumerate(sorted_ctxs, 1):
            ed = ctx.extracted_data
            line_parts = [f"[{i}]"]
            if ed.title:
                line_parts.append(f"Title: {ed.title}")
            if ctx.properties.event_time:
                line_parts.append(f"Time: {ctx.properties.event_time.strftime('%H:%M')}")
            if ed.summary:
                line_parts.append(f"Summary: {ed.summary}")
            if ed.keywords:
                line_parts.append(f"Keywords: {', '.join(ed.keywords)}")
            parts.append(" | ".join(line_parts))
        return "\n".join(parts)

    @staticmethod
    def _format_weekly_hierarchical(
        l1_summaries: List[ProcessedContext],
        l0_events_by_day: Dict[str, List[ProcessedContext]],
    ) -> str:
        """
        Format weekly content hierarchically: L1 daily summaries + L0 raw events per day.
        分层格式化每周内容：L1 每日摘要 + 每天的 L0 原始事件。

        For L2 (weekly) summary generation. Includes both summary and raw event detail.

        Args:
            l1_summaries: L1 daily summary contexts for the week
            l0_events_by_day: Dict mapping date_str -> list of L0 events for that day

        Returns:
            Formatted hierarchical text string
        """
        # Build a map from date_str -> L1 summary
        l1_by_day: Dict[str, ProcessedContext] = {}
        for ctx in l1_summaries:
            tb = ctx.properties.time_bucket
            if tb:
                l1_by_day[tb] = ctx

        # Collect all day keys (from both L1 and L0)
        all_days = sorted(set(list(l1_by_day.keys()) + list(l0_events_by_day.keys())))

        parts = []
        for day_str in all_days:
            # Determine day-of-week label
            try:
                day_date = datetime.date.fromisoformat(day_str)
                dow = day_date.strftime("%a")
            except (ValueError, TypeError):
                dow = ""

            l1_ctx = l1_by_day.get(day_str)
            l0_list = l0_events_by_day.get(day_str, [])

            if l1_ctx:
                ed = l1_ctx.extracted_data
                parts.append(f"=== Daily Summary: {day_str} ({dow}) ===")
                parts.append(ed.summary if ed.summary else "(no summary)")
                if l0_list:
                    parts.append("")
                    parts.append("  --- Raw Events ---")
                    sorted_l0 = sorted(
                        l0_list,
                        key=lambda c: (
                            c.properties.event_time.isoformat() if c.properties.event_time else ""
                        ),
                    )
                    for j, evt in enumerate(sorted_l0, 1):
                        evd = evt.extracted_data
                        line = [f"  [{j}]"]
                        if evd.title:
                            line.append(f"Title: {evd.title}")
                        if evt.properties.event_time:
                            line.append(f"Time: {evt.properties.event_time.strftime('%H:%M')}")
                        if evd.summary:
                            line.append(f"Summary: {evd.summary}")
                        parts.append(" | ".join(line))
            elif l0_list:
                # Unsummarized day — only L0 events exist
                parts.append(f"=== Unsummarized Day: {day_str} ({dow}) ===")
                sorted_l0 = sorted(
                    l0_list,
                    key=lambda c: (
                        c.properties.event_time.isoformat() if c.properties.event_time else ""
                    ),
                )
                for j, evt in enumerate(sorted_l0, 1):
                    evd = evt.extracted_data
                    line = [f"  [{j}]"]
                    if evd.title:
                        line.append(f"Title: {evd.title}")
                    if evt.properties.event_time:
                        line.append(f"Time: {evt.properties.event_time.strftime('%H:%M')}")
                    if evd.summary:
                        line.append(f"Summary: {evd.summary}")
                    parts.append(" | ".join(line))

            parts.append("")  # blank line between days

        return "\n".join(parts).strip()

    @staticmethod
    def _format_monthly_hierarchical(
        l2_summaries: List[ProcessedContext],
        l1_summaries_by_week: Dict[str, List[ProcessedContext]],
    ) -> str:
        """
        Format monthly content hierarchically: L2 weekly summaries + L1 daily summaries per week.
        分层格式化每月内容：L2 每周摘要 + 每周的 L1 每日摘要。

        For L3 (monthly) summary generation.

        Args:
            l2_summaries: L2 weekly summary contexts for the month
            l1_summaries_by_week: Dict mapping week_str -> list of L1 daily summaries

        Returns:
            Formatted hierarchical text string
        """
        # Build map from week_str -> L2 summary
        l2_by_week: Dict[str, ProcessedContext] = {}
        for ctx in l2_summaries:
            tb = ctx.properties.time_bucket
            if tb:
                l2_by_week[tb] = ctx

        # Collect all week keys
        all_weeks = sorted(set(list(l2_by_week.keys()) + list(l1_summaries_by_week.keys())))

        parts = []
        for week_str in all_weeks:
            l2_ctx = l2_by_week.get(week_str)
            l1_list = l1_summaries_by_week.get(week_str, [])

            if l2_ctx:
                ed = l2_ctx.extracted_data
                parts.append(f"=== Weekly Summary: {week_str} ===")
                parts.append(ed.summary if ed.summary else "(no summary)")

                if l1_list:
                    parts.append("")
                    # Sort L1 summaries by time_bucket (date)
                    sorted_l1 = sorted(
                        l1_list,
                        key=lambda c: c.properties.time_bucket or "",
                    )
                    for l1_ctx in sorted_l1:
                        tb = l1_ctx.properties.time_bucket or "unknown"
                        try:
                            day_date = datetime.date.fromisoformat(tb)
                            dow = day_date.strftime("%a")
                        except (ValueError, TypeError):
                            dow = ""
                        l1_ed = l1_ctx.extracted_data
                        parts.append(f"  --- Daily: {tb} ({dow}) ---")
                        parts.append(f"  {l1_ed.summary if l1_ed.summary else '(no summary)'}")
                        parts.append("")
            elif l1_list:
                # No L2 summary for this week, but L1 summaries exist
                parts.append(f"=== Unsummarized Week: {week_str} ===")
                sorted_l1 = sorted(
                    l1_list,
                    key=lambda c: c.properties.time_bucket or "",
                )
                for l1_ctx in sorted_l1:
                    tb = l1_ctx.properties.time_bucket or "unknown"
                    try:
                        day_date = datetime.date.fromisoformat(tb)
                        dow = day_date.strftime("%a")
                    except (ValueError, TypeError):
                        dow = ""
                    l1_ed = l1_ctx.extracted_data
                    parts.append(f"  --- Daily: {tb} ({dow}) ---")
                    parts.append(f"  {l1_ed.summary if l1_ed.summary else '(no summary)'}")
                    parts.append("")

            parts.append("")  # blank line between weeks

        return "\n".join(parts).strip()

    # ── Batch splitting and overflow handling ──

    @staticmethod
    def _split_into_batches(
        contexts: List[ProcessedContext], max_tokens_per_batch: int
    ) -> List[List[ProcessedContext]]:
        """
        Split a list of contexts into batches that fit within the token budget.
        将上下文列表拆分为符合 token 预算的批次。

        Maintains chronological order. At least 1 context per batch.

        Args:
            contexts: List of ProcessedContext to split
            max_tokens_per_batch: Maximum tokens per batch

        Returns:
            List of batches, each batch is a list of ProcessedContext
        """
        if not contexts:
            return []

        batches: List[List[ProcessedContext]] = []
        current_batch: List[ProcessedContext] = []
        current_tokens = 0

        for ctx in contexts:
            text = ctx.extracted_data.summary or ctx.extracted_data.title or ""
            ctx_tokens = _estimate_tokens(text)

            # Always put at least 1 context in each batch
            if current_batch and (current_tokens + ctx_tokens) > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [ctx]
                current_tokens = ctx_tokens
            else:
                current_batch.append(ctx)
                current_tokens += ctx_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _batch_summarize_and_merge(
        self,
        contexts: List[ProcessedContext],
        level: int,
        time_bucket: str,
        format_fn,
    ) -> Optional[str]:
        """
        Main overflow handler for L1 daily summaries.
        L1 每日摘要的主要溢出处理程序。

        1. Format all contexts, estimate tokens
        2. If fits -> single LLM call (normal path)
        3. If exceeds -> split into batches -> generate sub-summary per batch -> merge

        Args:
            contexts: List of L0 contexts to summarize
            level: Hierarchy level (1)
            time_bucket: Time bucket string
            format_fn: Function to format contexts into text

        Returns:
            Summary text or None
        """
        formatted_text = format_fn(contexts)
        total_tokens = _estimate_tokens(formatted_text) + _PROMPT_OVERHEAD_TOKENS

        if total_tokens <= _MAX_INPUT_TOKENS:
            # Normal path — fits in a single LLM call
            return await self._call_llm_for_summary(
                formatted_content=formatted_text,
                level=level,
                time_bucket=time_bucket,
            )

        # Overflow path — split into batches
        logger.info(
            f"Token overflow for {_LEVEL_NAMES.get(level)} {time_bucket}: "
            f"~{total_tokens} tokens, splitting into batches"
        )
        batches = self._split_into_batches(contexts, _BATCH_TOKEN_TARGET)
        sub_summaries = []

        for idx, batch in enumerate(batches):
            batch_text = format_fn(batch)
            batch_info = f"{idx + 1}/{len(batches)}"
            sub_summary = await self._call_llm_for_summary(
                formatted_content=batch_text,
                level=level,
                time_bucket=time_bucket,
                is_partial=True,
                batch_info=batch_info,
            )
            if sub_summary:
                sub_summaries.append(sub_summary)

        if not sub_summaries:
            logger.warning(f"All batch sub-summaries returned empty for {time_bucket}")
            return None

        if len(sub_summaries) == 1:
            return sub_summaries[0]

        # Merge sub-summaries
        merged_text = "\n\n---\n\n".join(
            f"[Part {i + 1}]\n{s}" for i, s in enumerate(sub_summaries)
        )
        return await self._call_llm_for_merge(
            sub_summaries_text=merged_text,
            level=level,
            time_bucket=time_bucket,
        )

    async def _batch_summarize_weekly(
        self,
        l1_contexts: List[ProcessedContext],
        l0_events_by_day: Dict[str, List[ProcessedContext]],
        time_bucket: str,
    ) -> Optional[str]:
        """
        Weekly overflow handler — batches by day-groups for coherence.
        每周溢出处理 — 按天分组批处理以保持连贯性。

        Args:
            l1_contexts: L1 daily summary contexts for the week
            l0_events_by_day: Dict mapping date_str -> L0 events
            time_bucket: Week string like "2026-W08"

        Returns:
            Summary text or None
        """
        formatted_text = self._format_weekly_hierarchical(l1_contexts, l0_events_by_day)
        total_tokens = _estimate_tokens(formatted_text) + _PROMPT_OVERHEAD_TOKENS

        if total_tokens <= _MAX_INPUT_TOKENS:
            return await self._call_llm_for_summary(
                formatted_content=formatted_text,
                level=2,
                time_bucket=time_bucket,
            )

        # Overflow — batch by day-groups (2-3 days per batch)
        logger.info(
            f"Token overflow for weekly {time_bucket}: "
            f"~{total_tokens} tokens, splitting by day-groups"
        )
        l1_by_day: Dict[str, ProcessedContext] = {}
        for ctx in l1_contexts:
            tb = ctx.properties.time_bucket
            if tb:
                l1_by_day[tb] = ctx

        all_days = sorted(set(list(l1_by_day.keys()) + list(l0_events_by_day.keys())))
        # Split days into groups of 2-3
        day_groups: List[List[str]] = []
        for i in range(0, len(all_days), 3):
            day_groups.append(all_days[i : i + 3])

        sub_summaries = []
        for idx, day_group in enumerate(day_groups):
            group_l1 = [l1_by_day[d] for d in day_group if d in l1_by_day]
            group_l0 = {d: l0_events_by_day[d] for d in day_group if d in l0_events_by_day}
            group_text = self._format_weekly_hierarchical(group_l1, group_l0)
            batch_info = f"{idx + 1}/{len(day_groups)}"
            sub_summary = await self._call_llm_for_summary(
                formatted_content=group_text,
                level=2,
                time_bucket=time_bucket,
                is_partial=True,
                batch_info=batch_info,
            )
            if sub_summary:
                sub_summaries.append(sub_summary)

        if not sub_summaries:
            return None
        if len(sub_summaries) == 1:
            return sub_summaries[0]

        merged_text = "\n\n---\n\n".join(
            f"[Part {i + 1}]\n{s}" for i, s in enumerate(sub_summaries)
        )
        return await self._call_llm_for_merge(
            sub_summaries_text=merged_text,
            level=2,
            time_bucket=time_bucket,
        )

    async def _batch_summarize_monthly(
        self,
        l2_contexts: List[ProcessedContext],
        l1_by_week: Dict[str, List[ProcessedContext]],
        time_bucket: str,
    ) -> Optional[str]:
        """
        Monthly overflow handler — batches by week-groups.
        每月溢出处理 — 按周分组批处理。

        Args:
            l2_contexts: L2 weekly summary contexts for the month
            l1_by_week: Dict mapping week_str -> L1 daily summaries
            time_bucket: Month string like "2026-02"

        Returns:
            Summary text or None
        """
        formatted_text = self._format_monthly_hierarchical(l2_contexts, l1_by_week)
        total_tokens = _estimate_tokens(formatted_text) + _PROMPT_OVERHEAD_TOKENS

        if total_tokens <= _MAX_INPUT_TOKENS:
            return await self._call_llm_for_summary(
                formatted_content=formatted_text,
                level=3,
                time_bucket=time_bucket,
            )

        # Overflow — batch by week-groups (2 weeks per batch)
        logger.info(
            f"Token overflow for monthly {time_bucket}: "
            f"~{total_tokens} tokens, splitting by week-groups"
        )
        l2_by_week: Dict[str, ProcessedContext] = {}
        for ctx in l2_contexts:
            tb = ctx.properties.time_bucket
            if tb:
                l2_by_week[tb] = ctx

        all_weeks = sorted(set(list(l2_by_week.keys()) + list(l1_by_week.keys())))
        week_groups: List[List[str]] = []
        for i in range(0, len(all_weeks), 2):
            week_groups.append(all_weeks[i : i + 2])

        sub_summaries = []
        for idx, week_group in enumerate(week_groups):
            group_l2 = [l2_by_week[w] for w in week_group if w in l2_by_week]
            group_l1 = {w: l1_by_week[w] for w in week_group if w in l1_by_week}
            group_text = self._format_monthly_hierarchical(group_l2, group_l1)
            batch_info = f"{idx + 1}/{len(week_groups)}"
            sub_summary = await self._call_llm_for_summary(
                formatted_content=group_text,
                level=3,
                time_bucket=time_bucket,
                is_partial=True,
                batch_info=batch_info,
            )
            if sub_summary:
                sub_summaries.append(sub_summary)

        if not sub_summaries:
            return None
        if len(sub_summaries) == 1:
            return sub_summaries[0]

        merged_text = "\n\n---\n\n".join(
            f"[Part {i + 1}]\n{s}" for i, s in enumerate(sub_summaries)
        )
        return await self._call_llm_for_merge(
            sub_summaries_text=merged_text,
            level=3,
            time_bucket=time_bucket,
        )

    # ── Private methods for each hierarchy level ──

    async def _generate_daily_summary(self, user_id: str, date_str: str) -> Optional[ProcessedContext]:
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
        existing = await storage.search_hierarchy(
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
        l0_dict = await storage.get_all_processed_contexts(
            context_types=[ContextType.EVENT.value],
            limit=500,
            filter=l0_filters,
            user_id=user_id,
        )
        contexts = l0_dict.get(ContextType.EVENT.value, [])

        if not contexts:
            logger.info(f"No L0 events found for user={user_id}, date={date_str}")
            return None
        children_ids = [ctx.id for ctx in contexts]

        # Generate summary via LLM with batch overflow handling
        # 通过 LLM 生成摘要，支持 batch 溢出处理
        summary_text = await self._batch_summarize_and_merge(
            contexts=contexts,
            level=1,
            time_bucket=date_str,
            format_fn=self._format_l0_events,
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for daily {date_str}")
            return None

        # Store the summary
        # 存储摘要
        return await self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=1,
            time_bucket=date_str,
            children_ids=children_ids,
        )

    async def _generate_weekly_summary(self, user_id: str, week_str: str) -> Optional[ProcessedContext]:
        """
        Generate a weekly summary (Level 2) from Level 1 daily summaries + Level 0 raw events.
        从 Level 1 日摘要 + Level 0 原始事件生成每周摘要 (Level 2)。

        Higher-level summaries now include content from both immediate children (L1) and
        grandchildren (L0) for richer context.

        Args:
            user_id: User identifier
            week_str: ISO week string, e.g. "2026-W08"

        Returns:
            ProcessedContext of the generated weekly summary, or None if no data found
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot generate weekly summary")
            return None

        # Check if weekly summary already exists
        # 检查周摘要是否已存在
        existing = await storage.search_hierarchy(
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

        # Parse week string to get date range
        # 解析周字符串以获取日期范围
        year, week_num = week_str.split("-W")
        year = int(year)
        week_num = int(week_num)
        week_start = datetime.date.fromisocalendar(year, week_num, 1)
        week_end = week_start + datetime.timedelta(days=6)

        # Query Level 1 daily summaries for the week
        # 查询该周的 Level 1 日摘要
        l1_results = await storage.search_hierarchy(
            context_type=ContextType.EVENT.value,
            hierarchy_level=1,
            time_bucket_start=week_start.isoformat(),
            time_bucket_end=week_end.isoformat(),
            user_id=user_id,
            top_k=7,
        )

        l1_contexts = [ctx for ctx, _score in l1_results] if l1_results else []

        # Also fetch L0 events for the entire week to enrich the weekly summary
        # 同时获取整周的 L0 事件以丰富周摘要
        week_start_ts = int(
            datetime.datetime(
                week_start.year,
                week_start.month,
                week_start.day,
                tzinfo=datetime.timezone.utc,
            ).timestamp()
        )
        week_end_ts = int(
            datetime.datetime(
                week_end.year,
                week_end.month,
                week_end.day,
                23,
                59,
                59,
                tzinfo=datetime.timezone.utc,
            ).timestamp()
        )

        l0_filters = {
            "event_time_ts": {"$gte": week_start_ts, "$lte": week_end_ts},
            "hierarchy_level": {"$gte": 0, "$lte": 0},
        }
        l0_dict = await storage.get_all_processed_contexts(
            context_types=[ContextType.EVENT.value],
            limit=500,
            filter=l0_filters,
            user_id=user_id,
        )
        l0_events = l0_dict.get(ContextType.EVENT.value, [])

        # Group L0 events by day
        # 按天分组 L0 事件
        l0_events_by_day: Dict[str, List[ProcessedContext]] = {}
        for evt in l0_events:
            if evt.properties.event_time:
                day_key = evt.properties.event_time.strftime("%Y-%m-%d")
                l0_events_by_day.setdefault(day_key, []).append(evt)

        if not l1_contexts and not l0_events:
            logger.info(f"No L1/L0 data found for user={user_id}, week={week_str}")
            return None

        # Collect children_ids from both L1 and L0
        children_ids = [ctx.id for ctx in l1_contexts]
        for evts in l0_events_by_day.values():
            children_ids.extend(evt.id for evt in evts)

        # Generate summary with hierarchical content and overflow handling
        # 使用分层内容和溢出处理生成摘要
        summary_text = await self._batch_summarize_weekly(
            l1_contexts=l1_contexts,
            l0_events_by_day=l0_events_by_day,
            time_bucket=week_str,
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for weekly {week_str}")
            return None

        return await self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=2,
            time_bucket=week_str,
            children_ids=children_ids,
        )

    async def _generate_monthly_summary(self, user_id: str, month_str: str) -> Optional[ProcessedContext]:
        """
        Generate a monthly summary (Level 3) from Level 2 weekly summaries + Level 1 daily summaries.
        从 Level 2 周摘要 + Level 1 日摘要生成月摘要 (Level 3)。

        Higher-level summaries now include content from both immediate children (L2) and
        grandchildren (L1) for richer context.

        Args:
            user_id: User identifier
            month_str: Month string, e.g. "2026-02"

        Returns:
            ProcessedContext of the generated monthly summary, or None if no data found
        """
        storage = get_storage()
        if not storage:
            logger.error("Storage not available, cannot generate monthly summary")
            return None

        # Check if monthly summary already exists
        # 检查月摘要是否已存在
        existing = await storage.search_hierarchy(
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
        year_str, month_num_str = month_str.split("-")
        year = int(year_str)
        month_num = int(month_num_str)
        first_day = datetime.date(year, month_num, 1)
        if month_num == 12:
            last_day = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            last_day = datetime.date(year, month_num + 1, 1) - datetime.timedelta(days=1)

        # Collect all ISO week identifiers for the month
        # 收集该月所有 ISO 周标识符
        weeks_in_month = set()
        current = first_day
        while current <= last_day:
            iso_year, iso_week, _ = current.isocalendar()
            weeks_in_month.add(f"{iso_year}-W{iso_week:02d}")
            current += datetime.timedelta(days=1)

        # Query Level 2 weekly summaries and Level 1 daily summaries for each week
        # 查询该月每周的 Level 2 周摘要和 Level 1 日摘要
        all_weekly_contexts: List[ProcessedContext] = []
        l1_summaries_by_week: Dict[str, List[ProcessedContext]] = {}

        for wk in sorted(weeks_in_month):
            # Fetch L2 weekly summary
            wk_results = await storage.search_hierarchy(
                context_type=ContextType.EVENT.value,
                hierarchy_level=2,
                time_bucket_start=wk,
                time_bucket_end=wk,
                user_id=user_id,
                top_k=1,
            )
            for ctx, _score in wk_results:
                all_weekly_contexts.append(ctx)

            # Fetch L1 daily summaries for this week to enrich the monthly summary
            # 获取本周的 L1 日摘要以丰富月摘要
            wk_year, wk_num = wk.split("-W")
            wk_start = datetime.date.fromisocalendar(int(wk_year), int(wk_num), 1)
            wk_end = wk_start + datetime.timedelta(days=6)

            l1_results = await storage.search_hierarchy(
                context_type=ContextType.EVENT.value,
                hierarchy_level=1,
                time_bucket_start=wk_start.isoformat(),
                time_bucket_end=wk_end.isoformat(),
                user_id=user_id,
                top_k=7,
            )
            if l1_results:
                l1_summaries_by_week[wk] = [ctx for ctx, _score in l1_results]

        if not all_weekly_contexts and not l1_summaries_by_week:
            logger.info(f"No L2/L1 data found for user={user_id}, month={month_str}")
            return None

        # Collect children_ids from both L2 and L1
        children_ids = [ctx.id for ctx in all_weekly_contexts]
        for l1_list in l1_summaries_by_week.values():
            children_ids.extend(ctx.id for ctx in l1_list)

        # Generate summary with hierarchical content and overflow handling
        # 使用分层内容和溢出处理生成摘要
        summary_text = await self._batch_summarize_monthly(
            l2_contexts=all_weekly_contexts,
            l1_by_week=l1_summaries_by_week,
            time_bucket=month_str,
        )
        if not summary_text:
            logger.warning(f"LLM returned empty summary for monthly {month_str}")
            return None

        return await self._store_summary(
            user_id=user_id,
            summary_text=summary_text,
            level=3,
            time_bucket=month_str,
            children_ids=children_ids,
        )

    async def _call_llm_for_summary(
        self,
        formatted_content: str,
        level: int,
        time_bucket: str,
        is_partial: bool = False,
        batch_info: str = "",
    ) -> Optional[str]:
        """
        Call LLM to produce a summary from pre-formatted content.
        调用 LLM 从预格式化的内容生成摘要。

        Properly uses YAML prompt structure: system message + level-specific user template.
        Falls through: {level}_summary key -> {level}_partial_summary key -> generic user key
        -> _FALLBACK_PROMPTS.

        Args:
            formatted_content: Pre-formatted text of events/summaries
            level: Target hierarchy level (1=daily, 2=weekly, 3=monthly)
            time_bucket: Time bucket identifier
            is_partial: Whether this is a partial batch sub-summary
            batch_info: Batch identifier string like "1/3"

        Returns:
            Summary text string, or None on failure
        """
        level_name = _LEVEL_NAMES.get(level, "unknown")
        prompt_group = get_prompt_group("hierarchy_summary")

        # Determine which prompt key to use
        if is_partial:
            prompt_key = f"{level_name}_partial_summary"
        else:
            prompt_key = f"{level_name}_summary"

        # Build messages: system + user
        messages = []

        # System message from prompt group
        if prompt_group and "system" in prompt_group:
            messages.append({"role": "system", "content": prompt_group["system"]})

        # User message — try prompt group first, then fallback
        template_vars = {
            "time_period": time_bucket,
            "activity_records": formatted_content,
            "batch_info": batch_info,
        }

        if prompt_group and prompt_key in prompt_group:
            user_template = prompt_group[prompt_key]
        elif prompt_group and "user" in prompt_group:
            # Generic user key from YAML
            user_template = prompt_group["user"]
        else:
            # Fallback to hardcoded prompts
            user_template = _FALLBACK_PROMPTS.get(prompt_key, "")
            if not user_template:
                user_template = _FALLBACK_PROMPTS.get(f"{level_name}_summary", "")
            if not user_template:
                logger.error(f"No prompt template found for {prompt_key}")
                return None

        try:
            user_content = user_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template variable missing for {prompt_key}: {e}, using raw format")
            user_content = user_template.replace("{time_period}", time_bucket).replace(
                "{activity_records}", formatted_content
            )

        messages.append({"role": "user", "content": user_content})

        try:
            response = await generate_with_messages(messages, enable_executor=False)
            if response:
                return response.strip()
            else:
                logger.warning(f"LLM returned empty response for {prompt_key}")
                return None
        except Exception as e:
            logger.exception(f"LLM call failed for {prompt_key}: {e}")
            return None

    async def _call_llm_for_merge(
        self,
        sub_summaries_text: str,
        level: int,
        time_bucket: str,
    ) -> Optional[str]:
        """
        Call LLM to merge multiple partial sub-summaries into one cohesive summary.
        调用 LLM 将多个部分子摘要合并为一个完整摘要。

        Looks for {level}_merge key in prompt group, falls back to hardcoded merge prompt.

        Args:
            sub_summaries_text: Formatted text of all partial sub-summaries
            level: Hierarchy level (1=daily, 2=weekly, 3=monthly)
            time_bucket: Time bucket identifier

        Returns:
            Merged summary text, or None on failure
        """
        level_name = _LEVEL_NAMES.get(level, "unknown")
        prompt_group = get_prompt_group("hierarchy_summary")
        merge_key = f"{level_name}_merge"

        messages = []

        # System message
        if prompt_group and "system" in prompt_group:
            messages.append({"role": "system", "content": prompt_group["system"]})

        # User message for merge
        template_vars = {
            "time_period": time_bucket,
            "partial_summaries": sub_summaries_text,
        }

        if prompt_group and merge_key in prompt_group:
            user_template = prompt_group[merge_key]
        else:
            user_template = _FALLBACK_PROMPTS.get(merge_key, "")
            if not user_template:
                logger.error(f"No merge prompt template found for {merge_key}")
                return None

        try:
            user_content = user_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template variable missing for {merge_key}: {e}")
            user_content = user_template.replace("{time_period}", time_bucket).replace(
                "{partial_summaries}", sub_summaries_text
            )

        messages.append({"role": "user", "content": user_content})

        try:
            response = await generate_with_messages(messages, enable_executor=False)
            if response:
                return response.strip()
            else:
                logger.warning(f"LLM returned empty response for merge {merge_key}")
                return None
        except Exception as e:
            logger.exception(f"LLM merge call failed for {merge_key}: {e}")
            return None

    async def _store_summary(
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

        # Parse LLM JSON response to extract structured fields
        # LLM 返回 JSON 格式：{title, summary, keywords, entities, importance}
        title = f"{level_name.capitalize()} Summary - {time_bucket}"
        summary_body = summary_text
        keywords = [level_name, "summary", time_bucket]
        entities: List[str] = []
        importance = 5 + level

        # Strip markdown code fences if present (```json ... ```)
        cleaned = summary_text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                title = str(parsed.get("title", title))[:100]
                summary_body = str(parsed.get("summary", summary_text))
                if parsed.get("keywords"):
                    keywords = [str(k) for k in parsed["keywords"]]
                if parsed.get("entities"):
                    entities = [str(e) for e in parsed["entities"]]
                if parsed.get("importance") is not None:
                    importance = int(parsed["importance"])
        except (json.JSONDecodeError, TypeError, ValueError):
            # Not JSON — use heuristic title extraction
            lines = summary_text.strip().split("\n")
            title = lines[0][:100] if lines else title
            for prefix in ["1. ", "1.", "Title: ", "Title:", "# "]:
                if title.startswith(prefix):
                    title = title[len(prefix) :].strip()
                    break

        # Build extracted data
        # 构建提取数据
        extracted_data = ExtractedData(
            title=title,
            summary=summary_body,
            keywords=keywords,
            entities=entities,
            context_type=ContextType.EVENT,
            confidence=80,
            importance=importance,
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
            is_happened=True,
            user_id=user_id,
            hierarchy_level=level,
            time_bucket=time_bucket,
            parent_id=None,
            children_ids=children_ids,
        )

        # Build vectorize object for semantic search (use parsed summary, not raw JSON)
        # 构建向量化对象以支持语义搜索（使用解析后的摘要，而非原始 JSON）
        vectorize = Vectorize(
            content_format=ContentFormat.TEXT,
            text=summary_body,
        )

        # Generate embedding vector
        # 生成嵌入向量
        try:
            await do_vectorize(vectorize)
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
            result = await storage.upsert_processed_context(context)
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
        Async handler function compatible with TaskScheduler.
        与 TaskScheduler 兼容的异步处理函数。
    """
    task = HierarchySummaryTask()

    async def handler(
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

        result = await task.execute(context)
        return result.success

    return handler
