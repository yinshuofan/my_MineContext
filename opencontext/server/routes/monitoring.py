# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Server component: monitoring routes - Monitoring API endpoints
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from opencontext.monitoring import get_monitor
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.opencontext import OpenContext
from opencontext.server.utils import get_context_lab

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/overview")
async def get_system_overview(
    opencontext: OpenContext = Depends(get_context_lab), _auth: str = auth_dependency
):
    """
    Get system monitoring overview
    """
    try:
        monitor = get_monitor()
        overview = await monitor.get_system_overview()
        return {"success": True, "data": overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system overview: {str(e)}")


@router.get("/context-types")
async def get_context_type_stats(
    force_refresh: bool = Query(False, description="Whether to force refresh cache"),
    opencontext: OpenContext = Depends(get_context_lab),
    _auth: str = auth_dependency,
):
    """
    Get candidate count statistics for each context_type
    """
    try:
        monitor = get_monitor()
        stats = await monitor.get_context_type_stats(force_refresh=force_refresh)
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get context type statistics: {str(e)}"
        )


@router.get("/token-usage")
async def get_token_usage_summary(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """
    Get model token consumption details
    """
    try:
        monitor = get_monitor()
        summary = await monitor.get_token_usage_summary(hours=hours)
        return {"success": True, "data": summary}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token usage statistics: {str(e)}"
        )


@router.get("/processing")
async def get_processing_metrics(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """
    Get processor performance metrics
    """
    try:
        monitor = get_monitor()
        metrics = monitor.get_processing_summary(hours=hours)
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get processing performance metrics: {str(e)}"
        )


@router.get("/stage-timing")
async def get_stage_timing_metrics(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """
    Get stage timing metrics (LLM API calls and processing stages)
    """
    try:
        monitor = get_monitor()
        metrics = await monitor.get_stage_timing_summary(hours=hours)
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stage timing metrics: {str(e)}")


@router.get("/data-stats")
async def get_data_stats(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """
    Get data statistics (documents, contexts)
    """
    try:
        monitor = get_monitor()
        stats = await monitor.get_data_stats_summary(hours=hours)
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data statistics: {str(e)}")


@router.get("/data-stats-trend")
async def get_data_stats_trend(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """
    Get data statistics trend with time series (documents, contexts over time)
    """
    try:
        monitor = get_monitor()
        trend = await monitor.get_data_stats_trend(hours=hours)
        return {"success": True, "data": trend}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get data statistics trend: {str(e)}"
        )


@router.get("/data-stats-range")
async def get_data_stats_by_range(
    start_time: datetime = Query(..., description="Start time of the query range (ISO format)"),
    end_time: datetime = Query(..., description="End time of the query range (ISO format)"),
    _auth: str = auth_dependency,
):
    """
    Get data statistics by custom time range (documents, contexts)
    """
    try:
        # Validate time range
        if start_time >= end_time:
            raise HTTPException(status_code=400, detail="start_time must be earlier than end_time")

        monitor = get_monitor()
        stats = await monitor.get_data_stats_by_range(start_time=start_time, end_time=end_time)
        return {"success": True, "data": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get data statistics by range: {str(e)}"
        )


@router.post("/refresh-context-stats")
async def refresh_context_type_stats(
    opencontext: OpenContext = Depends(get_context_lab), _auth: str = auth_dependency
):
    """
    Refresh context type statistics cache
    """
    try:
        monitor = get_monitor()
        stats = await monitor.get_context_type_stats(force_refresh=True)
        return {"success": True, "data": stats, "message": "Statistics data refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh statistics data: {str(e)}")


@router.get("/health")
async def monitoring_health(_auth: str = auth_dependency):
    """
    Monitoring system health check
    """
    try:
        monitor = get_monitor()
        uptime_seconds = (
            int((datetime.now() - monitor._start_time).total_seconds())
            if monitor._start_time
            else 0
        )
        return {"success": True, "data": {"monitor_active": True, "uptime_seconds": uptime_seconds}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/processing-errors")
async def get_processing_errors(
    hours: int = Query(1, ge=1, le=24, description="Statistics time range (hours)"),
    top: int = Query(5, ge=1, le=20, description="Number of top errors to return"),
    _auth: str = auth_dependency,
):
    """
    Get processing errors Top N
    """
    try:
        monitor = get_monitor()
        errors = monitor.get_processing_errors(hours=hours, top_n=top)
        return {"success": True, "data": errors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing errors: {str(e)}")


@router.get("/scheduler")
async def get_scheduler_metrics(
    hours: int = Query(24, ge=1, le=168, description="Statistics time range (hours)"),
    _auth: str = auth_dependency,
):
    """Get scheduler task execution summary"""
    try:
        monitor = get_monitor()
        summary = monitor.get_scheduler_summary(hours=hours)
        return {"success": True, "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler metrics: {str(e)}")


@router.get("/scheduler/queues")
async def get_scheduler_queue_depths(
    _auth: str = auth_dependency,
):
    """Get current queue depths for all scheduler task types (real-time from Redis)"""
    try:
        from opencontext.scheduler import get_scheduler

        scheduler = get_scheduler()
        if not scheduler:
            return {"success": True, "data": {"queues": {}, "message": "Scheduler not initialized"}}

        queues = await scheduler.get_queue_depths()
        return {"success": True, "data": {"queues": queues}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue depths: {str(e)}")


@router.get("/scheduler/failures")
async def get_scheduler_failures(
    hours: int = Query(1, ge=1, le=24, description="Time range (hours)"),
    _auth: str = auth_dependency,
):
    """Get scheduler failure rates and recent errors (for external alerting)"""
    try:
        monitor = get_monitor()
        summary = monitor.get_scheduler_summary(hours=hours)

        failure_data = {
            "time_range_hours": hours,
            "total_executions": summary["total_executions"],
            "by_task_type": {},
            "recent_failures": summary["recent_failures"],
        }

        for task_type, stats in summary["by_task_type"].items():
            count = stats["count"]
            failures = stats["failure_count"]
            failure_data["by_task_type"][task_type] = {
                "total": count,
                "failures": failures,
                "failure_rate": round(failures / count, 4) if count > 0 else 0,
            }

        return {"success": True, "data": failure_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler failures: {str(e)}")


@router.post("/trigger-task")
async def trigger_task(
    task_type: str = Query(..., description="Task type to trigger (e.g. hierarchy_summary)"),
    user_id: str = Query(..., description="User ID"),
    device_id: str = Query("default", description="Device ID"),
    agent_id: str = Query("default", description="Agent ID"),
    level: str = Query(
        "auto", description="Summary level: auto (full execute), daily, weekly, monthly"
    ),
    target: str = Query(
        None, description="Target period: date (2026-03-01), week (2026-W08), month (2026-02)"
    ),
    _auth: str = auth_dependency,
):
    """Manually trigger a periodic task for testing (bypasses scheduler delay)"""
    try:
        if task_type == "hierarchy_summary":
            from opencontext.periodic_task.hierarchy_summary import HierarchySummaryTask

            task = HierarchySummaryTask()

            if level == "auto":
                from opencontext.periodic_task import create_hierarchy_handler

                handler = create_hierarchy_handler()
                success = await handler(user_id=user_id, device_id=device_id, agent_id=agent_id)
                return {
                    "success": success,
                    "message": f"hierarchy_summary {'completed' if success else 'failed'}",
                }

            if not target:
                raise HTTPException(
                    status_code=400, detail="target parameter required for specific level"
                )

            result = None
            if level == "daily":
                result = await task._generate_daily_summary(user_id, target)
            elif level == "weekly":
                result = await task._generate_weekly_summary(user_id, target)
            elif level == "monthly":
                result = await task._generate_monthly_summary(user_id, target)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown level: {level}")

            if result:
                return {
                    "success": True,
                    "message": f"{level} summary generated for {target}",
                    "data": {
                        "id": result.id,
                        "title": result.extracted_data.title if result.extracted_data else None,
                        "summary": result.extracted_data.summary if result.extracted_data else None,
                        "hierarchy_level": result.properties.hierarchy_level,
                        "time_bucket": result.properties.time_bucket,
                    },
                }
            else:
                return {"success": False, "message": f"No summary generated for {level} {target}"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")
