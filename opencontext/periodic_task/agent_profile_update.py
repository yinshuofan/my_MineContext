"""Agent Profile Update Task — updates agent_profile from daily events."""


from opencontext.config.global_config import get_prompt_group
from opencontext.llm.global_vlm_client import generate_with_messages
from opencontext.models.enums import ContextType
from opencontext.periodic_task.base import BasePeriodicTask, TaskContext, TaskResult
from opencontext.scheduler.base import TriggerMode
from opencontext.storage.global_storage import get_storage
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as tz_now
from opencontext.utils.time_utils import today_start

logger = get_logger(__name__)


class AgentProfileUpdateTask(BasePeriodicTask):
    """
    Updates agent_profile based on today's events.

    Triggered via user_activity (scheduled from push endpoint when agent_memory is used).
    Skipped if agent_id is not registered or is "default".
    """

    def __init__(self, interval: int = 86400, timeout: int = 300):
        super().__init__(
            name="agent_profile_update",
            description="Update agent profile from daily interaction events",
            trigger_mode=TriggerMode.USER_ACTIVITY,
            interval=interval,
            timeout=timeout,
            task_ttl=14400,
            max_retries=2,
        )

    async def execute(self, context: TaskContext) -> TaskResult:
        user_id = context.user_id
        device_id = context.device_id or "default"
        agent_id = context.agent_id or "default"

        # Should not happen (filtered at scheduling time), but guard anyway
        if not agent_id or agent_id == "default":
            return TaskResult.ok("Skipped: no agent_id")

        storage = get_storage()
        if not storage:
            return TaskResult.fail("Storage not initialized")

        # Verify agent is still registered
        agent = await storage.get_agent(agent_id)
        if not agent:
            return TaskResult.ok(f"Skipped: agent {agent_id} not registered")

        agent_name = agent.get("name", agent_id)

        # 1. Fetch today's events using filter (same pattern as hierarchy_summary)
        day_start = today_start()
        day_start_ts = float(int(day_start.timestamp()))

        event_filters = {
            "event_time_start_ts": {"$gte": day_start_ts},
            "hierarchy_level": 0,
        }

        events_dict = await storage.get_all_processed_contexts(
            context_types=[ContextType.EVENT.value],
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            filter=event_filters,
            limit=50,
        )

        events = events_dict.get(ContextType.EVENT.value, [])

        if not events:
            return TaskResult.ok("Skipped: no events today")

        # 2. Fetch current agent_profile and base_profile
        current_profile = await storage.get_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            context_type="agent_profile",
        )
        base_profile = await storage.get_profile(
            user_id="__base__",
            device_id=device_id,
            agent_id=agent_id,
            context_type="agent_base_profile",
        )

        if not base_profile:
            return TaskResult.fail(f"No base profile for agent {agent_id}")

        base_persona = base_profile.get("factual_profile", "")
        current_persona = current_profile.get("factual_profile", "") if current_profile else ""

        # 3. Format today's events
        events_text = self._format_events(events)

        # 4. Load prompt and call LLM
        prompt_group = get_prompt_group("processing.extraction.agent_profile_update")
        if not prompt_group:
            return TaskResult.fail("agent_profile_update prompt not found")

        system_prompt = prompt_group.get("system", "")
        system_prompt = system_prompt.replace("{agent_name}", agent_name)
        system_prompt = system_prompt.replace("{base_profile}", base_persona)
        system_prompt = system_prompt.replace(
            "{current_profile}", current_persona or "(No existing profile)"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_group.get("user", "").format(
                    current_time=tz_now().isoformat(),
                    today_events=events_text,
                ),
            },
        ]

        response = await generate_with_messages(messages, enable_executor=False)
        if not response or not response.strip():
            return TaskResult.fail("LLM returned empty response")

        # 5. Store updated profile — direct upsert, no LLM merge
        # (this task already produced the final profile via LLM, merging again would be redundant)
        updated_profile = response.strip()
        success = await storage.upsert_profile(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            factual_profile=updated_profile,
            entities=None,
            importance=7,
            metadata=None,
            context_type="agent_profile",
        )

        if success:
            # Invalidate memory cache so next read picks up the new profile
            try:
                from opencontext.server.cache.memory_cache_manager import get_memory_cache_manager

                manager = get_memory_cache_manager()
                await manager.invalidate_snapshot(user_id, device_id, agent_id)
            except Exception as e:
                logger.warning(f"[agent_profile_update] Cache invalidation failed: {e}")

            logger.info(
                f"[agent_profile_update] Updated profile for user={user_id}, agent={agent_id}"
            )
            return TaskResult.ok(
                f"Profile updated for user={user_id}, agent={agent_id}",
                data={"events_count": len(events)},
            )
        else:
            return TaskResult.fail("upsert_profile returned False")

    @staticmethod
    def _format_events(events: list) -> str:
        """Format events into text for the LLM prompt."""
        lines = []
        for ctx in events:
            title = ctx.extracted_data.title or "Untitled"
            summary = ctx.extracted_data.summary or ""
            commentary = ctx.extracted_data.agent_commentary or ""
            time_str = (
                ctx.properties.event_time_start.strftime("%H:%M")
                if ctx.properties and ctx.properties.event_time_start
                else ""
            )
            lines.append(f"[{time_str}] {title}")
            if summary:
                lines.append(f"  {summary}")
            if commentary:
                lines.append(f"  My thoughts: {commentary}")
            lines.append("")
        return "\n".join(lines).strip()


def create_agent_profile_update_handler():
    """Create handler function for the scheduler."""
    task = AgentProfileUpdateTask()

    async def handler(
        user_id: str,
        device_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        context = TaskContext(
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            task_type="agent_profile_update",
        )
        if not task.validate_context(context):
            return False
        result = await task.execute(context)
        return result.success

    return handler
