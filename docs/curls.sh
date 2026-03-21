#!/usr/bin/env bash
# =============================================================================
# MineContext API - cURL Collection
# =============================================================================
# All API endpoints as cURL commands, organized by category.
# Used for Apifox import and as a quick API reference.
#
# Base URL: http://localhost:1733
# Auth: X-API-Key header (disabled by default, uncomment if enabled)
# =============================================================================


# ============================================================================
# 1. Health & Auth Status
# ============================================================================

# Basic Health Check
curl -X GET http://localhost:1733/health

# Detailed Health Check (with component status)
# Response includes: config, storage, llm, document_db, redis, scheduler
# scheduler field (local):  {"initialized": true, "running": true, "in_flight_tasks": 0, "registered_handlers": ["hierarchy_summary", ...]}
# scheduler field (remote): {"initialized": true, "running": true, "in_flight_tasks": 0, "registered_handlers": [...], "remote": true, "last_heartbeat": "2026-03-04T..."}
curl -X GET http://localhost:1733/api/health
# -H "X-API-Key: your-api-key"

# Auth Status
curl -X GET http://localhost:1733/api/auth/status

# Readiness Probe
curl -X GET http://localhost:1733/api/ready


# ============================================================================
# 2. Push - Chat
# ============================================================================
# Messages are persisted to chat_batches, then dispatched to processors in background.
# Response includes batch_id for tracking.
#
# processors parameter (default: ["user_memory"]) controls which processors run.
# NOTE: process_mode and flush_immediately have been removed (breaking change).

# Push Chat Messages (default processors: ["user_memory"])
# Response: {"code": 0, "status": 200, "message": "Chat messages submitted for processing",
#            "data": {"batch_id": "...", "message_count": 1}}
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "I prefer using Python for data analysis"}]
      }
    ],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Push Chat Messages (multiple messages)
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Meeting with John about Q3 budget review tomorrow at 3 PM"}]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Got it, I have noted the meeting with John for Q3 budget review."}]
      }
    ],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Push Chat Messages (explicit processors)
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Can you help me with the project timeline?"}]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Sure! Let me look at the project schedule."}]
      }
    ],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default",
    "processors": ["user_memory"]
  }'
# -H "X-API-Key: your-api-key"

# Push Chat (dual processor - user + agent memory)
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ],
    "user_id": "user_001",
    "agent_id": "assistant_01",
    "processors": ["user_memory", "agent_memory"]
  }'
# -H "X-API-Key: your-api-key"

# Push Multimodal Chat (text + image + video, OpenAI multimodal format)
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Here is the photo from our team outing and a short video clip"},
          {"type": "image_url", "image_url": {"url": "https://example.com/photos/team-outing.jpg"}},
          {"type": "video_url", "video_url": {"url": "https://example.com/videos/team-clip.mp4"}}
        ]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Nice! I have saved the team outing photo and video."}]
      }
    ],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Push Multimodal Chat (image via base64)
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this screenshot?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
      }
    ],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 3. Push - Document
# ============================================================================

# Push Document (JSON - file path or base64)
curl -X POST http://localhost:1733/api/push/document \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "user_id": "user_001",
    "device_id": "default",
    "metadata": {"category": "technical"}
  }'
# -H "X-API-Key: your-api-key"

# Push Document (multipart upload)
curl -X POST http://localhost:1733/api/push/document/upload \
  -F "file=@/path/to/document.pdf" \
  -F "user_id=user_001" \
  -F "device_id=default"
# -H "X-API-Key: your-api-key"


# ============================================================================
# 4. Search
# ============================================================================

# Event Search (semantic query with drill-up)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "project timeline and budget"}],
    "top_k": 20,
    "score_threshold": 0.5,
    "drill_up": true,
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"
#
# Response (tree structure — ancestors as parent nodes, search hits as children):
# {
#   "success": true,
#   "events": [
#     {
#       "id": "f4b61534-...",
#       "hierarchy_level": 1,
#       "refs": {},
#       "title": "Daily Summary",
#       "summary": "Daily summary text...",
#       "event_time_start": "2026-03-04T00:00:00+00:00",
#       "event_time_end": "2026-03-04T23:59:59+00:00",
#       "create_time": "2026-03-04T09:29:32.894067+00:00",
#       "is_search_hit": false,
#       "children": [
#         {
#           "id": "05278626-88c4-4f85-8eec-e69ac143914c",
#           "hierarchy_level": 0,
#           "refs": {"daily_summary": ["f4b61534-..."]},
#           "title": "Event title",
#           "summary": "Event summary text",
#           "event_time_start": "2026-03-04T09:17:26.626423+00:00",
#           "event_time_end": "2026-03-04T09:17:26.626423+00:00",
#           "create_time": "2026-03-04T09:17:26.626077",
#           "is_search_hit": true,
#           "children": [],
#           "content": "id: ...\ntitle: ...\nsummary: ...\n...",
#           "keywords": ["keyword1", "keyword2"],
#           "entities": ["entity1", "entity2"],
#           "score": 0.855,
#           "metadata": {"source": "default", "todo_id": "default"}
#         }
#       ]
#     }
#   ],
#   "metadata": {
#     "query": "project timeline and budget",
#     "total_results": 1,
#     "search_time_ms": 556.02
#   }
# }

# Event Search (by event IDs)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "event_ids": ["evt_abc123", "evt_def456"],
    "drill_up": true,
    "user_id": "user_001"
  }'
# -H "X-API-Key: your-api-key"

# Event Search (filters only — time range + hierarchy levels)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "time_range": {"start": 1709251200, "end": 1709856000},
    "hierarchy_levels": [0, 1],
    "drill_up": true,
    "top_k": 20,
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Event Search (no drill-up, flat results)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "meeting with John"}],
    "hierarchy_levels": [0],
    "drill_up": false,
    "top_k": 10,
    "user_id": "user_001"
  }'
# -H "X-API-Key: your-api-key"

# Multimodal Search (text query + image, finds visually and semantically similar events)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      {"type": "text", "text": "team outing photos"},
      {"type": "image_url", "image_url": {"url": "https://example.com/photos/reference-image.jpg"}}
    ],
    "top_k": 10,
    "score_threshold": 0.5,
    "drill_up": true,
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"
#
# Response (drill_up=false, flat root nodes):
# {
#   "success": true,
#   "events": [
#     {
#       "id": "...",
#       "hierarchy_level": 0,
#       "refs": {"daily_summary": ["..."]},
#       "title": "...",
#       "summary": "...",
#       "event_time_start": "...",
#       "event_time_end": "...",
#       "create_time": "...",
#       "is_search_hit": true,
#       "children": [],
#       "content": "...",
#       "keywords": [...],
#       "entities": [...],
#       "score": 0.812,
#       "metadata": {...}
#     }
#   ],
#   "metadata": {
#     "query": "meeting with John",
#     "total_results": 1,
#     "search_time_ms": 430.0
#   }
# }

# Search agent events
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "meeting notes"}],
    "memory_owner": "agent",
    "user_id": "user_001",
    "agent_id": "assistant_01"
  }'
# -H "X-API-Key: your-api-key"

# Direct Vector Search
curl -X POST http://localhost:1733/api/vector_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "microservice architecture patterns",
    "top_k": 10,
    "score_threshold": 0.5,
    "context_types": ["knowledge", "document"],
    "filters": {},
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 5. Memory Cache
# ============================================================================

# Get Memory Cache Snapshot (default sections: profile, events, accessed)
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&device_id=default&agent_id=default&recent_days=7&max_recent_events_today=30&max_accessed=20&force_refresh=false"
# -H "X-API-Key: your-api-key"

# Get only profile
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&include=profile"
# -H "X-API-Key: your-api-key"

# Get profile + events (no recently accessed)
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&include=profile,events"
# -H "X-API-Key: your-api-key"

# Get agent memory cache
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&agent_id=assistant_01&memory_owner=agent"
# -H "X-API-Key: your-api-key"

# Invalidate Memory Cache
curl -X DELETE "http://localhost:1733/api/memory-cache?user_id=user_001&device_id=default&agent_id=default"
# -H "X-API-Key: your-api-key"


# ============================================================
# Chat Batches (Debug)
# ============================================================

# List chat batches
curl -X GET "http://localhost:1733/api/chat-batches?user_id=user_001&limit=10"

# Get batch detail with messages
curl -X GET "http://localhost:1733/api/chat-batches/{batch_id}"

# Get contexts produced by a batch
curl -X GET "http://localhost:1733/api/chat-batches/{batch_id}/contexts"


# ============================================================================
# 6. Contexts
# ============================================================================

# Get Context Detail (HTML)
curl -X POST http://localhost:1733/contexts/detail \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ctx_abc123",
    "context_type": "knowledge"
  }'
# -H "X-API-Key: your-api-key"

# Get Context by ID (JSON)
curl -X GET "http://localhost:1733/api/contexts/ctx_abc123?context_type=knowledge"
# -H "X-API-Key: your-api-key"

# List Context Types
curl -X GET http://localhost:1733/api/context_types
# -H "X-API-Key: your-api-key"

# Delete Context
curl -X POST http://localhost:1733/contexts/delete \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ctx_abc123",
    "context_type": "knowledge"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 7. Agent Chat
# ============================================================================

# Agent Chat (synchronous)
curl -X POST http://localhost:1733/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What meetings do I have this week?",
    "context": {},
    "session_id": "session_001",
    "user_id": "user_001",
    "conversation_id": 1
  }'
# -H "X-API-Key: your-api-key"

# Agent Chat (streaming via SSE)
curl -X POST http://localhost:1733/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Summarize my recent activities",
    "context": {},
    "session_id": "session_001",
    "user_id": "user_001",
    "conversation_id": 1
  }'
# -H "X-API-Key: your-api-key"

# Resume Workflow
curl -X POST http://localhost:1733/api/agent/resume/wf_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "wf_abc123",
    "user_input": "Yes, please proceed"
  }'
# -H "X-API-Key: your-api-key"

# Get Workflow State
curl -X GET http://localhost:1733/api/agent/state/wf_abc123
# -H "X-API-Key: your-api-key"

# Cancel Workflow
curl -X DELETE http://localhost:1733/api/agent/cancel/wf_abc123
# -H "X-API-Key: your-api-key"

# Agent Test
curl -X GET http://localhost:1733/api/agent/test
# -H "X-API-Key: your-api-key"


# ============================================================================
# 8. Conversations
# ============================================================================

# Create Conversation
curl -X POST http://localhost:1733/api/agent/chat/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "page_name": "home",
    "document_id": "doc_001"
  }'
# -H "X-API-Key: your-api-key"

# List Conversations
curl -X GET "http://localhost:1733/api/agent/chat/conversations/list?limit=20&offset=0&page_name=home&status=active"
# -H "X-API-Key: your-api-key"

# Get Conversation Detail
curl -X GET http://localhost:1733/api/agent/chat/conversations/1
# -H "X-API-Key: your-api-key"

# Update Conversation Title
curl -X PATCH http://localhost:1733/api/agent/chat/conversations/1/update \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Weekly Planning Discussion"
  }'
# -H "X-API-Key: your-api-key"

# Delete Conversation
curl -X DELETE http://localhost:1733/api/agent/chat/conversations/1/update
# -H "X-API-Key: your-api-key"


# ============================================================================
# 9. Messages
# ============================================================================

# Create Message
curl -X POST http://localhost:1733/api/agent/chat/message/msg_001/create \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": 1,
    "role": "user",
    "content": "What is the status of the Q3 project?",
    "is_complete": true,
    "token_count": 12
  }'
# -H "X-API-Key: your-api-key"

# Create Streaming Message
curl -X POST http://localhost:1733/api/agent/chat/message/stream/msg_002/create \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": 1,
    "role": "assistant"
  }'
# -H "X-API-Key: your-api-key"

# Update Message Content
curl -X POST http://localhost:1733/api/agent/chat/message/msg_002/update \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": 2,
    "new_content": "The Q3 project is on track with 80% completion.",
    "is_complete": true,
    "token_count": 15
  }'
# -H "X-API-Key: your-api-key"

# Append Message Content
curl -X POST http://localhost:1733/api/agent/chat/message/msg_002/append \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": 2,
    "content_chunk": " The remaining tasks include testing and documentation.",
    "token_count": 8
  }'
# -H "X-API-Key: your-api-key"

# Mark Message as Finished
curl -X POST http://localhost:1733/api/agent/chat/message/msg_002/finished
# -H "X-API-Key: your-api-key"

# List Conversation Messages
curl -X GET http://localhost:1733/api/agent/chat/conversations/1/messages
# -H "X-API-Key: your-api-key"

# Interrupt Message
curl -X POST http://localhost:1733/api/agent/chat/messages/2/interrupt
# -H "X-API-Key: your-api-key"


# ============================================================================
# 10. Documents & WebLinks
# ============================================================================

# Upload Document (via file path)
curl -X POST http://localhost:1733/api/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/report.pdf"
  }'
# -H "X-API-Key: your-api-key"

# Upload WebLink
curl -X POST http://localhost:1733/api/weblinks/upload \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article/best-practices",
    "filename_hint": "best-practices.html"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 11. Vaults (Document Management)
# ============================================================================

# List Vault Documents
curl -X GET "http://localhost:1733/api/vaults/list?limit=50&offset=0"
# -H "X-API-Key: your-api-key"

# Create Vault Document
curl -X POST http://localhost:1733/api/vaults/create \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Architecture Decision Record",
    "content": "# ADR-001: Use Event-Driven Architecture\n\n## Context\nWe need a scalable communication pattern between services.\n\n## Decision\nAdopt event-driven architecture using Redis pub/sub.",
    "summary": "ADR for event-driven architecture adoption",
    "tags": "architecture,adr,decision",
    "document_type": "note"
  }'
# -H "X-API-Key: your-api-key"

# Get Vault Document
curl -X GET http://localhost:1733/api/vaults/1
# -H "X-API-Key: your-api-key"

# Update Vault Document
curl -X POST http://localhost:1733/api/vaults/1 \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Architecture Decision Record (Updated)",
    "content": "# ADR-001: Use Event-Driven Architecture\n\n## Context\nUpdated content here...",
    "summary": "Updated ADR for event-driven architecture",
    "tags": "architecture,adr,decision,updated",
    "document_type": "note"
  }'
# -H "X-API-Key: your-api-key"

# Delete Vault Document
curl -X DELETE http://localhost:1733/api/vaults/1
# -H "X-API-Key: your-api-key"

# Get Vault Document Context Status
curl -X GET http://localhost:1733/api/vaults/1/context
# -H "X-API-Key: your-api-key"


# ============================================================================
# 12. Monitoring
# ============================================================================

# System Overview
curl -X GET http://localhost:1733/api/monitoring/overview
# -H "X-API-Key: your-api-key"

# Context Type Statistics
curl -X GET "http://localhost:1733/api/monitoring/context-types?force_refresh=false"
# -H "X-API-Key: your-api-key"

# Token Usage
curl -X GET "http://localhost:1733/api/monitoring/token-usage?hours=24"
# -H "X-API-Key: your-api-key"

# Processing Metrics
curl -X GET "http://localhost:1733/api/monitoring/processing?hours=24"
# -H "X-API-Key: your-api-key"

# Stage Timing Metrics
curl -X GET "http://localhost:1733/api/monitoring/stage-timing?hours=24"
# -H "X-API-Key: your-api-key"

# Data Statistics
curl -X GET "http://localhost:1733/api/monitoring/data-stats?hours=24"
# -H "X-API-Key: your-api-key"

# Data Statistics Trend
curl -X GET "http://localhost:1733/api/monitoring/data-stats-trend?hours=24"
# -H "X-API-Key: your-api-key"

# Data Statistics by Time Range
curl -X GET "http://localhost:1733/api/monitoring/data-stats-range?start_time=2026-02-23T00:00:00Z&end_time=2026-02-24T23:59:59Z"
# -H "X-API-Key: your-api-key"

# Refresh Context Statistics Cache
curl -X POST http://localhost:1733/api/monitoring/refresh-context-stats
# -H "X-API-Key: your-api-key"

# Monitoring Health
curl -X GET http://localhost:1733/api/monitoring/health
# -H "X-API-Key: your-api-key"

# Processing Errors
curl -X GET "http://localhost:1733/api/monitoring/processing-errors?hours=1&top=5"
# -H "X-API-Key: your-api-key"


# === Scheduler Monitoring ===

# Get scheduler execution summary (last 24 hours)
curl -X GET "http://localhost:1733/api/monitoring/scheduler?hours=24"
# -H "X-API-Key: your-api-key"

# Get real-time queue depths for all task types
curl -X GET "http://localhost:1733/api/monitoring/scheduler/queues"
# -H "X-API-Key: your-api-key"

# Get scheduler failure rates and recent errors (last 1 hour, for alerting)
curl -X GET "http://localhost:1733/api/monitoring/scheduler/failures?hours=1"
# -H "X-API-Key: your-api-key"


# === Manual Task Trigger (for testing) ===

# Trigger hierarchy summary — auto mode (full execute: L1 + L2 + L3)
curl -X POST "http://localhost:1733/api/monitoring/trigger-task?task_type=hierarchy_summary&user_id=user_001"
# -H "X-API-Key: your-api-key"

# Trigger daily summary for a specific date
curl -X POST "http://localhost:1733/api/monitoring/trigger-task?task_type=hierarchy_summary&user_id=user_001&level=daily&target=2026-03-01"
# -H "X-API-Key: your-api-key"

# Trigger weekly summary for a specific ISO week
curl -X POST "http://localhost:1733/api/monitoring/trigger-task?task_type=hierarchy_summary&user_id=user_001&level=weekly&target=2026-W09"
# -H "X-API-Key: your-api-key"

# Trigger monthly summary for a specific month
curl -X POST "http://localhost:1733/api/monitoring/trigger-task?task_type=hierarchy_summary&user_id=user_001&level=monthly&target=2026-03"
# -H "X-API-Key: your-api-key"


# ============================================================================
# 13. Settings - Model
# ============================================================================

# Get Model Settings (returns 3 model configs: llm, vlm_model, embedding_model)
curl -X GET http://localhost:1733/api/model_settings/get
# -H "X-API-Key: your-api-key"

# Update Model Settings (partial — only non-null sections are saved)
curl -X POST http://localhost:1733/api/model_settings/update \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {
      "provider": "openai",
      "model": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-your-api-key",
      "max_concurrent": 30
    },
    "vlm_model": {
      "provider": "openai",
      "model": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-your-api-key",
      "max_concurrent": 30
    },
    "embedding_model": {
      "provider": "openai",
      "model": "text-embedding-3-large",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-your-api-key",
      "max_concurrent": 60,
      "output_dim": 2048
    }
  }'
# -H "X-API-Key: your-api-key"

# Validate Model Settings (any combination of 3 models, without saving)
curl -X POST http://localhost:1733/api/model_settings/validate \
  -H "Content-Type: application/json" \
  -d '{
    "vlm_model": {
      "provider": "openai",
      "model": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-your-api-key"
    }
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 14. Settings - General
# ============================================================================

# Get General Settings (returns 7 sections: capture, processing, logging,
#   document_processing, scheduler, memory_cache, tools)
curl -X GET http://localhost:1733/api/settings/general
# -H "X-API-Key: your-api-key"

# Update General Settings (partial — only non-null sections are saved)
curl -X POST http://localhost:1733/api/settings/general \
  -H "Content-Type: application/json" \
  -d '{
    "capture": {
      "enabled": true,
      "text_chat": { "enabled": true, "buffer_size": 10 }
    },
    "processing": {
      "enabled": true,
      "context_merger": { "enabled": true, "similarity_threshold": 0.01 }
    },
    "document_processing": {
      "batch_size": 3,
      "dpi": 200,
      "text_threshold_per_page": 50
    },
    "scheduler": {
      "enabled": true,
      "tasks": {
        "hierarchy_summary": { "enabled": true, "trigger_mode": "user_activity", "interval": 86400 }
      }
    },
    "memory_cache": {
      "snapshot_ttl": 3600,
      "recent_days": 7,
      "max_entities": 20
    }
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 15. Settings - Prompts
# ============================================================================

# Get Prompts
curl -X GET http://localhost:1733/api/settings/prompts
# -H "X-API-Key: your-api-key"

# Update Prompts
curl -X POST http://localhost:1733/api/settings/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": {
      "context_extraction": {
        "system": "You are a context extraction assistant.",
        "user": "Extract key information from: {content}"
      }
    }
  }'
# -H "X-API-Key: your-api-key"

# Import Prompts (file upload)
curl -X POST http://localhost:1733/api/settings/prompts/import \
  -F "file=@/path/to/prompts.yaml"
# -H "X-API-Key: your-api-key"

# Export Prompts (downloads YAML file)
curl -X GET http://localhost:1733/api/settings/prompts/export \
  -o prompts_export.yaml
# -H "X-API-Key: your-api-key"

# Get Prompt Language
curl -X GET http://localhost:1733/api/settings/prompts/language
# -H "X-API-Key: your-api-key"

# Change Prompt Language
curl -X POST http://localhost:1733/api/settings/prompts/language \
  -H "Content-Type: application/json" \
  -d '{
    "language": "en"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 16. Settings - Reset
# ============================================================================

# Reset All Settings to Defaults
curl -X POST http://localhost:1733/api/settings/reset
# -H "X-API-Key: your-api-key"

# Apply Settings (restart components with latest config)
curl -X POST http://localhost:1733/api/settings/apply
# -H "X-API-Key: your-api-key"



# ============================================================================
# 17. Agents — CRUD
# ============================================================================

# Create Agent
curl -X POST http://localhost:1733/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant_01",
    "name": "Personal Assistant",
    "description": "A helpful personal assistant that remembers user preferences"
  }'
# -H "X-API-Key: your-api-key"

# List Agents
curl -X GET http://localhost:1733/api/agents
# -H "X-API-Key: your-api-key"

# Get Agent by ID
curl -X GET http://localhost:1733/api/agents/assistant_01
# -H "X-API-Key: your-api-key"

# Update Agent
curl -X PUT http://localhost:1733/api/agents/assistant_01 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Smart Assistant",
    "description": "Updated description for the assistant"
  }'
# -H "X-API-Key: your-api-key"

# Delete Agent (soft delete)
curl -X DELETE http://localhost:1733/api/agents/assistant_01
# -H "X-API-Key: your-api-key"


# ============================================================================
# 18. Agents — Base Memory (Profile & Events)
# ============================================================================
# Base memory is pre-configured agent knowledge, separate from conversation-extracted memory.
# Both base profile and base events use user_id="__base__" internally to distinguish from per-user data.

# Set Base Profile
curl -X POST http://localhost:1733/api/agents/assistant_01/base/profile \
  -H "Content-Type: application/json" \
  -d '{
    "factual_profile": "I am a personal assistant specialized in scheduling and task management.",
    "behavioral_profile": "Responds in a friendly, professional tone. Proactively suggests reminders.",
    "entities": ["calendar", "task management", "reminders"],
    "importance": 8
  }'
# -H "X-API-Key: your-api-key"

# Get Base Profile
curl -X GET http://localhost:1733/api/agents/assistant_01/base/profile
# -H "X-API-Key: your-api-key"

# Push Base Events (flat L0 events, no LLM extraction; generates embeddings directly)
curl -X POST http://localhost:1733/api/agents/assistant_01/base/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "title": "Product launch v2.0",
        "summary": "The product team launched version 2.0 with new scheduling features.",
        "event_time_start": "2026-03-15T09:00:00+08:00",
        "keywords": ["product launch", "v2.0", "scheduling"],
        "entities": ["product team"],
        "importance": 8
      },
      {
        "title": "Company policy update",
        "summary": "New remote work policy allows 3 days WFH per week.",
        "keywords": ["policy", "remote work"],
        "importance": 6
      }
    ]
  }'
# -H "X-API-Key: your-api-key"

# Push Base Events with hierarchy tree (L1 daily summary containing L0 children)
# hierarchy_level: 0=raw event, 1=daily summary, 2=weekly summary, 3=monthly summary
# Validation: hierarchy_level > 0 requires event_time_end + children;
#             children must have hierarchy_level == parent - 1;
#             parent time range must cover all children; max 500 total events.
curl -X POST http://localhost:1733/api/agents/assistant_01/base/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "title": "Standalone event",
        "summary": "A simple L0 event",
        "event_time_start": "2026-03-15T10:00:00+08:00",
        "importance": 5
      },
      {
        "title": "Daily Summary",
        "summary": "Summary of the day...",
        "event_time_start": "2026-03-15T00:00:00+08:00",
        "event_time_end": "2026-03-15T23:59:59+08:00",
        "hierarchy_level": 1,
        "children": [
          {
            "title": "Morning standup",
            "summary": "Discussed sprint progress",
            "event_time_start": "2026-03-15T09:00:00+08:00",
            "keywords": ["standup", "sprint"],
            "importance": 6
          },
          {
            "title": "Code review",
            "summary": "Reviewed auth module PR",
            "event_time_start": "2026-03-15T14:00:00+08:00",
            "keywords": ["code review", "auth"],
            "importance": 5
          }
        ]
      }
    ]
  }'
# -H "X-API-Key: your-api-key"

# List Base Events (all levels)
curl http://localhost:1733/api/agents/assistant_01/base/events
# -H "X-API-Key: your-api-key"

# List Base Events (L0 raw events only)
curl "http://localhost:1733/api/agents/assistant_01/base/events?hierarchy_level=0"
# -H "X-API-Key: your-api-key"

# List Base Events (L1 daily summaries only)
curl "http://localhost:1733/api/agents/assistant_01/base/events?hierarchy_level=1"
# -H "X-API-Key: your-api-key"

# Delete Base Event
curl -X DELETE http://localhost:1733/api/agents/assistant_01/base/events/evt_abc123
# -H "X-API-Key: your-api-key"


# ============================================================================
# 19. Agent Memory via Push Chat
# ============================================================================
# Use processors: ["agent_memory"] to extract memories from the agent's perspective.
# Requires a registered agent (see section 17).

# Push chat with agent memory extraction
curl -X POST http://localhost:1733/api/push/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to prepare the quarterly report by Friday"},
      {"role": "assistant", "content": "Got it! I will remind you about the quarterly report deadline on Friday."}
    ],
    "user_id": "user_001",
    "agent_id": "assistant_01",
    "processors": ["user_memory", "agent_memory"]
  }'
# -H "X-API-Key: your-api-key"

# Search agent memories (memory_owner="agent")
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [{"type": "text", "text": "quarterly report"}],
    "user_id": "user_001",
    "agent_id": "assistant_01",
    "memory_owner": "agent",
    "top_k": 10
  }'
# -H "X-API-Key: your-api-key"

# Get agent memory cache snapshot (memory_owner="agent")
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&agent_id=assistant_01&memory_owner=agent&include=profile,events"
# -H "X-API-Key: your-api-key"

# Invalidate agent memory cache
curl -X DELETE "http://localhost:1733/api/memory-cache?user_id=user_001&agent_id=assistant_01&memory_owner=agent"
# -H "X-API-Key: your-api-key"


# ============================================================================
# NOT REGISTERED IN MAIN ROUTER (route files exist but not included in api.py)
# ============================================================================
# The following route modules exist but are NOT registered in the main router:
# - screenshots.py  (POST /api/add_screenshot, /api/add_screenshots)
# - completions.py  (POST /api/completions/suggest, etc.)
# These endpoints will NOT be accessible unless their routers are added to api.py.
