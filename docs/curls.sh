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
curl -X GET http://localhost:1733/api/health
# -H "X-API-Key: your-api-key"

# Auth Status
curl -X GET http://localhost:1733/api/auth/status

# Readiness Probe
curl -X GET http://localhost:1733/api/ready


# ============================================================================
# 2. Push - Chat
# ============================================================================

# Push Single Chat Message
curl -X POST http://localhost:1733/api/push/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": [{"type": "text", "text": "I prefer using Python for data analysis"}],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default",
    "timestamp": "2026-02-24T10:00:00Z",
    "metadata": {}
  }'
# -H "X-API-Key: your-api-key"

# Push Batch Chat Messages
curl -X POST http://localhost:1733/api/push/chat/messages \
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
    "flush_immediately": false
  }'
# -H "X-API-Key: your-api-key"

# Process Chat Messages (bypass buffer)
curl -X POST http://localhost:1733/api/push/chat/process \
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

# Flush Chat Buffer
curl -X POST http://localhost:1733/api/push/chat/flush \
  -H "Content-Type: application/json" \
  -d '{
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
# 4. Push - Context
# ============================================================================

# Push Generic Context
curl -X POST http://localhost:1733/api/push/context \
  -H "Content-Type: application/json" \
  -d '{
    "source": "clipboard",
    "content_format": "text",
    "content_text": "Best practices for microservice architecture: use API gateways, implement circuit breakers, and adopt event-driven communication.",
    "create_time": "2026-02-24T10:30:00Z",
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default",
    "additional_info": {"topic": "architecture"},
    "enable_merge": true
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 5. Push - Batch
# ============================================================================

# Batch Push (multiple types in one request)
curl -X POST http://localhost:1733/api/push/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "type": "chat",
        "data": {
          "role": "user",
          "content": [{"type": "text", "text": "Remind me to review the PR"}]
        }
      },
      {
        "type": "document",
        "data": {
          "file_path": "/path/to/report.pdf"
        }
      }
    ],
    "user_id": "user_001",
    "device_id": "default"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 6. Search
# ============================================================================

# Unified Search (fast strategy)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project timeline and budget",
    "strategy": "fast",
    "top_k": 20,
    "context_types": ["document", "event", "knowledge"],
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Unified Search (intelligent strategy)
curl -X POST http://localhost:1733/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What do I know about John and our recent meetings?",
    "strategy": "intelligent",
    "top_k": 10,
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"

# Direct Vector Search
curl -X POST http://localhost:1733/api/vector_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "microservice architecture patterns",
    "top_k": 10,
    "context_types": ["knowledge", "document"],
    "filters": {},
    "user_id": "user_001",
    "device_id": "default",
    "agent_id": "default"
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 7. Memory Cache
# ============================================================================

# Get Memory Cache Snapshot
curl -X GET "http://localhost:1733/api/memory-cache?user_id=user_001&device_id=default&agent_id=default&recent_days=7&max_accessed=20&force_refresh=false"
# -H "X-API-Key: your-api-key"

# Invalidate Memory Cache
curl -X DELETE "http://localhost:1733/api/memory-cache?user_id=user_001&device_id=default&agent_id=default"
# -H "X-API-Key: your-api-key"


# ============================================================================
# 8. Contexts
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
# 9. Agent Chat
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
# 10. Conversations
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
# 11. Messages
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
# 12. Documents & WebLinks
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
# 13. Vaults (Document Management)
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
# 14. Events
# ============================================================================

# Fetch and Clear Events
curl -X GET http://localhost:1733/api/events/fetch
# -H "X-API-Key: your-api-key"

# Get Event System Status
curl -X GET http://localhost:1733/api/events/status
# -H "X-API-Key: your-api-key"

# Publish Event
curl -X POST http://localhost:1733/api/events/publish \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "context_updated",
    "data": {
      "context_id": "ctx_abc123",
      "context_type": "knowledge",
      "action": "created"
    }
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 15. Monitoring
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


# ============================================================================
# 16. Settings - Model
# ============================================================================

# Get Model Settings
curl -X GET http://localhost:1733/api/model_settings/get
# -H "X-API-Key: your-api-key"

# Update Model Settings
curl -X POST http://localhost:1733/api/model_settings/update \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "modelPlatform": "openai",
      "modelId": "gpt-4o",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "sk-your-api-key",
      "embeddingModelId": "text-embedding-3-large",
      "embeddingBaseUrl": "https://api.openai.com/v1",
      "embeddingApiKey": "sk-your-api-key",
      "embeddingModelPlatform": "openai"
    }
  }'
# -H "X-API-Key: your-api-key"

# Validate Model Settings
curl -X POST http://localhost:1733/api/model_settings/validate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "modelPlatform": "openai",
      "modelId": "gpt-4o",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "sk-your-api-key",
      "embeddingModelId": "text-embedding-3-large",
      "embeddingBaseUrl": "https://api.openai.com/v1",
      "embeddingApiKey": "sk-your-api-key",
      "embeddingModelPlatform": "openai"
    }
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 17. Settings - General
# ============================================================================

# Get General Settings
curl -X GET http://localhost:1733/api/settings/general
# -H "X-API-Key: your-api-key"

# Update General Settings
curl -X POST http://localhost:1733/api/settings/general \
  -H "Content-Type: application/json" \
  -d '{
    "capture": {
      "screenshot_interval": 30
    },
    "processing": {
      "batch_size": 10
    },
    "logging": {
      "level": "INFO"
    }
  }'
# -H "X-API-Key: your-api-key"


# ============================================================================
# 18. Settings - Prompts
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
# 19. Settings - Reset
# ============================================================================

# Reset All Settings to Defaults
curl -X POST http://localhost:1733/api/settings/reset
# -H "X-API-Key: your-api-key"



# ============================================================================
# NOT REGISTERED IN MAIN ROUTER (route files exist but not included in api.py)
# ============================================================================
# The following route modules exist but are NOT registered in the main router:
# - screenshots.py  (POST /api/add_screenshot, /api/add_screenshots)
# - completions.py  (POST /api/completions/suggest, etc.)
# These endpoints will NOT be accessible unless their routers are added to api.py.
