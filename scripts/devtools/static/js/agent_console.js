/**
 * MineContext DevTools Console — Agent Console
 *
 * Agent list management, basic info editing, profile editing,
 * base events tree display and CRUD.
 *
 * Dependencies (loaded by base.html before this file):
 *   - Bootstrap 5  (bootstrap.Modal, tabs)
 *   - feather-icons
 *   - shared.js     (esc, escAttr, refreshIcons, levelBadge, LEVEL_BADGE_CLASS)
 */

/* ============================================================
   State
   ============================================================ */

let selectedAgentId = null;
let _allAgents = [];
let _eventTreeMap = {};   // id -> tree node (with _children), for cascade delete
let _currentEvents = [];  // flat event list from API, for optimistic UI updates

/* ============================================================
   Init
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {
    // Deep-linking: ?agent_id=xxx
    var params = new URLSearchParams(window.location.search);
    var linked = params.get('agent_id');
    if (linked) {
        selectedAgentId = linked;
    }

    loadAgentList();

    // Tab-switch listener (skeleton for Task 6)
    var tabEl = document.getElementById('agentTabs');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            // Task 6 will add per-tab lazy-loading here
            // e.target is the newly activated tab button
        });
    }
});

/* ============================================================
   Agent List
   ============================================================ */

async function loadAgentList() {
    try {
        var resp = await fetch('/api/agents');
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to load agents');
        var agents = data.data.agents || [];
        _allAgents = agents;
        renderAgentList(agents);

        // Auto-select deep-linked agent
        if (selectedAgentId) {
            var found = agents.some(function (a) { return a.agent_id === selectedAgentId; });
            if (found) {
                selectAgent(selectedAgentId);
            }
        }
    } catch (e) {
        console.error('Load agents failed:', e);
    }
}

function renderAgentList(agents) {
    var list = document.getElementById('agentList');
    var countEl = document.getElementById('agentCount');

    if (agents.length === 0) {
        list.innerHTML = '<div class="text-muted text-center py-4"><small>暂无 Agent</small></div>';
        countEl.textContent = '0 个 Agent';
        return;
    }

    countEl.textContent = agents.length + ' 个 Agent';
    list.innerHTML = agents.map(function (a) {
        var isActive = a.agent_id === selectedAgentId ? ' active' : '';
        return '<div class="sidebar-item' + isActive + '" data-agent-id="' + escAttr(a.agent_id) + '" ' +
            'onclick="selectAgent(\'' + escAttr(a.agent_id) + '\')" title="' + escAttr(a.agent_id) + '">' +
            '<div class="fw-semibold" style="font-size:0.84rem;">' + esc(a.name) + '</div>' +
            '<small class="text-muted" style="font-size:0.75rem;">' + esc(a.agent_id) + '</small>' +
            '</div>';
    }).join('');
}

function filterAgentList(query) {
    var q = (query || '').trim().toLowerCase();
    if (!q) {
        renderAgentList(_allAgents);
        return;
    }
    var filtered = _allAgents.filter(function (a) {
        return (a.agent_id || '').toLowerCase().indexOf(q) !== -1 ||
               (a.name || '').toLowerCase().indexOf(q) !== -1;
    });
    renderAgentList(filtered);
}

function selectAgent(agentId) {
    selectedAgentId = agentId;

    // Update URL without reload
    var url = new URL(window.location);
    url.searchParams.set('agent_id', agentId);
    window.history.replaceState(null, '', url);

    // Highlight active sidebar item
    document.querySelectorAll('#agentList .sidebar-item').forEach(function (el) {
        el.classList.toggle('active', el.getAttribute('data-agent-id') === agentId);
    });

    loadAgentDetail(agentId);
}

/* ============================================================
   Tab 1: Basic Info — Load & Render
   ============================================================ */

async function loadAgentDetail(agentId) {
    var area = document.getElementById('infoArea');
    area.innerHTML = '<div class="text-center py-5"><div class="spinner-border" role="status"></div></div>';

    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(agentId));
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Agent not found');
        var agent = data.data.agent;

        var results = await Promise.all([
            fetch('/api/agents/' + encodeURIComponent(agentId) + '/base/profile').catch(function () { return null; }),
            fetch('/api/agents/' + encodeURIComponent(agentId) + '/base/events').catch(function () { return null; })
        ]);

        var profileResp = results[0];
        var eventsResp = results[1];

        var profile = null;
        if (profileResp && profileResp.ok) {
            var pd = await profileResp.json();
            profile = pd.data.profile;
        }

        var events = [];
        if (eventsResp && eventsResp.ok) {
            var ed = await eventsResp.json();
            events = ed.data.events || [];
        }

        renderInfoTab(agent, profile, events);
    } catch (e) {
        area.innerHTML = '<div class="alert alert-danger">加载失败: ' + esc(e.message) + '</div>';
    }
}

function renderInfoTab(agent, profile, events) {
    var area = document.getElementById('infoArea');
    area.innerHTML = renderInfoCard(agent) + renderProfileCard(profile) + renderEventsCard(events);
    refreshIcons();
}

/* ============================================================
   Card 1: Basic Info
   ============================================================ */

function renderInfoCard(agent) {
    return '<div class="card detail-card">' +
        '<div class="card-header d-flex justify-content-between align-items-center">' +
            '<h6 class="mb-0" role="button" tabindex="0" data-bs-toggle="collapse" data-bs-target="#infoCardBody">' +
                '<i data-feather="info" style="width:16px;height:16px;" class="me-2"></i>基本信息</h6>' +
            '<div>' +
                '<button class="btn btn-sm btn-outline-primary me-1" onclick="startEditInfo()">编辑</button>' +
                '<button class="btn btn-sm btn-outline-danger" onclick="deleteAgent()">删除</button>' +
            '</div>' +
        '</div>' +
        '<div class="collapse show" id="infoCardBody">' +
            '<div class="card-body" id="infoContent">' +
                '<div class="row">' +
                    '<div class="col-md-6 mb-2"><strong>Agent ID:</strong> ' + esc(agent.agent_id) + '</div>' +
                    '<div class="col-md-6 mb-2"><strong>名称:</strong> <span id="infoName">' + esc(agent.name) + '</span></div>' +
                    '<div class="col-12 mb-2"><strong>描述:</strong> <span id="infoDesc">' + esc(agent.description || '无') + '</span></div>' +
                    '<div class="col-md-6"><small class="text-muted">创建: ' + (agent.created_at || '-') + '</small></div>' +
                    '<div class="col-md-6"><small class="text-muted">更新: ' + (agent.updated_at || '-') + '</small></div>' +
                '</div>' +
            '</div>' +
        '</div>' +
    '</div>';
}

function startEditInfo() {
    var nameEl = document.getElementById('infoName');
    var descEl = document.getElementById('infoDesc');
    if (!nameEl || !descEl) return;

    var name = nameEl.textContent;
    var desc = descEl.textContent === '无' ? '' : descEl.textContent;

    nameEl.innerHTML = '<input type="text" class="form-control form-control-sm d-inline-block" ' +
        'style="width:200px" id="editName" value="' + escAttr(name) + '">';
    descEl.innerHTML = '<input type="text" class="form-control form-control-sm d-inline-block" ' +
        'style="width:300px" id="editDesc" value="' + escAttr(desc) + '">';

    var content = document.getElementById('infoContent');
    if (!document.getElementById('editInfoBtns')) {
        content.insertAdjacentHTML('beforeend',
            '<div id="editInfoBtns" class="mt-2">' +
                '<button class="btn btn-sm btn-primary" onclick="saveInfo()">保存</button>' +
                '<button class="btn btn-sm btn-secondary ms-1" onclick="loadAgentDetail(selectedAgentId)">取消</button>' +
            '</div>');
    }
}

async function saveInfo() {
    var name = document.getElementById('editName').value.trim();
    var desc = document.getElementById('editDesc').value.trim();
    if (!name) { alert('名称不能为空'); return; }
    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, description: desc })
        });
        if (!resp.ok) {
            var d = await resp.json();
            throw new Error(d.detail || 'Update failed');
        }
        loadAgentList();
        loadAgentDetail(selectedAgentId);
    } catch (e) {
        alert('更新失败: ' + e.message);
    }
}

async function deleteAgent() {
    if (!confirm('确定要删除 Agent "' + selectedAgentId + '" 吗？此操作不可撤销。')) return;
    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId), { method: 'DELETE' });
        if (!resp.ok) {
            var d = await resp.json();
            throw new Error(d.detail || 'Delete failed');
        }
        selectedAgentId = null;

        // Update URL
        var url = new URL(window.location);
        url.searchParams.delete('agent_id');
        window.history.replaceState(null, '', url);

        // Reset info area
        document.getElementById('infoArea').innerHTML =
            '<div class="text-muted text-center py-5">' +
                '<i data-feather="users" style="width:48px;height:48px;" class="mb-3"></i>' +
                '<p>请从左侧选择一个 Agent</p>' +
            '</div>';
        refreshIcons();
        loadAgentList();
    } catch (e) {
        alert('删除失败: ' + e.message);
    }
}

/* ============================================================
   Card 2: Base Profile
   ============================================================ */

function renderProfileCard(profile) {
    var text = profile ? (profile.factual_profile || '无内容') : null;
    var content = text
        ? '<div class="profile-display" id="profileText">' + esc(text) + '</div>'
        : '<p class="text-muted" id="profileText">暂无 Base Profile，点击编辑添加</p>';

    return '<div class="card detail-card">' +
        '<div class="card-header d-flex justify-content-between align-items-center">' +
            '<h6 class="mb-0" role="button" tabindex="0" data-bs-toggle="collapse" data-bs-target="#profileCardBody">' +
                '<i data-feather="user" style="width:16px;height:16px;" class="me-2"></i>Base Profile</h6>' +
            '<button class="btn btn-sm btn-outline-primary" onclick="startEditProfile()">编辑</button>' +
        '</div>' +
        '<div class="collapse show" id="profileCardBody">' +
            '<div class="card-body" id="profileContent">' + content + '</div>' +
        '</div>' +
    '</div>';
}

function startEditProfile() {
    var container = document.getElementById('profileContent');
    var existing = document.getElementById('profileText');
    var text = existing
        ? (existing.classList.contains('profile-display') ? existing.textContent : '')
        : '';

    container.innerHTML =
        '<textarea class="form-control" id="editProfileText" rows="5">' + esc(text) + '</textarea>' +
        '<div class="mt-2">' +
            '<button class="btn btn-sm btn-primary" onclick="saveProfile()">保存</button>' +
            '<button class="btn btn-sm btn-secondary ms-1" onclick="loadAgentDetail(selectedAgentId)">取消</button>' +
        '</div>';
}

async function saveProfile() {
    var text = document.getElementById('editProfileText').value.trim();
    if (!text) { alert('Profile 内容不能为空'); return; }
    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId) + '/base/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ factual_profile: text })
        });
        if (!resp.ok) {
            var d = await resp.json();
            throw new Error(d.detail || 'Save failed');
        }
        loadAgentDetail(selectedAgentId);
    } catch (e) {
        alert('保存失败: ' + e.message);
    }
}

/* ============================================================
   Card 3: Base Events
   ============================================================ */

function renderEventsCard(events) {
    _currentEvents = events;
    var content;
    if (events.length === 0) {
        content = '<p class="text-muted">暂无 Base Events</p>';
    } else {
        var tree = buildEventTree(events);
        content = tree.map(function (node) { return renderEventNode(node); }).join('');
    }

    return '<div class="card detail-card">' +
        '<div class="card-header d-flex justify-content-between align-items-center">' +
            '<h6 class="mb-0" role="button" tabindex="0" data-bs-toggle="collapse" data-bs-target="#eventsCardBody">' +
                '<i data-feather="zap" style="width:16px;height:16px;" class="me-2"></i>Base Events ' +
                '<span class="badge bg-secondary">' + events.length + '</span></h6>' +
            '<div>' +
                '<button class="btn btn-sm btn-outline-primary me-1" onclick="showBatchImportModal()">批量导入</button>' +
                '<button class="btn btn-sm btn-outline-success" onclick="showAddEventModal()">+ 添加</button>' +
            '</div>' +
        '</div>' +
        '<div class="collapse show" id="eventsCardBody">' +
            '<div class="card-body" id="eventsContent">' + content + '</div>' +
        '</div>' +
    '</div>';
}

/* ============================================================
   Event Tree Building & Rendering
   ============================================================ */

function buildEventTree(events) {
    // Build a map of id -> event (with a _children array)
    var map = {};
    var i, evt;
    for (i = 0; i < events.length; i++) {
        evt = events[i];
        map[evt.id] = Object.assign({}, evt, { _children: [] });
    }

    // Use downward refs only to build parent-child relationships
    // Downward: parent (higher level) refs child (lower level)
    var childIds = new Set();
    for (i = 0; i < events.length; i++) {
        evt = events[i];
        if (evt.refs) {
            var evtLevel = evt.hierarchy_level || 0;
            var refKeys = Object.keys(evt.refs);
            for (var k = 0; k < refKeys.length; k++) {
                var ids = evt.refs[refKeys[k]];
                if (!Array.isArray(ids)) continue;
                for (var j = 0; j < ids.length; j++) {
                    var cid = ids[j];
                    if (map[cid] && (map[cid].hierarchy_level || 0) < evtLevel) {
                        map[evt.id]._children.push(map[cid]);
                        childIds.add(cid);
                    }
                }
            }
        }
    }

    // Sort children by event_time_start
    var allNodes = Object.values(map);
    for (i = 0; i < allNodes.length; i++) {
        allNodes[i]._children.sort(function (a, b) {
            return (a.event_time_start || '').localeCompare(b.event_time_start || '');
        });
    }

    // Store map globally for cascade delete
    _eventTreeMap = map;

    // Roots are events that are not children of any other event
    var roots = events.filter(function (e) { return !childIds.has(e.id); })
                      .map(function (e) { return map[e.id]; });

    // Sort roots: higher level first, then by event_time_start desc
    roots.sort(function (a, b) {
        return (b.hierarchy_level || 0) - (a.hierarchy_level || 0) ||
               (b.event_time_start || '').localeCompare(a.event_time_start || '');
    });

    return roots;
}

function renderEventNode(node) {
    var level = node.hierarchy_level || 0;
    var levelClass = level > 0 ? ' event-item-' + level : '';
    var hasChildren = node._children && node._children.length > 0;
    var nodeId = 'evt-' + node.id.replace(/[^a-zA-Z0-9]/g, '_');

    var toggleHtml = hasChildren
        ? '<span class="tree-toggle me-1" onclick="event.stopPropagation(); toggleEvtChildren(\'' + nodeId + '\', this)">' +
              '<i data-feather="chevron-down" style="width:14px;height:14px;"></i>' +
          '</span>'
        : '';

    var timeDisplay = '';
    if (node.event_time_start) {
        if (node.event_time_end && node.event_time_end !== node.event_time_start) {
            timeDisplay = new Date(node.event_time_start).toLocaleString('zh-CN') +
                ' ~ ' + new Date(node.event_time_end).toLocaleString('zh-CN');
        } else {
            timeDisplay = new Date(node.event_time_start).toLocaleString('zh-CN');
        }
    }

    var childrenHtml = hasChildren
        ? '<div class="tree-children" id="' + nodeId + '">' +
              node._children.map(function (c) { return renderEventNode(c); }).join('') +
          '</div>'
        : '';

    var keywordsHtml = '';
    if (node.keywords && node.keywords.length) {
        keywordsHtml = '<div class="mt-1">' +
            node.keywords.map(function (k) {
                return '<span class="badge bg-info bg-opacity-75 me-1">' + esc(k) + '</span>';
            }).join('') +
        '</div>';
    }

    return '<div class="event-item' + levelClass + '">' +
        '<div class="d-flex justify-content-between align-items-start">' +
            '<div>' +
                '<div class="d-flex align-items-center">' +
                    toggleHtml +
                    levelBadge(level) +
                    '<span class="fw-semibold ms-1">' + esc(node.title || '未命名') + '</span>' +
                '</div>' +
                (node.summary ? '<small class="text-muted">' + esc(node.summary) + '</small>' : '') +
                keywordsHtml +
            '</div>' +
            '<button class="btn btn-sm btn-outline-danger ms-2" onclick="deleteEvent(\'' + escAttr(node.id) + '\')" title="删除">' +
                '<i data-feather="trash-2" style="width:14px;height:14px;"></i>' +
            '</button>' +
        '</div>' +
        (timeDisplay ? '<small class="text-muted">' + timeDisplay + '</small>' : '') +
    '</div>' + childrenHtml;
}

function toggleEvtChildren(id, el) {
    var container = document.getElementById(id);
    if (!container) return;
    container.classList.toggle('d-none');
    var icon = el.querySelector('svg');
    if (icon) {
        icon.style.transform = container.classList.contains('d-none') ? 'rotate(-90deg)' : '';
    }
}

/* ============================================================
   Event Delete (cascade, leaf-first)
   ============================================================ */

function _collectDescendantIds(node) {
    // Post-order: children's descendants first, then child itself (leaf-first)
    var ids = [];
    if (node._children) {
        for (var i = 0; i < node._children.length; i++) {
            var child = node._children[i];
            ids = ids.concat(_collectDescendantIds(child));
            ids.push(child.id);
        }
    }
    return ids;
}

async function deleteEvent(eventId) {
    var node = _eventTreeMap[eventId];
    var descendantIds = node ? _collectDescendantIds(node) : [];
    // Leaf-first order: descendants (already post-order), then the root node last
    var allIds = descendantIds.concat([eventId]);

    var msg = allIds.length > 1
        ? '确定删除此事件及其 ' + descendantIds.length + ' 个子级事件？（共 ' + allIds.length + ' 条）'
        : '确定删除此事件？';
    if (!confirm(msg)) return;

    var failed = 0;
    var deletedIds = new Set();
    for (var i = 0; i < allIds.length; i++) {
        var id = allIds[i];
        try {
            var resp = await fetch(
                '/api/agents/' + encodeURIComponent(selectedAgentId) +
                '/base/events/' + encodeURIComponent(id),
                { method: 'DELETE' }
            );
            if (!resp.ok) {
                console.error('Delete failed for event:', id);
                failed++;
            } else {
                deletedIds.add(id);
            }
        } catch (e) {
            console.error('Delete failed for event:', id, e);
            failed++;
        }
    }
    if (failed > 0) {
        alert((allIds.length - failed) + ' 条删除成功，' + failed + ' 条删除失败');
    }

    // Optimistic UI: remove deleted events from local list and re-render
    var remaining = _currentEvents.filter(function (e) { return !deletedIds.has(e.id); });
    var container = document.getElementById('eventsContent');
    if (container) {
        _currentEvents = remaining;
        if (remaining.length === 0) {
            container.innerHTML = '<p class="text-muted">暂无 Base Events</p>';
        } else {
            var tree = buildEventTree(remaining);
            container.innerHTML = tree.map(function (node) { return renderEventNode(node); }).join('');
        }
        // Update count badge
        var badge = document.querySelector('#eventsCardBody');
        if (badge) {
            var card = badge.closest('.card');
            if (card) {
                var b = card.querySelector('.badge.bg-secondary');
                if (b) b.textContent = remaining.length;
            }
        }
        refreshIcons();
    }
}

/* ============================================================
   Modal: Create Agent
   ============================================================ */

function showCreateAgentModal() {
    document.getElementById('newAgentId').value = '';
    document.getElementById('newAgentName').value = '';
    document.getElementById('newAgentDesc').value = '';
    new bootstrap.Modal(document.getElementById('createAgentModal')).show();
}

async function createAgent() {
    var agentId = document.getElementById('newAgentId').value.trim();
    var name = document.getElementById('newAgentName').value.trim();
    var desc = document.getElementById('newAgentDesc').value.trim();
    if (!agentId || !name) { alert('Agent ID 和名称不能为空'); return; }

    try {
        var resp = await fetch('/api/agents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId, name: name, description: desc })
        });
        if (!resp.ok) {
            var d = await resp.json();
            throw new Error(d.detail || 'Create failed');
        }
        bootstrap.Modal.getInstance(document.getElementById('createAgentModal')).hide();
        document.getElementById('newAgentId').value = '';
        document.getElementById('newAgentName').value = '';
        document.getElementById('newAgentDesc').value = '';
        await loadAgentList();
        selectAgent(agentId);
    } catch (e) {
        alert('创建失败: ' + e.message);
    }
}

/* ============================================================
   Modal: Add Base Event
   ============================================================ */

function showAddEventModal() {
    document.getElementById('newEventTitle').value = '';
    document.getElementById('newEventSummary').value = '';
    document.getElementById('newEventTime').value = '';
    document.getElementById('newEventKeywords').value = '';
    document.getElementById('newEventImportance').value = '5';
    document.getElementById('importanceValue').textContent = '5';
    new bootstrap.Modal(document.getElementById('addEventModal')).show();
}

async function addBaseEvent() {
    var title = document.getElementById('newEventTitle').value.trim();
    var summary = document.getElementById('newEventSummary').value.trim();
    if (!title || !summary) { alert('标题和摘要不能为空'); return; }

    var eventObj = { title: title, summary: summary };

    var eventTime = document.getElementById('newEventTime').value;
    if (eventTime) eventObj.event_time_start = new Date(eventTime).toISOString();

    var kw = document.getElementById('newEventKeywords').value.trim();
    if (kw) eventObj.keywords = kw.split(',').map(function (s) { return s.trim(); }).filter(Boolean);

    var imp = parseInt(document.getElementById('newEventImportance').value, 10);
    if (!isNaN(imp)) eventObj.importance = Math.max(0, Math.min(10, imp));

    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId) + '/base/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ events: [eventObj] })
        });
        if (!resp.ok) {
            var d = await resp.json();
            throw new Error(d.detail || 'Add failed');
        }
        bootstrap.Modal.getInstance(document.getElementById('addEventModal')).hide();
        loadAgentDetail(selectedAgentId);
    } catch (e) {
        alert('添加失败: ' + e.message);
    }
}

/* ============================================================
   Modal: Batch Import
   ============================================================ */

function showBatchImportModal() {
    document.getElementById('batchJsonInput').value = '';
    document.getElementById('batchError').classList.add('d-none');
    document.getElementById('batchSubmitBtn').disabled = false;
    document.getElementById('batchSubmitBtn').textContent = '导入';
    new bootstrap.Modal(document.getElementById('batchImportModal')).show();
}

function fillBatchExample() {
    document.getElementById('batchJsonInput').value = JSON.stringify({
        "events": [
            {
                "title": "Standalone event",
                "summary": "A simple L0 event",
                "event_time_start": "2026-03-15T10:00:00+08:00",
                "importance": 5
            },
            {
                "title": "Daily Summary",
                "summary": "Summary of the day",
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
                    }
                ]
            }
        ]
    }, null, 2);
}

async function submitBatchImport() {
    var errorEl = document.getElementById('batchError');
    var btn = document.getElementById('batchSubmitBtn');
    errorEl.classList.add('d-none');

    var raw = document.getElementById('batchJsonInput').value.trim();
    if (!raw) {
        errorEl.textContent = 'JSON 不能为空';
        errorEl.classList.remove('d-none');
        return;
    }

    var payload;
    try {
        payload = JSON.parse(raw);
    } catch (e) {
        errorEl.textContent = 'JSON 解析失败: ' + e.message;
        errorEl.classList.remove('d-none');
        return;
    }

    if (!payload.events || !Array.isArray(payload.events) || payload.events.length === 0) {
        errorEl.textContent = '请求体必须包含非空的 events 数组';
        errorEl.classList.remove('d-none');
        return;
    }

    btn.disabled = true;
    btn.textContent = '导入中...';
    try {
        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId) + '/base/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!resp.ok) {
            var detail = 'HTTP ' + resp.status;
            try {
                var errData = await resp.json();
                detail = errData.detail || errData.message || detail;
            } catch (_) { /* ignore parse error */ }
            throw new Error(detail);
        }
        var data = await resp.json();
        bootstrap.Modal.getInstance(document.getElementById('batchImportModal')).hide();
        alert('导入成功: ' + (data.data && data.data.count != null ? data.data.count : '?') + ' 条事件');
        loadAgentDetail(selectedAgentId);
    } catch (e) {
        errorEl.textContent = '导入失败: ' + e.message;
        errorEl.classList.remove('d-none');
    } finally {
        btn.disabled = false;
        btn.textContent = '导入';
    }
}
