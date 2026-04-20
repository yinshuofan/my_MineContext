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

    // Tab-switch listener
    var tabEl = document.getElementById('agentTabs');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            if (!selectedAgentId) return;
            var target = e.target.getAttribute('data-bs-target');
            if (target === '#tab-batches') initBatchesTab();
            if (target === '#tab-contexts') initContextsTab();
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

    // Reinitialize the active non-info tab when agent changes
    var activeTab = document.querySelector('#agentTabs .nav-link.active');
    if (activeTab) {
        var target = activeTab.getAttribute('data-bs-target');
        if (target === '#tab-batches') initBatchesTab();
        if (target === '#tab-contexts') initContextsTab();
    }
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
                '<button class="btn btn-sm btn-outline-danger me-1" onclick="deleteAllEvents()"' +
                    (events.length === 0 ? ' disabled' : '') + '>清空</button>' +
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
   Event Delete (single DELETE, server prunes the subtree)
   ============================================================ */

function _estimateSubtreeSize(eventId) {
    var node = _eventTreeMap ? _eventTreeMap[eventId] : null;
    if (!node) return 1;
    var count = 1;
    var stack = (node._children || []).slice();
    while (stack.length) {
        var cur = stack.pop();
        count += 1;
        if (cur._children) {
            for (var i = 0; i < cur._children.length; i++) stack.push(cur._children[i]);
        }
    }
    return count;
}

async function deleteEvent(eventId) {
    var subtreeSize = _estimateSubtreeSize(eventId);
    var confirmMsg = subtreeSize > 1
        ? '确定删除此事件及其 ' + (subtreeSize - 1) + ' 个子级事件？'
        : '确定删除此事件？';
    if (!confirm(confirmMsg)) return;

    try {
        var resp = await fetch(
            '/api/agents/' + encodeURIComponent(selectedAgentId) +
            '/base/events/' + encodeURIComponent(eventId),
            { method: 'DELETE' }
        );
        if (!resp.ok) {
            var errBody = {};
            try { errBody = await resp.json(); } catch (_) { /* ignore */ }
            alert('删除失败：' + (errBody.detail || resp.statusText));
            return;
        }
        var body = await resp.json();
        var deletedIds = (body.data && body.data.deleted_ids) || [];
        var deleted = new Set(deletedIds);

        // Update local state based on server response.
        // VikingDB indexing can lag — rely on the authoritative server set.
        _currentEvents = _currentEvents.filter(function (e) { return !deleted.has(e.id); });

        // Re-render the events card body in place.
        var container = document.getElementById('eventsContent');
        if (container) {
            if (_currentEvents.length === 0) {
                container.innerHTML = '<p class="text-muted">暂无 Base Events</p>';
            } else {
                var tree = buildEventTree(_currentEvents);
                container.innerHTML = tree.map(function (node) { return renderEventNode(node); }).join('');
            }
            // Update count badge
            var eventsBody = document.getElementById('eventsCardBody');
            if (eventsBody) {
                var card = eventsBody.closest('.card');
                if (card) {
                    var b = card.querySelector('.badge.bg-secondary');
                    if (b) b.textContent = _currentEvents.length;
                }
            }
            refreshIcons();
        }
    } catch (err) {
        console.error('deleteEvent failed', err);
        alert('删除失败：' + err.message);
    }
}

async function deleteAllEvents() {
    if (!selectedAgentId) return;
    if (_currentEvents.length === 0) return;
    if (!confirm('确定清空此 Agent 的全部 ' + _currentEvents.length + ' 条 Base Events？此操作不可撤销。')) return;

    try {
        var resp = await fetch(
            '/api/agents/' + encodeURIComponent(selectedAgentId) + '/base/events',
            { method: 'DELETE' }
        );
        if (!resp.ok) {
            var errBody = {};
            try { errBody = await resp.json(); } catch (_) { /* ignore */ }
            alert('清空失败：' + (errBody.detail || resp.statusText));
            return;
        }

        _currentEvents = [];

        var container = document.getElementById('eventsContent');
        if (container) {
            container.innerHTML = '<p class="text-muted">暂无 Base Events</p>';
            var eventsBody = document.getElementById('eventsCardBody');
            var card = eventsBody ? eventsBody.closest('.card') : null;
            if (card) {
                var badge = card.querySelector('.badge.bg-secondary');
                if (badge) badge.textContent = '0';
                var clearBtn = card.querySelector('.card-header .btn-outline-danger');
                if (clearBtn) clearBtn.disabled = true;
            }
            refreshIcons();
        }
    } catch (err) {
        console.error('deleteAllEvents failed', err);
        alert('清空失败：' + err.message);
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

/**
 * Convert flat _currentEvents list (with bidirectional refs) back to the
 * nested tree shape the POST endpoint accepts. Each root node has its
 * descendants inlined via `children`.
 */
function _toPushTree(flatEvents) {
    if (!flatEvents || flatEvents.length === 0) return [];

    // Build id -> event map.
    var byId = {};
    var i, ev;
    for (i = 0; i < flatEvents.length; i++) {
        ev = flatEvents[i];
        byId[ev.id] = ev;
    }

    // Detect downward refs: a ref is downward when the target has a smaller hierarchy_level.
    var childrenById = {};  // parent_id -> [child_id, ...]
    for (i = 0; i < flatEvents.length; i++) {
        ev = flatEvents[i];
        var parentLevel = ev.hierarchy_level != null ? ev.hierarchy_level : 0;
        var refs = ev.refs || {};
        var refTypes = Object.keys(refs);
        for (var k = 0; k < refTypes.length; k++) {
            var refIds = refs[refTypes[k]] || [];
            for (var j = 0; j < refIds.length; j++) {
                var refId = refIds[j];
                var refEv = byId[refId];
                if (!refEv) continue;
                var refLevel = refEv.hierarchy_level != null ? refEv.hierarchy_level : 0;
                if (refLevel < parentLevel) {
                    if (!childrenById[ev.id]) childrenById[ev.id] = [];
                    childrenById[ev.id].push(refId);
                }
            }
        }
    }

    // Identify roots: events with no upward ref (no ref target has higher level).
    var hasParent = new Set();
    var parentIds = Object.keys(childrenById);
    for (i = 0; i < parentIds.length; i++) {
        var kids = childrenById[parentIds[i]];
        for (j = 0; j < kids.length; j++) hasParent.add(kids[j]);
    }

    function toNode(evNode) {
        var node = {
            title: evNode.title,
            summary: evNode.summary,
            hierarchy_level: evNode.hierarchy_level != null ? evNode.hierarchy_level : 0,
            importance: evNode.importance != null ? evNode.importance : 5,
            keywords: evNode.keywords || [],
            entities: evNode.entities || []
        };
        if (evNode.event_time_start) node.event_time_start = evNode.event_time_start;
        if (evNode.event_time_end) node.event_time_end = evNode.event_time_end;
        var childIds = childrenById[evNode.id];
        if (childIds && childIds.length) {
            node.children = childIds.map(function (cid) { return toNode(byId[cid]); }).filter(Boolean);
        }
        return node;
    }

    return flatEvents.filter(function (e) { return !hasParent.has(e.id); }).map(toNode);
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
        // Replace-semantics POST — fetch current tree, append, re-POST.
        var existing = _currentEvents || [];
        var mergedTree = _toPushTree(existing);
        mergedTree.push(eventObj);

        var resp = await fetch('/api/agents/' + encodeURIComponent(selectedAgentId) + '/base/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ events: mergedTree })
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
    document.getElementById('batchJsonInput').classList.remove('drop-active');
    document.getElementById('batchError').classList.add('d-none');
    document.getElementById('batchSubmitBtn').disabled = false;
    document.getElementById('batchSubmitBtn').textContent = '导入';
    new bootstrap.Modal(document.getElementById('batchImportModal')).show();
}

var _BATCH_MAX_BYTES = 5 * 1024 * 1024;  // 5 MB; ~500 events typically <1 MB

function handleBatchFile(file) {
    var errorEl = document.getElementById('batchError');
    errorEl.classList.add('d-none');
    if (!file) return;
    var name = (file.name || '').toLowerCase();
    var type = (file.type || '').toLowerCase();
    if (!name.endsWith('.json') && type !== 'application/json') {
        errorEl.textContent = '不是 .json 文件：' + (file.name || '(未命名)');
        errorEl.classList.remove('d-none');
        return;
    }
    if (file.size > _BATCH_MAX_BYTES) {
        errorEl.textContent = '文件过大 (' + (file.size / 1024 / 1024).toFixed(1)
            + ' MB)，上限 ' + (_BATCH_MAX_BYTES / 1024 / 1024) + ' MB';
        errorEl.classList.remove('d-none');
        return;
    }
    var reader = new FileReader();
    reader.onload = function () {
        document.getElementById('batchJsonInput').value = reader.result;
    };
    reader.onerror = function () {
        var msg = (reader.error && reader.error.message) || 'unknown';
        errorEl.textContent = '读取文件失败：' + msg;
        errorEl.classList.remove('d-none');
    };
    reader.readAsText(file);
}

function batchDragOver(event) {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    event.currentTarget.classList.add('drop-active');
}

function batchDragLeave(event) {
    event.currentTarget.classList.remove('drop-active');
}

function batchDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('drop-active');
    var files = event.dataTransfer && event.dataTransfer.files;
    if (!files || files.length === 0) return;
    if (files.length > 1) {
        var errorEl = document.getElementById('batchError');
        errorEl.textContent = '一次只能导入一个文件（已忽略 ' + (files.length - 1) + ' 个）';
        errorEl.classList.remove('d-none');
    }
    handleBatchFile(files[0]);
}

// Prevent the browser from navigating away when a file is accidentally
// dropped outside the textarea (default behavior opens the file as the page).
window.addEventListener('dragover', function (e) { e.preventDefault(); });
window.addEventListener('drop', function (e) { e.preventDefault(); });

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
        alert('导入成功: ' + (data.data && data.data.upserted != null ? data.data.upserted : '?') + ' 条事件');
        loadAgentDetail(selectedAgentId);
    } catch (e) {
        errorEl.textContent = '导入失败: ' + e.message;
        errorEl.classList.remove('d-none');
    } finally {
        btn.disabled = false;
        btn.textContent = '导入';
    }
}

/* ============================================================
   Tab 2: Message Tracing (消息追踪)
   ============================================================ */

let batchPage = 1;
const batchLimit = 20;
let expandedBatchId = null;

function initBatchesTab() {
    var container = document.getElementById('tab-batches');
    if (!container) return;

    container.innerHTML =
        // Filter bar
        '<div class="card mb-3" style="border-top: 3px solid #2c3e50;">' +
            '<div class="card-body py-3 px-3">' +
                '<div class="row g-2 align-items-end">' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">User ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="batchFilterUserId" placeholder="可选">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Device ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="batchFilterDeviceId" placeholder="可选">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Agent ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="batchFilterAgentId" value="' + escAttr(selectedAgentId || '') + '" disabled>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">开始时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="batchFilterStartDate">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">结束时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="batchFilterEndDate">' +
                    '</div>' +
                    '<div class="col-12 col-md-auto">' +
                        '<div class="d-flex gap-2">' +
                            '<button class="btn btn-sm btn-dark" onclick="loadBatches(1)">筛选</button>' +
                            '<button class="btn btn-sm btn-outline-secondary" onclick="resetBatchFilters()">重置</button>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +
        // Table
        '<div class="card" style="border-top: 3px solid #2c3e50;">' +
            '<div class="card-body p-0">' +
                '<div class="table-responsive">' +
                    '<table class="table table-hover mb-0" id="batchTable">' +
                        '<thead class="table-light">' +
                            '<tr>' +
                                '<th style="width:40px;"></th>' +
                                '<th>Batch ID</th>' +
                                '<th>User ID</th>' +
                                '<th>消息数</th>' +
                                '<th>创建时间</th>' +
                            '</tr>' +
                        '</thead>' +
                        '<tbody id="batchTableBody">' +
                            '<tr><td colspan="5" class="text-center text-muted py-4"><div class="spinner-border spinner-border-sm"></div> 加载中...</td></tr>' +
                        '</tbody>' +
                    '</table>' +
                '</div>' +
            '</div>' +
        '</div>' +
        // Pagination
        '<div id="batchPaginationArea" class="mt-3"></div>';

    loadBatches(1);
}

function resetBatchFilters() {
    var uid = document.getElementById('batchFilterUserId');
    var did = document.getElementById('batchFilterDeviceId');
    var sd = document.getElementById('batchFilterStartDate');
    var ed = document.getElementById('batchFilterEndDate');
    if (uid) uid.value = '';
    if (did) did.value = '';
    if (sd) sd.value = '';
    if (ed) ed.value = '';
    loadBatches(1);
}

function getBatchFilterParams() {
    var params = new URLSearchParams();
    params.set('agent_id', selectedAgentId);
    var userId = document.getElementById('batchFilterUserId');
    var deviceId = document.getElementById('batchFilterDeviceId');
    var startDate = document.getElementById('batchFilterStartDate');
    var endDate = document.getElementById('batchFilterEndDate');
    if (userId && userId.value.trim()) params.set('user_id', userId.value.trim());
    if (deviceId && deviceId.value.trim()) params.set('device_id', deviceId.value.trim());
    if (startDate && startDate.value) params.set('start_date', startDate.value);
    if (endDate && endDate.value) params.set('end_date', endDate.value);
    return params;
}

async function loadBatches(page) {
    batchPage = page;
    var params = getBatchFilterParams();
    params.set('page', page);
    params.set('limit', batchLimit);

    var tbody = document.getElementById('batchTableBody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4"><div class="spinner-border spinner-border-sm"></div> 加载中...</td></tr>';

    try {
        var resp = await fetch('/api/chat-batches?' + params);
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || '加载失败');

        var batches = data.data.batches || [];
        var total = data.data.total || 0;
        var totalPages = data.data.total_pages || 1;

        if (batches.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-5">没有找到消息批次</td></tr>';
            document.getElementById('batchPaginationArea').innerHTML = '';
            return;
        }

        expandedBatchId = null;
        tbody.innerHTML = batches.map(function (b) { return renderBatchRow(b); }).join('');
        renderBatchPagination(page, totalPages, total);
        refreshIcons();
    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">加载失败: ' + esc(e.message) + '</td></tr>';
    }
}

function userLink(userId, deviceId, agentId) {
    if (!userId || userId === 'default') return esc(userId || '');
    var href = '/console/memory?user_id=' + encodeURIComponent(userId) +
        '&device_id=' + encodeURIComponent(deviceId || 'default') +
        '&agent_id=' + encodeURIComponent(agentId || 'default');
    return '<a href="' + escAttr(href) + '" title="在记忆浏览器中查看">' + esc(userId) + '</a>';
}

function renderBatchRow(batch) {
    var id = batch.batch_id || '';
    var safeId = escAttr(id);
    var shortId = esc(id).substring(0, 8);
    var time = batch.created_at ? String(batch.created_at).substring(0, 19).replace('T', ' ') : '-';
    return '<tr class="batch-row" data-batch-id="' + safeId + '" onclick="toggleBatchExpand(\'' + safeId + '\')" style="cursor:pointer;">' +
            '<td class="text-center"><i data-feather="chevron-right" class="expand-btn" id="batch-expand-' + safeId + '" style="width:16px;height:16px;"></i></td>' +
            '<td><code title="' + safeId + '">' + shortId + '</code></td>' +
            '<td>' + userLink(batch.user_id, batch.device_id, batch.agent_id) + '</td>' +
            '<td><span class="badge bg-secondary">' + (batch.message_count || 0) + '</span></td>' +
            '<td><small>' + esc(time) + '</small></td>' +
        '</tr>' +
        '<tr class="detail-row d-none" id="batch-detail-' + safeId + '">' +
            '<td colspan="5" style="padding:0 !important; background:#f8f9fa;">' +
                '<div class="detail-content" id="batch-detail-content-' + safeId + '" style="padding:16px 24px;"></div>' +
            '</td>' +
        '</tr>';
}

async function toggleBatchExpand(batchId) {
    var detailRow = document.getElementById('batch-detail-' + batchId);
    var expandBtn = document.getElementById('batch-expand-' + batchId);

    if (expandedBatchId === batchId) {
        // Collapse
        if (detailRow) detailRow.classList.add('d-none');
        if (expandBtn) expandBtn.classList.remove('open');
        expandedBatchId = null;
        return;
    }

    // Collapse previous
    if (expandedBatchId) {
        var prevDetail = document.getElementById('batch-detail-' + expandedBatchId);
        var prevBtn = document.getElementById('batch-expand-' + expandedBatchId);
        if (prevDetail) prevDetail.classList.add('d-none');
        if (prevBtn) prevBtn.classList.remove('open');
    }

    // Expand new
    expandedBatchId = batchId;
    if (detailRow) detailRow.classList.remove('d-none');
    if (expandBtn) expandBtn.classList.add('open');

    var content = document.getElementById('batch-detail-content-' + batchId);
    if (!content) return;
    content.innerHTML = '<div class="text-center py-3"><div class="spinner-border spinner-border-sm"></div> 加载详情...</div>';

    try {
        var results = await Promise.all([
            fetch('/api/chat-batches/' + encodeURIComponent(batchId)),
            fetch('/api/chat-batches/' + encodeURIComponent(batchId) + '/contexts')
        ]);

        var batchResp = results[0];
        var ctxResp = results[1];
        var batchData = await batchResp.json();
        var ctxData = await ctxResp.json();

        if (!batchResp.ok) throw new Error(batchData.detail || '加载批次失败');

        var batch = batchData.data.batch;
        var contexts = (ctxResp.ok && ctxData.data) ? (ctxData.data.contexts || []) : [];

        content.innerHTML =
            '<div class="row">' +
                '<div class="col-md-6">' +
                    '<h6 class="mb-2">原始消息 (' + (batch.message_count || 0) + ')</h6>' +
                    '<div class="border rounded p-2" style="max-height:400px;overflow-y:auto;">' +
                        renderBatchMessages(batch.messages || []) +
                    '</div>' +
                '</div>' +
                '<div class="col-md-6">' +
                    '<h6 class="mb-2">关联数据 (' + contexts.length + ')</h6>' +
                    '<div style="max-height:400px;overflow-y:auto;">' +
                        (contexts.length > 0
                            ? contexts.map(function (c) { return renderBatchContextCard(c); }).join('')
                            : '<p class="text-muted">暂无关联数据</p>') +
                    '</div>' +
                '</div>' +
            '</div>';
        refreshIcons();
    } catch (e) {
        content.innerHTML = '<div class="alert alert-danger mb-0">加载失败: ' + esc(e.message) + '</div>';
    }
}

function renderBatchMessages(messages) {
    if (!messages || messages.length === 0) return '<p class="text-muted">无消息</p>';
    return messages.map(function (m) {
        var role = (m.role || 'user').toLowerCase();
        var roleClass = role === 'assistant' ? 'msg-assistant' : role === 'system' ? 'msg-system' : 'msg-user';
        var roleBadge = role === 'user' ? 'primary' : role === 'assistant' ? 'success' : 'warning';
        var msgContent = '';
        if (typeof m.content === 'string') {
            msgContent = esc(m.content);
        } else if (Array.isArray(m.content)) {
            msgContent = m.content.map(function (part) {
                if (part.type === 'text') return esc(part.text || '');
                if (part.type === 'image_url') return '<span class="badge bg-info">图片</span>';
                if (part.type === 'video_url') return '<span class="badge bg-dark">视频</span>';
                return '';
            }).join(' ');
        }
        return '<div class="msg-bubble ' + roleClass + '">' +
            '<span class="badge bg-' + roleBadge + ' me-1" style="font-size:0.65rem;">' + esc(role) + '</span>' +
            '<span style="font-size:0.85rem;">' + msgContent + '</span>' +
        '</div>';
    }).join('');
}

function renderBatchContextCard(ctx) {
    var TYPE_COLORS = {
        event: 'warning', knowledge: 'purple',
        document: 'success', profile: 'primary', agent_profile: 'primary',
        daily_summary: 'info', weekly_summary: 'info', monthly_summary: 'info'
    };
    var color = TYPE_COLORS[ctx.context_type] || 'secondary';
    var cssClass = 'batch-ctx-card-' + (ctx.context_type || 'event');
    var keywords = (ctx.keywords || []).map(function (k) {
        return '<span class="badge bg-info bg-opacity-75 me-1">' + esc(k) + '</span>';
    }).join('');
    return '<div class="batch-ctx-card ' + cssClass + '">' +
        '<div class="d-flex justify-content-between align-items-start">' +
            '<div>' +
                '<span class="badge bg-' + color + '" style="font-size:0.7rem;">' + esc(ctx.context_type) + '</span> ' +
                '<strong>' + esc(ctx.title || '未命名') + '</strong>' +
            '</div>' +
            (ctx.importance != null ? '<span class="badge bg-light text-dark">重要性: ' + ctx.importance + '</span>' : '') +
        '</div>' +
        (ctx.summary ? '<p class="text-muted small mb-1 mt-1">' + esc(ctx.summary) + '</p>' : '') +
        (keywords ? '<div class="mt-1">' + keywords + '</div>' : '') +
    '</div>';
}

function renderBatchPagination(page, totalPages, total) {
    var area = document.getElementById('batchPaginationArea');
    if (!area) return;
    if (totalPages <= 1) {
        area.innerHTML = '<div class="text-muted small text-center">共 ' + total + ' 条</div>';
        return;
    }

    var btns = '';
    btns += '<button class="btn btn-sm btn-outline-dark me-1" ' + (page <= 1 ? 'disabled' : '') + ' onclick="loadBatches(' + (page - 1) + ')">&laquo;</button>';
    for (var p = Math.max(1, page - 2); p <= Math.min(totalPages, page + 2); p++) {
        btns += '<button class="btn btn-sm ' + (p === page ? 'btn-dark' : 'btn-outline-dark') + ' me-1" onclick="loadBatches(' + p + ')">' + p + '</button>';
    }
    btns += '<button class="btn btn-sm btn-outline-dark" ' + (page >= totalPages ? 'disabled' : '') + ' onclick="loadBatches(' + (page + 1) + ')">&raquo;</button>';

    area.innerHTML = '<div class="d-flex justify-content-between align-items-center">' +
        '<small class="text-muted">共 ' + total + ' 条，第 ' + page + '/' + totalPages + ' 页</small>' +
        '<div>' + btns + '</div>' +
    '</div>';
}

/* ============================================================
   Tab 3: Related Contexts (关联 Contexts)
   ============================================================ */

let ctxPage = 1;
const ctxLimit = 15;

function initContextsTab() {
    var container = document.getElementById('tab-contexts');
    if (!container) return;

    container.innerHTML =
        // Filter bar
        '<div class="card mb-3" style="border-top: 3px solid #2c3e50;">' +
            '<div class="card-body py-3 px-3">' +
                '<div class="row g-2 align-items-end">' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Type</label>' +
                        '<select class="form-select form-select-sm" id="ctxFilterType">' +
                            '<option value="">全部</option>' +
                        '</select>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">User ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="ctxFilterUserId" placeholder="可选">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Device ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="ctxFilterDeviceId" placeholder="可选">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Agent ID</label>' +
                        '<input type="text" class="form-control form-control-sm" id="ctxFilterAgentId" value="' + escAttr(selectedAgentId || '') + '" disabled>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">层级</label>' +
                        '<select class="form-select form-select-sm" id="ctxFilterLevel">' +
                            '<option value="">全部</option>' +
                            '<option value="0">L0 原始事件</option>' +
                            '<option value="1">L1 日摘要</option>' +
                            '<option value="2">L2 周摘要</option>' +
                            '<option value="3">L3 月摘要</option>' +
                        '</select>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">开始时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="ctxFilterStartDate">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">结束时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="ctxFilterEndDate">' +
                    '</div>' +
                    '<div class="col-12 col-md-auto">' +
                        '<div class="d-flex gap-2">' +
                            '<button class="btn btn-sm btn-dark" onclick="loadAgentContexts(1)">筛选</button>' +
                            '<button class="btn btn-sm btn-outline-secondary" onclick="resetCtxFilters()">重置</button>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +
        // Card grid
        '<div id="ctxCardGrid" class="row g-3"></div>' +
        // Pagination
        '<div id="ctxPaginationArea" class="mt-3"></div>';

    // Populate type dropdown from API
    populateCtxTypeDropdown();
    loadAgentContexts(1);
}

async function populateCtxTypeDropdown() {
    var select = document.getElementById('ctxFilterType');
    if (!select) return;
    try {
        var resp = await fetch('/api/contexts?limit=1&agent_id=' + encodeURIComponent(selectedAgentId));
        var data = await resp.json();
        if (resp.ok && data.data && data.data.context_types) {
            var types = data.data.context_types;
            // Keep the "all" option, then add each type
            var html = '<option value="">全部</option>';
            for (var i = 0; i < types.length; i++) {
                html += '<option value="' + escAttr(types[i]) + '">' + esc(types[i]) + '</option>';
            }
            select.innerHTML = html;
        }
    } catch (e) {
        // Silently fail — dropdown just stays at default "all"
    }
}

function resetCtxFilters() {
    var type = document.getElementById('ctxFilterType');
    var uid = document.getElementById('ctxFilterUserId');
    var did = document.getElementById('ctxFilterDeviceId');
    var level = document.getElementById('ctxFilterLevel');
    var sd = document.getElementById('ctxFilterStartDate');
    var ed = document.getElementById('ctxFilterEndDate');
    if (type) type.value = '';
    if (uid) uid.value = '';
    if (did) did.value = '';
    if (level) level.value = '';
    if (sd) sd.value = '';
    if (ed) ed.value = '';
    loadAgentContexts(1);
}

function getCtxFilterParams() {
    var params = new URLSearchParams();
    params.set('agent_id', selectedAgentId);
    var type = document.getElementById('ctxFilterType');
    var userId = document.getElementById('ctxFilterUserId');
    var deviceId = document.getElementById('ctxFilterDeviceId');
    var level = document.getElementById('ctxFilterLevel');
    var startDate = document.getElementById('ctxFilterStartDate');
    var endDate = document.getElementById('ctxFilterEndDate');
    if (type && type.value) params.set('type', type.value);
    if (userId && userId.value.trim()) params.set('user_id', userId.value.trim());
    if (deviceId && deviceId.value.trim()) params.set('device_id', deviceId.value.trim());
    if (level && level.value !== '') params.set('hierarchy_level', level.value);
    if (startDate && startDate.value) params.set('start_date', startDate.value);
    if (endDate && endDate.value) params.set('end_date', endDate.value);
    return params;
}

async function loadAgentContexts(page) {
    ctxPage = page;
    var params = getCtxFilterParams();
    params.set('page', page);
    params.set('limit', ctxLimit);

    var grid = document.getElementById('ctxCardGrid');
    var pagArea = document.getElementById('ctxPaginationArea');
    if (!grid) return;
    grid.innerHTML = '<div class="col-12 text-center py-4"><div class="spinner-border spinner-border-sm"></div> 加载中...</div>';
    if (pagArea) pagArea.innerHTML = '';

    try {
        var resp = await fetch('/api/contexts?' + params);
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.message || '加载失败');

        var contexts = data.data.contexts || [];
        var total = data.data.total || 0;
        var totalPages = data.data.total_pages || 1;

        if (contexts.length === 0) {
            grid.innerHTML = '<div class="col-12 text-center text-muted py-5">没有找到 Contexts</div>';
            return;
        }

        grid.innerHTML = contexts.map(function (ctx) {
            return '<div class="col-md-6 col-lg-4">' + renderContextCardHTML(ctx) + '</div>';
        }).join('');

        if (pagArea) {
            pagArea.innerHTML = renderPaginationHTML(page, totalPages, total, ctxLimit, 'loadAgentContexts');
        }

        refreshIcons();
    } catch (e) {
        grid.innerHTML = '<div class="col-12 text-center text-danger py-4">加载失败: ' + esc(e.message) + '</div>';
    }
}
