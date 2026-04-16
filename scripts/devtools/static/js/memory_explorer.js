/**
 * MineContext DevTools Console — Memory Explorer
 *
 * User list management and Tab 1: Memory Snapshot (ported from memory_cache.html).
 * Tab 2 (Semantic Search) and Tab 3 (Contexts List) are added in Task 8.
 *
 * Dependencies (loaded by base.html before this file):
 *   - Bootstrap 5
 *   - feather-icons
 *   - shared.js  (esc, escAttr, refreshIcons, wrapCard, TYPE_COLORS, typeBadgeClass,
 *                  formatDatetime, formatDateShort)
 */

/* ============================================================
   State
   ============================================================ */

let selectedUser = null;  // { user_id, device_id, agent_id }
let _allUsers = [];

/* ============================================================
   Init
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {
    loadUserList();

    // Tab-switch listener skeleton (Tab 2 & 3 filled in Task 8)
    var tabEl = document.getElementById('memoryTabs');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            if (!selectedUser) return;
            // var target = e.target.getAttribute('data-bs-target');
            // if (target === '#tab-search') initSearchTab();
            // if (target === '#tab-mcontexts') initContextsTab();
        });
    }
});

/* ============================================================
   User List
   ============================================================ */

async function loadUserList() {
    try {
        var resp = await fetch('/api/users');
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to load users');
        var users = (data.data && data.data.users) || [];
        _allUsers = users;
        renderUserList(users);

        // Auto-select deep-linked user
        var params = new URLSearchParams(window.location.search);
        var linkedUserId = params.get('user_id');
        var linkedDeviceId = params.get('device_id');
        var linkedAgentId = params.get('agent_id');
        if (linkedUserId) {
            var found = users.find(function (u) {
                return u.user_id === linkedUserId &&
                    (linkedDeviceId ? u.device_id === linkedDeviceId : true) &&
                    (linkedAgentId ? u.agent_id === linkedAgentId : true);
            });
            if (found) {
                selectUser(found);
            }
        }
    } catch (e) {
        console.error('Load users failed:', e);
        var list = document.getElementById('userList');
        if (list) {
            list.innerHTML = '<div class="text-danger text-center py-3"><small>加载失败</small></div>';
        }
    }
}

function renderUserList(users) {
    var list = document.getElementById('userList');
    var countEl = document.getElementById('userCount');

    if (users.length === 0) {
        list.innerHTML = '<div class="text-muted text-center py-4"><small>暂无用户</small></div>';
        countEl.textContent = '0 个用户';
        return;
    }

    var html = '';
    users.forEach(function (u) {
        var isActive = selectedUser &&
            selectedUser.user_id === u.user_id &&
            selectedUser.device_id === u.device_id &&
            selectedUser.agent_id === u.agent_id;
        var key = escAttr(u.user_id) + '|' + escAttr(u.device_id) + '|' + escAttr(u.agent_id);
        html += '<div class="sidebar-item' + (isActive ? ' active' : '') + '" ' +
            'onclick="selectUserByKey(\'' + key + '\')" ' +
            'data-user-key="' + key + '" ' +
            'title="' + escAttr(u.user_id) + '">' +
            '<div style="overflow:hidden;text-overflow:ellipsis;">' + esc(u.user_id) + '</div>' +
            '<small class="text-muted" style="font-size:0.72rem;">' +
            esc(u.device_id) + ' / ' + esc(u.agent_id) + '</small>' +
            '</div>';
    });

    list.innerHTML = html;
    countEl.textContent = users.length + ' 个用户';
}

function filterUserList(query) {
    if (!query) {
        renderUserList(_allUsers);
        return;
    }
    var q = query.toLowerCase();
    var filtered = _allUsers.filter(function (u) {
        return u.user_id.toLowerCase().indexOf(q) !== -1;
    });
    renderUserList(filtered);
}

function selectUserByKey(key) {
    var parts = key.split('|');
    var user = { user_id: parts[0], device_id: parts[1], agent_id: parts[2] };
    selectUser(user);
}

function selectUser(user) {
    selectedUser = user;

    // Highlight active sidebar item
    document.querySelectorAll('#userList .sidebar-item').forEach(function (el) {
        var k = el.getAttribute('data-user-key');
        var expected = user.user_id + '|' + user.device_id + '|' + user.agent_id;
        if (k === expected) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Update URL
    var params = new URLSearchParams();
    params.set('user_id', user.user_id);
    if (user.device_id && user.device_id !== 'default') params.set('device_id', user.device_id);
    if (user.agent_id && user.agent_id !== 'default') params.set('agent_id', user.agent_id);
    history.replaceState(null, '', window.location.pathname + '?' + params.toString());

    // Load snapshot (Tab 1)
    loadSnapshot();
}

/* ============================================================
   Tab 1: Memory Snapshot
   ============================================================ */

function loadSnapshot() {
    if (!selectedUser) return;

    var pane = document.getElementById('tab-snapshot');
    pane.innerHTML =
        '<div class="snapshot-config-bar">' +
            '<form id="snapshotForm" onsubmit="handleSnapshotSubmit(event)">' +
                '<div class="d-flex flex-wrap align-items-end gap-3">' +
                    // Include sections
                    '<div>' +
                        '<label class="form-label mb-1">返回内容</label><br>' +
                        '<div class="form-check form-check-inline">' +
                            '<input class="form-check-input" type="checkbox" id="snapIncludeAgentPrompt" checked>' +
                            '<label class="form-check-label" for="snapIncludeAgentPrompt">Agent Prompt</label>' +
                        '</div>' +
                        '<div class="form-check form-check-inline">' +
                            '<input class="form-check-input" type="checkbox" id="snapIncludeProfile" checked>' +
                            '<label class="form-check-label" for="snapIncludeProfile">User Profile</label>' +
                        '</div>' +
                        '<div class="form-check form-check-inline">' +
                            '<input class="form-check-input" type="checkbox" id="snapIncludeEvents" checked>' +
                            '<label class="form-check-label" for="snapIncludeEvents">Events</label>' +
                        '</div>' +
                        '<div class="form-check form-check-inline">' +
                            '<input class="form-check-input" type="checkbox" id="snapIncludeAccessed" checked>' +
                            '<label class="form-check-label" for="snapIncludeAccessed">Recently Accessed</label>' +
                        '</div>' +
                    '</div>' +
                    // Numeric inputs
                    '<div>' +
                        '<label class="form-label mb-1" for="snapRecentDays">Recent Days</label>' +
                        '<input type="number" class="form-control form-control-sm" id="snapRecentDays" value="3" min="1" max="90">' +
                    '</div>' +
                    '<div>' +
                        '<label class="form-label mb-1" for="snapMaxTodayEvents">Max Today Events</label>' +
                        '<input type="number" class="form-control form-control-sm" id="snapMaxTodayEvents" value="5" min="1" max="100">' +
                    '</div>' +
                    '<div>' +
                        '<label class="form-label mb-1" for="snapMaxAccessed">Max Accessed</label>' +
                        '<input type="number" class="form-control form-control-sm" id="snapMaxAccessed" value="5" min="1" max="100">' +
                    '</div>' +
                    // Force refresh
                    '<div>' +
                        '<div class="form-check" style="margin-top:22px;">' +
                            '<input class="form-check-input" type="checkbox" id="snapForceRefresh">' +
                            '<label class="form-check-label" for="snapForceRefresh">Force Refresh</label>' +
                        '</div>' +
                    '</div>' +
                    // Submit button
                    '<div>' +
                        '<button type="submit" class="btn btn-primary btn-sm" id="snapQueryBtn">' +
                            '<span class="spinner-border spinner-border-sm d-none" id="snapSpinner"></span> ' +
                            '<span id="snapBtnText">查询</span>' +
                        '</button>' +
                    '</div>' +
                '</div>' +
            '</form>' +
        '</div>' +
        '<div id="snapshotResults">' +
            '<div class="text-muted text-center py-4">' +
                '<span class="spinner-border spinner-border-sm"></span> 加载中...' +
            '</div>' +
        '</div>';

    refreshIcons();

    // Auto-query on load
    querySnapshot();
}

function handleSnapshotSubmit(event) {
    event.preventDefault();
    querySnapshot();
}

async function querySnapshot() {
    if (!selectedUser) return;

    setSnapshotLoading(true);

    try {
        // Build include parameter from checkboxes
        var includeSections = [];
        if (document.getElementById('snapIncludeProfile').checked) includeSections.push('profile');
        if (document.getElementById('snapIncludeEvents').checked) includeSections.push('events');
        if (document.getElementById('snapIncludeAccessed').checked) includeSections.push('accessed');
        if (document.getElementById('snapIncludeAgentPrompt').checked) includeSections.push('agent_prompt');

        var recentDays = document.getElementById('snapRecentDays').value || 3;
        var maxTodayEvents = document.getElementById('snapMaxTodayEvents').value || 5;
        var maxAccessed = document.getElementById('snapMaxAccessed').value || 5;
        var forceRefresh = document.getElementById('snapForceRefresh').checked;

        var params = new URLSearchParams({
            user_id: selectedUser.user_id,
            device_id: selectedUser.device_id || 'default',
            agent_id: selectedUser.agent_id || 'default',
            recent_days: recentDays,
            max_recent_events_today: maxTodayEvents,
            max_accessed: maxAccessed,
            force_refresh: forceRefresh
        });

        // Only add include param if not all sections selected
        if (includeSections.length > 0 && includeSections.length < 4) {
            params.set('include', includeSections.join(','));
        }

        var response = await fetch('/api/memory-cache?' + params);
        var data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || data.detail || '查询失败');
        }

        renderSnapshotResults(data);
    } catch (error) {
        console.error('Snapshot query error:', error);
        var resultsEl = document.getElementById('snapshotResults');
        if (resultsEl) {
            resultsEl.innerHTML =
                '<div class="alert alert-danger">' +
                    '<i data-feather="alert-circle" style="width:16px;height:16px;" class="me-2"></i>' +
                    '查询失败: ' + esc(error.message) +
                '</div>';
            refreshIcons();
        }
    } finally {
        setSnapshotLoading(false);
    }
}

function setSnapshotLoading(loading) {
    var spinner = document.getElementById('snapSpinner');
    var btnText = document.getElementById('snapBtnText');
    var btn = document.getElementById('snapQueryBtn');
    if (!spinner || !btnText || !btn) return;

    if (loading) {
        spinner.classList.remove('d-none');
        btnText.textContent = '查询中...';
        btn.disabled = true;
    } else {
        spinner.classList.add('d-none');
        btnText.textContent = '查询';
        btn.disabled = false;
    }
}

/* ============================================================
   Snapshot Section Renderers
   ============================================================ */

function renderSnapshotResults(data) {
    var html = '';
    if (data.agent_prompt != null) html += renderAgentPrompt(data.agent_prompt);
    if (data.profile != null) html += renderProfile(data.profile);
    if (data.recently_accessed != null) html += renderRecentlyAccessed(data.recently_accessed);
    if (data.today_events != null) html += renderTodayEvents(data.today_events);
    if (data.daily_summaries != null) html += renderDailySummaries(data.daily_summaries);

    if (!html) {
        html = '<div class="text-muted text-center py-5"><p>无返回数据</p></div>';
    }

    var resultsEl = document.getElementById('snapshotResults');
    if (resultsEl) {
        resultsEl.innerHTML = '<div class="snapshot-results">' + html + '</div>';
        refreshIcons();
    }
}

function renderProfile(profile) {
    var content;
    if (!profile) {
        content = '<p class="text-muted">暂无用户画像</p>';
    } else {
        content = '<div class="profile-display">' +
            esc(profile.factual_profile || '无内容') + '</div>';
    }
    return wrapCard('用户画像', 'user', content, 'profile-section');
}

function renderAgentPrompt(agentPrompt) {
    var factual = (agentPrompt && agentPrompt.factual_profile) || '';
    var behavioral = (agentPrompt && agentPrompt.behavioral_profile) || '';
    var blocks = [];
    if (factual) {
        blocks.push(
            '<div class="mb-2"><small class="text-muted fw-semibold">Factual Profile</small></div>' +
            '<div class="profile-display mb-2">' + esc(factual) + '</div>'
        );
    }
    if (behavioral) {
        blocks.push(
            '<div class="mb-2"><small class="text-muted fw-semibold">Behavioral Profile</small></div>' +
            '<div class="profile-display">' + esc(behavioral) + '</div>'
        );
    }
    var content = blocks.length ? blocks.join('') : '<p class="text-muted">暂无 Agent 提示词</p>';
    return wrapCard('Agent 提示词', 'cpu', content, 'agent-prompt-section');
}

function renderRecentlyAccessed(items) {
    var content;
    if (!items || items.length === 0) {
        content = '<p class="text-muted">暂无最近访问记录</p>';
    } else {
        content = items.map(function (item) {
            var accessTime = item.accessed_ts
                ? new Date(item.accessed_ts * 1000).toLocaleString('zh-CN')
                : '';
            var color = TYPE_COLORS[item.context_type] || 'secondary';
            var badgeCls = typeBadgeClass(item.context_type);

            // Media thumbnails
            var mediaHtml = '';
            var refs = (item.metadata && item.metadata.media_refs) || item.media_refs || [];
            if (refs.length > 0) {
                mediaHtml = '<div class="mt-1">' + refs.map(function (ref) {
                    if (ref.type === 'image') {
                        return '<a href="' + escAttr(ref.url) + '" target="_blank" rel="noopener">' +
                            '<img src="' + escAttr(ref.url) + '" ' +
                            'style="width:40px;height:40px;object-fit:cover;border-radius:3px;border:1px solid #dee2e6;margin-right:4px;" ' +
                            'loading="lazy" onerror="this.style.display=\'none\'"></a>';
                    }
                    if (ref.type === 'video') {
                        return '<a href="' + escAttr(ref.url) + '" target="_blank" rel="noopener" ' +
                            'class="badge bg-dark me-1">&#9654; 视频</a>';
                    }
                    return '';
                }).join('') + '</div>';
            }

            return '<div class="accessed-item d-flex justify-content-between align-items-start">' +
                '<div class="flex-grow-1">' +
                    '<div>' +
                        '<span class="badge ' + badgeCls + ' me-1">' + esc(item.context_type) + '</span>' +
                        '<span class="fw-semibold">' + esc(item.title || '未命名') + '</span>' +
                    '</div>' +
                    (item.summary
                        ? '<small class="text-muted">' +
                          esc(item.summary.length > 100 ? item.summary.substring(0, 100) + '...' : item.summary) +
                          '</small>'
                        : '') +
                    mediaHtml +
                '</div>' +
                '<div class="text-end ms-2" style="min-width:100px;">' +
                    (item.score != null
                        ? '<span class="badge bg-primary">' + item.score.toFixed(3) + '</span>'
                        : '') +
                    '<div><small class="text-muted">' + esc(accessTime) + '</small></div>' +
                '</div>' +
            '</div>';
        }).join('');
    }

    var count = (items && items.length) || 0;
    return wrapCard('最近访问 (' + count + ')', 'clock', content, 'accessed-section');
}

function renderTodayEvents(events) {
    var content;
    if (!events || events.length === 0) {
        content = '<p class="text-muted">今日暂无事件</p>';
    } else {
        content = events.map(function (evt) {
            var timeStr = '';
            if (evt.event_time_start) {
                var d = new Date(evt.event_time_start);
                var hh = String(d.getHours()).padStart(2, '0');
                var mm = String(d.getMinutes()).padStart(2, '0');
                timeStr = '<span class="badge bg-light text-dark me-1">' + hh + ':' + mm + '</span>';
            }
            return '<div class="event-timeline-item">' +
                '<div class="fw-semibold">' + timeStr + esc(evt.title || '未命名') + '</div>' +
                (evt.summary ? '<small class="text-muted">' + esc(evt.summary) + '</small>' : '') +
            '</div>';
        }).join('');
    }

    var count = (events && events.length) || 0;
    return wrapCard('今日事件 (' + count + ')', 'zap', content, 'today-section');
}

function renderDailySummaries(summaries) {
    var content;
    if (!summaries || summaries.length === 0) {
        content = '<p class="text-muted">暂无历史日摘要</p>';
    } else {
        content = summaries.map(function (s) {
            return '<div class="summary-item">' +
                '<div class="mb-1">' +
                    '<span class="badge bg-success">' + esc(s.event_time_start) + '</span>' +
                    (s.title ? ' <span class="fw-semibold">' + esc(s.title) + '</span>' : '') +
                '</div>' +
                (s.summary
                    ? '<small>' + esc(s.summary) + '</small>'
                    : '<small class="text-muted">暂无摘要</small>') +
            '</div>';
        }).join('');
    }

    var count = (summaries && summaries.length) || 0;
    return wrapCard('历史日摘要 (' + count + ')', 'calendar', content, 'summaries-section');
}
