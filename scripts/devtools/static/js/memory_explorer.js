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

// Tab 2: Semantic Search state
let uploadedMediaList = [];  // [{ url, type: 'image'|'video', filename }, ...]

// Tab 3: Contexts List state
let mCtxPage = 1;
const mCtxLimit = 15;

/* ============================================================
   Init
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {
    loadUserList();

    // Tab-switch listener
    var tabEl = document.getElementById('memoryTabs');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            if (!selectedUser) return;
            var target = e.target.getAttribute('data-bs-target');
            if (target === '#tab-search') initSearchTab();
            if (target === '#tab-mcontexts') initMemoryContextsTab();
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

    // Load active tab content
    var activeTab = document.querySelector('#memoryTabs .nav-link.active');
    var activeTarget = activeTab ? activeTab.getAttribute('data-bs-target') : '';
    if (activeTarget === '#tab-search') {
        initSearchTab();
    } else if (activeTarget === '#tab-mcontexts') {
        initMemoryContextsTab();
    } else {
        loadSnapshot();
    }
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

/* ============================================================
   Tab 2: Semantic Search
   ============================================================ */

function initSearchTab() {
    var container = document.getElementById('tab-search');
    if (!container || !selectedUser) return;

    // Reset media list on re-init
    uploadedMediaList = [];

    container.innerHTML =
        '<div class="row">' +
            // Left column: search parameters
            '<div class="col-lg-5 col-xl-4">' +
                '<div class="card search-params-card">' +
                    '<div class="card-header"><h6 class="mb-0">' +
                        '<i data-feather="sliders" style="width:16px;height:16px;" class="me-2"></i>搜索参数</h6>' +
                    '</div>' +
                    '<div class="card-body">' +
                        '<form id="memorySearchForm">' +
                            // Query textarea
                            '<div class="mb-3">' +
                                '<label for="mSearchQuery" class="form-label">查询内容</label>' +
                                '<textarea class="form-control" id="mSearchQuery" rows="3" maxlength="2000" placeholder="输入语义搜索内容..."></textarea>' +
                                '<div class="form-text">支持自然语言查询，可与图片组合进行多模态搜索</div>' +
                            '</div>' +
                            // Media upload zone
                            '<div class="mb-3">' +
                                '<label class="form-label">搜索媒体（可选）</label>' +
                                '<div id="mMediaPasteZone" class="paste-zone" tabindex="0">' +
                                    '<div id="mMediaPlaceholder" class="paste-placeholder">' +
                                        '<i data-feather="upload" style="width:24px;height:24px;" class="mb-1"></i>' +
                                        '<div>粘贴图片 / 拖放文件 / <a href="#" id="mMediaFileLink" class="text-decoration-none">选择文件</a></div>' +
                                        '<small class="text-muted">支持图片和视频（jpg, png, gif, mp4, mov...）</small>' +
                                    '</div>' +
                                    '<div id="mMediaPreview" class="d-none" style="width:100%;">' +
                                        '<div id="mMediaPreviewContent" class="d-flex flex-wrap gap-2 align-items-start"></div>' +
                                    '</div>' +
                                    '<input type="file" id="mMediaFileInput" class="d-none" accept="image/*,video/*" multiple>' +
                                '</div>' +
                                '<div id="mMediaUploadProgress" class="progress mt-1 d-none" style="height:4px;">' +
                                    '<div class="progress-bar" role="progressbar" style="width:0%"></div>' +
                                '</div>' +
                                '<div class="mt-2">' +
                                    '<div class="input-group input-group-sm">' +
                                        '<span class="input-group-text" style="font-size:0.8rem;">视频 URL</span>' +
                                        '<input type="text" class="form-control" id="mVideoUrlInput" placeholder="https://... (已有视频链接可直接粘贴)">' +
                                        '<button class="btn btn-outline-secondary" type="button" id="mVideoUrlConfirmBtn" style="font-size:0.8rem;">确认</button>' +
                                    '</div>' +
                                '</div>' +
                            '</div>' +
                            // Event ID lookup
                            '<div class="mb-3">' +
                                '<label for="mEventIds" class="form-label">或：事件ID精确查找</label>' +
                                '<textarea class="form-control" id="mEventIds" rows="2" placeholder="每行一个事件ID"></textarea>' +
                                '<div class="form-text">填写后优先走ID查找，忽略上方的查询内容和媒体</div>' +
                            '</div>' +
                            '<hr class="my-3">' +
                            // Top-K
                            '<div class="mb-3">' +
                                '<label for="mTopK" class="form-label">返回数量 (Top-K)</label>' +
                                '<input type="number" class="form-control" id="mTopK" value="10" min="1" max="100">' +
                            '</div>' +
                            // Hierarchy levels
                            '<div class="mb-3">' +
                                '<label class="form-label">层级过滤</label>' +
                                '<div>' +
                                    '<div class="form-check">' +
                                        '<input class="form-check-input m-hierarchy-level-cb" type="checkbox" value="0" id="mLevel0">' +
                                        '<label class="form-check-label" for="mLevel0">L0 原始事件</label>' +
                                    '</div>' +
                                    '<div class="form-check">' +
                                        '<input class="form-check-input m-hierarchy-level-cb" type="checkbox" value="1" id="mLevel1">' +
                                        '<label class="form-check-label" for="mLevel1">L1 日总结</label>' +
                                    '</div>' +
                                    '<div class="form-check">' +
                                        '<input class="form-check-input m-hierarchy-level-cb" type="checkbox" value="2" id="mLevel2">' +
                                        '<label class="form-check-label" for="mLevel2">L2 周总结</label>' +
                                    '</div>' +
                                    '<div class="form-check">' +
                                        '<input class="form-check-input m-hierarchy-level-cb" type="checkbox" value="3" id="mLevel3">' +
                                        '<label class="form-check-label" for="mLevel3">L3 月总结</label>' +
                                    '</div>' +
                                '</div>' +
                                '<div class="form-text">不选则不过滤层级</div>' +
                            '</div>' +
                            // Drill direction
                            '<div class="mb-3">' +
                                '<label for="mDrill" class="form-label">层级展开方向</label>' +
                                '<select class="form-select" id="mDrill">' +
                                    '<option value="up" selected>向上（祖先链）</option>' +
                                    '<option value="down">向下（子事件）</option>' +
                                    '<option value="both">双向（祖先 + 子事件）</option>' +
                                    '<option value="none">不展开</option>' +
                                '</select>' +
                            '</div>' +
                            // Start/end time
                            '<div class="mb-3">' +
                                '<label for="mStartTime" class="form-label">开始时间</label>' +
                                '<input type="datetime-local" class="form-control" id="mStartTime">' +
                            '</div>' +
                            '<div class="mb-3">' +
                                '<label for="mEndTime" class="form-label">结束时间</label>' +
                                '<input type="datetime-local" class="form-control" id="mEndTime">' +
                            '</div>' +
                            // User triple (locked)
                            '<div class="mb-3">' +
                                '<label class="form-label">User ID</label>' +
                                '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.user_id) + '">' +
                            '</div>' +
                            '<div class="mb-3">' +
                                '<label class="form-label">Device ID</label>' +
                                '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.device_id || 'default') + '">' +
                            '</div>' +
                            '<div class="mb-3">' +
                                '<label class="form-label">Agent ID</label>' +
                                '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.agent_id || 'default') + '">' +
                            '</div>' +
                            // Submit
                            '<button type="submit" class="btn btn-primary w-100">' +
                                '<span class="spinner-border spinner-border-sm d-none" role="status" id="mSearchSpinner"></span> ' +
                                '<span id="mSearchBtnText">开始搜索</span>' +
                            '</button>' +
                        '</form>' +
                    '</div>' +
                '</div>' +
            '</div>' +
            // Right column: results
            '<div class="col-lg-7 col-xl-8">' +
                '<div class="card">' +
                    '<div class="card-header d-flex justify-content-between align-items-center">' +
                        '<h6 class="mb-0"><i data-feather="list" style="width:16px;height:16px;" class="me-2"></i>搜索结果</h6>' +
                        '<span class="badge bg-secondary" id="mResultCount">0 条命中</span>' +
                    '</div>' +
                    '<div class="card-body">' +
                        '<div id="mSearchMetadata" class="d-none mb-3"></div>' +
                        '<div id="mSearchResults">' +
                            '<div class="text-muted text-center py-5">' +
                                '<i data-feather="search" class="mb-3" style="width:48px;height:48px;"></i>' +
                                '<p>输入查询内容、粘贴图片或选择过滤条件，然后点击"开始搜索"</p>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>';

    refreshIcons();

    // Bind search form submit
    document.getElementById('memorySearchForm').addEventListener('submit', handleSearch);

    // Initialize media zone interactions
    initMediaZone();
}

/* ---- Media Upload Zone ---- */

function initMediaZone() {
    var zone = document.getElementById('mMediaPasteZone');
    var placeholder = document.getElementById('mMediaPlaceholder');
    var preview = document.getElementById('mMediaPreview');
    var previewContent = document.getElementById('mMediaPreviewContent');
    var fileInput = document.getElementById('mMediaFileInput');
    var fileLink = document.getElementById('mMediaFileLink');
    var progressBar = document.getElementById('mMediaUploadProgress');
    var videoUrlInput = document.getElementById('mVideoUrlInput');
    var videoUrlConfirmBtn = document.getElementById('mVideoUrlConfirmBtn');

    if (!zone) return;

    // Click "选择文件" link opens file dialog
    fileLink.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    // Click zone opens file dialog
    zone.addEventListener('click', function (e) {
        if (e.target === zone || e.target.closest('.paste-placeholder')) {
            fileInput.click();
        }
    });

    // File input change (supports multiple files)
    fileInput.addEventListener('change', function () {
        for (var i = 0; i < fileInput.files.length; i++) {
            uploadFile(fileInput.files[i]);
        }
        fileInput.value = '';
    });

    // Paste on form level
    document.getElementById('memorySearchForm').addEventListener('paste', function (e) {
        if (document.activeElement === videoUrlInput) return;
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            if (item.kind === 'file' && (item.type.startsWith('image/') || item.type.startsWith('video/'))) {
                var f = item.getAsFile();
                if (!f) continue;
                e.preventDefault();
                uploadFile(f);
                return;
            }
        }
        // Fallback: check clipboardData.files
        var files = e.clipboardData.files;
        if (files && files.length > 0) {
            var ff = files[0];
            var ext = (ff.name || '').split('.').pop().toLowerCase();
            if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'webm'].indexOf(ext) !== -1) {
                e.preventDefault();
                uploadFile(ff);
            }
        }
    });

    // Drag and drop
    zone.addEventListener('dragover', function (e) { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', function () { zone.classList.remove('drag-over'); });
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        var files = e.dataTransfer.files;
        if (files.length > 0 && (files[0].type.startsWith('image/') || files[0].type.startsWith('video/'))) {
            uploadFile(files[0]);
        }
    });

    // Video URL confirm
    videoUrlConfirmBtn.addEventListener('click', function () {
        var url = videoUrlInput.value.trim();
        if (!url) return;
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            alert('请输入有效的视频 URL（以 http:// 或 https:// 开头）');
            return;
        }
        addMedia({ url: url, type: 'video', filename: url.split('/').pop() || 'video' });
        videoUrlInput.value = '';
    });
    videoUrlInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); videoUrlConfirmBtn.click(); }
    });

    function uploadFile(file) {
        if (!file) {
            alert('无法读取剪贴板文件，请使用文件选择或拖拽方式上传');
            return;
        }
        zone.classList.add('uploading');
        progressBar.classList.remove('d-none');
        progressBar.querySelector('.progress-bar').style.width = '30%';

        var formData = new FormData();
        formData.append('file', file);

        progressBar.querySelector('.progress-bar').style.width = '60%';

        fetch('/api/media/upload', {
            method: 'POST',
            body: formData
        }).then(function (resp) {
            return resp.json().then(function (data) {
                progressBar.querySelector('.progress-bar').style.width = '100%';
                if (!resp.ok) {
                    var msg = data.error || (Array.isArray(data.detail) ? data.detail.map(function (d) { return d.msg; }).join('; ') : data.detail) || '上传失败';
                    throw new Error(msg);
                }
                addMedia(data);
            });
        }).catch(function (err) {
            console.error('Upload failed:', err);
            alert('媒体上传失败: ' + err.message);
        }).finally(function () {
            zone.classList.remove('uploading');
            setTimeout(function () {
                progressBar.classList.add('d-none');
                progressBar.querySelector('.progress-bar').style.width = '0%';
            }, 500);
        });
    }

    function addMedia(media) {
        if (uploadedMediaList.length >= 10) {
            alert('最多支持 10 个媒体文件');
            return;
        }
        uploadedMediaList.push(media);
        renderMediaPreview();
    }

    function removeMedia(index) {
        uploadedMediaList.splice(index, 1);
        renderMediaPreview();
    }

    function clearMedia() {
        uploadedMediaList = [];
        renderMediaPreview();
    }

    function renderMediaPreview() {
        if (uploadedMediaList.length === 0) {
            previewContent.innerHTML = '';
            preview.classList.add('d-none');
            placeholder.classList.remove('d-none');
            zone.classList.remove('has-media');
            return;
        }
        preview.classList.remove('d-none');
        placeholder.classList.add('d-none');
        zone.classList.add('has-media');
        var items = uploadedMediaList.map(function (m, i) {
            var removeHtml = '<button type="button" class="btn-close" style="position:absolute;top:-4px;right:-4px;font-size:0.5rem;background-color:#fff;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,0.2);padding:4px;" data-remove-idx="' + i + '" aria-label="移除"></button>';
            if (m.type === 'image') {
                return '<div style="position:relative;display:inline-block;"><img src="' + esc(m.url) + '" class="paste-preview-img" alt="preview">' + removeHtml + '</div>';
            }
            return '<div style="position:relative;display:inline-flex;"><div class="paste-video-info"><i data-feather="film" style="width:16px;height:16px;"></i><span style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(m.filename) + '</span></div>' + removeHtml + '</div>';
        }).join('');
        var clearAllHtml = uploadedMediaList.length > 1
            ? '<button type="button" class="btn btn-sm btn-outline-secondary" id="mMediaClearAllBtn" style="font-size:0.7rem;padding:2px 8px;align-self:center;">全部清除</button>'
            : '';
        previewContent.innerHTML = items + clearAllHtml;
        refreshIcons();
        // Bind remove buttons
        previewContent.querySelectorAll('[data-remove-idx]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                removeMedia(parseInt(this.dataset.removeIdx));
            });
        });
        var clearAllBtn = document.getElementById('mMediaClearAllBtn');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                clearMedia();
            });
        }
    }
}

/* ---- Search Execution ---- */

async function handleSearch(event) {
    event.preventDefault();

    var query = document.getElementById('mSearchQuery').value.trim();
    var topK = parseInt(document.getElementById('mTopK').value) || 10;
    var drill = document.getElementById('mDrill').value;
    var startTime = document.getElementById('mStartTime').value;
    var endTime = document.getElementById('mEndTime').value;

    var eventIdsRaw = document.getElementById('mEventIds').value.trim();
    var eventIds = eventIdsRaw ? eventIdsRaw.split(/\n/).map(function (s) { return s.trim(); }).filter(Boolean) : [];

    var selectedLevels = Array.from(document.querySelectorAll('.m-hierarchy-level-cb:checked'))
        .map(function (cb) { return parseInt(cb.value); });

    var hasQuery = query.length > 0;
    var hasMedia = uploadedMediaList.length > 0;
    var hasLevels = selectedLevels.length > 0;
    var hasTimeRange = startTime || endTime;
    var hasEventIds = eventIds.length > 0;

    if (!hasQuery && !hasMedia && !hasLevels && !hasTimeRange && !hasEventIds) {
        displaySearchError('请至少输入查询内容、事件ID、选择层级过滤或设置时间范围');
        return;
    }

    setSearchLoading(true);

    try {
        var requestBody = {
            top_k: topK,
            drill: drill
        };

        if (hasEventIds) {
            requestBody.event_ids = eventIds;
        } else if (hasQuery || hasMedia) {
            var queryParts = [];
            if (hasQuery) queryParts.push({ type: 'text', text: query });
            for (var i = 0; i < uploadedMediaList.length; i++) {
                var media = uploadedMediaList[i];
                if (media.type === 'image') {
                    queryParts.push({ type: 'image_url', image_url: { url: media.url } });
                } else {
                    queryParts.push({ type: 'video_url', video_url: { url: media.url, fps: 1.0 } });
                }
            }
            requestBody.query = queryParts;
        }
        if (hasLevels) requestBody.hierarchy_levels = selectedLevels;
        requestBody.user_id = selectedUser.user_id;
        requestBody.device_id = selectedUser.device_id || 'default';
        requestBody.agent_id = selectedUser.agent_id || 'default';
        if (hasTimeRange) {
            requestBody.time_range = {};
            if (startTime) requestBody.time_range.start = Math.floor(new Date(startTime).getTime() / 1000);
            if (endTime) requestBody.time_range.end = Math.floor(new Date(endTime).getTime() / 1000);
        }

        var response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        var data = await response.json();

        if (!response.ok) {
            var detail = data.detail;
            if (Array.isArray(detail)) {
                throw new Error(detail.map(function (d) { return d.msg || JSON.stringify(d); }).join('; '));
            }
            throw new Error(detail || '搜索失败');
        }

        if (!data.success) {
            throw new Error(data.message || '搜索失败');
        }

        displaySearchMetadata(data.metadata);
        displaySearchResults(data.events);

    } catch (error) {
        console.error('Search error:', error);
        displaySearchError(error.message);
    } finally {
        setSearchLoading(false);
    }
}

function setSearchLoading(loading) {
    var spinner = document.getElementById('mSearchSpinner');
    var btnText = document.getElementById('mSearchBtnText');
    var submitBtn = document.querySelector('#memorySearchForm button[type="submit"]');
    if (!spinner || !btnText || !submitBtn) return;

    if (loading) {
        spinner.classList.remove('d-none');
        btnText.textContent = '搜索中...';
        submitBtn.disabled = true;
    } else {
        spinner.classList.add('d-none');
        btnText.textContent = '开始搜索';
        submitBtn.disabled = false;
    }
}

/* ---- Search Results Rendering ---- */

function displaySearchMetadata(metadata) {
    var container = document.getElementById('mSearchMetadata');
    if (!container) return;
    if (!metadata) {
        container.classList.add('d-none');
        return;
    }
    container.classList.remove('d-none');
    var queryText = metadata.query ? '查询: <strong>' + esc(metadata.query) + '</strong>' : '过滤搜索';
    container.innerHTML =
        '<div class="search-metadata-bar d-flex gap-3 text-muted small flex-wrap">' +
            '<span>' + queryText + '</span>' +
            '<span>命中: <strong>' + metadata.total_results + '</strong></span>' +
            '<span>耗时: <strong>' + metadata.search_time_ms.toFixed(0) + ' ms</strong></span>' +
        '</div>';
}

function countSearchNodes(nodes) {
    var hits = 0, total = 0;
    for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        total++;
        if (node.is_search_hit) hits++;
        if (node.children && node.children.length > 0) {
            var child = countSearchNodes(node.children);
            hits += child.hits;
            total += child.total;
        }
    }
    return { hits: hits, total: total };
}

function displaySearchResults(events) {
    var container = document.getElementById('mSearchResults');
    var resultCount = document.getElementById('mResultCount');
    if (!container || !resultCount) return;

    var counts = countSearchNodes(events);
    resultCount.textContent = counts.total > counts.hits
        ? counts.hits + ' 条命中 / ' + counts.total + ' 条总计'
        : counts.hits + ' 条命中';

    if (!events || events.length === 0) {
        container.innerHTML =
            '<div class="text-muted text-center py-5">' +
                '<i data-feather="inbox" class="mb-3" style="width:48px;height:48px;"></i>' +
                '<p>未找到匹配的事件</p>' +
            '</div>';
        refreshIcons();
        return;
    }

    var html = '';
    for (var i = 0; i < events.length; i++) {
        html += renderSearchTreeNode(events[i]);
    }

    container.innerHTML = html;
    refreshIcons();
}

function renderSearchTreeNode(node) {
    if (node.is_search_hit) {
        return renderSearchHitCard(node) + renderSearchChildren(node);
    } else {
        return renderSearchContextCard(node) + renderSearchChildren(node);
    }
}

function renderSearchChildren(node) {
    if (!node.children || node.children.length === 0) return '';

    var level = node.hierarchy_level || 0;
    var childrenId = 'mchildren-' + (node.id || '').replace(/[^a-zA-Z0-9]/g, '');

    var childrenHtml = '';
    for (var i = 0; i < node.children.length; i++) {
        childrenHtml += renderSearchTreeNode(node.children[i]);
    }

    return '<div class="tree-children tree-children-' + level + '" id="' + childrenId + '">' + childrenHtml + '</div>';
}

function renderSearchHitCard(event) {
    var level = event.hierarchy_level || 0;
    var title = event.title || '未命名事件';
    var summary = event.summary || '';
    var eventTimeStart = event.event_time_start ? new Date(event.event_time_start).toLocaleString('zh-CN') : '';
    var createTime = event.create_time ? new Date(event.create_time).toLocaleString('zh-CN') : '';
    var displayTime = eventTimeStart || createTime;
    var score = event.score || 0;
    var scoreColor = score >= 0.9 ? 'success' : score >= 0.7 ? 'primary' : 'secondary';
    var cardId = 'mcard-' + (event.id || '').replace(/[^a-zA-Z0-9]/g, '');

    var keywords = (event.keywords || [])
        .map(function (k) { return '<span class="badge bg-info bg-opacity-75 me-1">' + esc(k) + '</span>'; })
        .join('');

    var summaryHtml = '';
    if (summary) {
        var needsTruncate = summary.length > 200;
        var summaryId = 'msummary-' + cardId;
        summaryHtml =
            '<div class="summary-text' + (needsTruncate ? '' : ' expanded') + '" id="' + summaryId + '">' +
                '<p class="mb-0 text-muted">' + esc(summary) + '</p>' +
            '</div>' +
            (needsTruncate ? '<a href="#" class="small text-decoration-none" onclick="toggleSearchSummary(\'' + summaryId + '\', this); return false;">展开</a>' : '');
    }

    var hasChildren = event.children && event.children.length > 0;
    var childrenId = 'mchildren-' + (event.id || '').replace(/[^a-zA-Z0-9]/g, '');
    var toggleHtml = hasChildren
        ? '<span class="toggle-btn small text-muted ms-2" onclick="toggleSearchChildren(\'' + childrenId + '\', this)">' +
              '<i data-feather="chevron-down" style="width:14px;height:14px;"></i>' +
          '</span>'
        : '';

    return '<div class="border rounded mb-2 p-3 search-hit-card">' +
        '<div class="d-flex justify-content-between align-items-start mb-2">' +
            '<div class="d-flex align-items-center gap-2 flex-wrap">' +
                '<span class="badge level-badge level-badge-' + level + '">' + (LEVEL_SHORT[level] || 'L?') + '</span>' +
                '<span class="badge search-hit-badge">命中</span>' +
                '<h6 class="mb-0">' +
                    '<a href="#" onclick="viewContext(\'' + escAttr(event.id) + '\', \'' + escAttr(event.context_type || 'event') + '\'); return false;" class="text-decoration-none">' + esc(title) + '</a>' +
                '</h6>' +
                (eventTimeStart ? '<small class="text-muted">' + esc(eventTimeStart) + '</small>' : '') +
                toggleHtml +
            '</div>' +
            '<span class="badge bg-' + scoreColor + ' score-badge">' + score.toFixed(4) + '</span>' +
        '</div>' +
        summaryHtml +
        (keywords ? '<div class="mt-2">' + keywords + '</div>' : '') +
        renderMediaRefs(event.media_refs) +
        '<div class="d-flex justify-content-between align-items-center mt-2">' +
            (displayTime ? '<small class="text-muted">' + (eventTimeStart ? '事件时间' : '创建时间') + ': ' + displayTime + '</small>' : '<span></span>') +
            '<button onclick="viewContext(\'' + escAttr(event.id) + '\', \'' + escAttr(event.context_type || 'event') + '\')" class="btn btn-sm btn-outline-primary">查看详情</button>' +
        '</div>' +
    '</div>';
}

function renderSearchContextCard(node) {
    var level = node.hierarchy_level || 0;
    var title = node.title || '';
    var summary = node.summary || '';
    var eventTimeStart = node.event_time_start || '';

    var hasChildren = node.children && node.children.length > 0;
    var childrenId = 'mchildren-' + (node.id || '').replace(/[^a-zA-Z0-9]/g, '');
    var toggleHtml = hasChildren
        ? '<span class="toggle-btn" onclick="toggleSearchChildren(\'' + childrenId + '\', this)">' +
              '<i data-feather="chevron-down" style="width:14px;height:14px;"></i>' +
          '</span>'
        : '';

    var titleHtml = title ? '<strong class="ms-1">' + esc(title) + '</strong>' : '';
    var summaryHtml = summary ? '<p class="mb-0 text-muted small mt-1">' + esc(summary) + '</p>' : '';

    return '<div class="search-context-card search-context-card-' + level + ' mb-2">' +
        '<div class="d-flex align-items-center gap-2">' +
            toggleHtml +
            '<span class="badge level-badge level-badge-' + level + '">' + (LEVEL_SHORT[level] || 'L?') + '</span>' +
            '<span class="badge search-ctx-badge">关联</span>' +
            (eventTimeStart ? '<small class="text-muted">' + esc(eventTimeStart) + '</small>' : '') +
            titleHtml +
        '</div>' +
        summaryHtml +
    '</div>';
}

function toggleSearchChildren(id, el) {
    var container = document.getElementById(id);
    if (!container) return;
    container.classList.toggle('d-none');
    var icon = el.querySelector('svg');
    if (icon) {
        var isHidden = container.classList.contains('d-none');
        icon.style.transform = isHidden ? 'rotate(-90deg)' : '';
    }
}

function toggleSearchSummary(id, link) {
    var el = document.getElementById(id);
    if (el) {
        el.classList.toggle('expanded');
        link.textContent = el.classList.contains('expanded') ? '收起' : '展开';
    }
}

function renderMediaRefs(mediaRefs) {
    if (!mediaRefs || mediaRefs.length === 0) return '';
    var items = mediaRefs.map(function (ref) {
        if (ref.type === 'image') {
            return '<a href="' + escAttr(ref.url) + '" target="_blank" rel="noopener"><img src="' + escAttr(ref.url) + '" class="media-ref-thumb" alt="image" loading="lazy" onerror="this.style.display=\'none\'"></a>';
        } else if (ref.type === 'video') {
            return '<a href="' + escAttr(ref.url) + '" target="_blank" rel="noopener" class="media-ref-video"><i data-feather="play-circle" style="width:14px;height:14px;"></i> 视频</a>';
        }
        return '';
    }).filter(Boolean).join('');
    return items ? '<div class="media-refs">' + items + '</div>' : '';
}

function displaySearchError(message) {
    var container = document.getElementById('mSearchResults');
    var metaContainer = document.getElementById('mSearchMetadata');
    var resultCount = document.getElementById('mResultCount');
    if (metaContainer) metaContainer.classList.add('d-none');
    if (resultCount) resultCount.textContent = '0 条命中';
    if (container) {
        container.innerHTML =
            '<div class="alert alert-danger">' +
                '<i data-feather="alert-circle" class="me-2"></i>' +
                '搜索出错: ' + esc(message) +
            '</div>';
        refreshIcons();
    }
}

/* ============================================================
   Tab 3: Contexts List
   ============================================================ */

function agentLink(agentId) {
    if (!agentId || agentId === 'default') return '';
    return ' <a href="/console/agents?agent_id=' + encodeURIComponent(agentId) + '" title="在 Agent 调试台中查看"><i data-feather="external-link" style="width:12px;height:12px;"></i></a>';
}

function initMemoryContextsTab() {
    var container = document.getElementById('tab-mcontexts');
    if (!container || !selectedUser) return;

    container.innerHTML =
        // Filter bar
        '<div class="card mb-3" style="border-top: 3px solid #2c3e50;">' +
            '<div class="card-body py-3 px-3">' +
                '<div class="row g-2 align-items-end">' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Type</label>' +
                        '<select class="form-select form-select-sm" id="mCtxFilterType">' +
                            '<option value="">全部</option>' +
                        '</select>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">层级</label>' +
                        '<select class="form-select form-select-sm" id="mCtxFilterLevel">' +
                            '<option value="">全部</option>' +
                            '<option value="0">L0 原始事件</option>' +
                            '<option value="1">L1 日摘要</option>' +
                            '<option value="2">L2 周摘要</option>' +
                            '<option value="3">L3 月摘要</option>' +
                        '</select>' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">开始时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="mCtxFilterStartDate">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">结束时间</label>' +
                        '<input type="datetime-local" class="form-control form-control-sm" id="mCtxFilterEndDate">' +
                    '</div>' +
                    // User triple locked
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">User ID</label>' +
                        '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.user_id) + '">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Device ID</label>' +
                        '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.device_id || 'default') + '">' +
                    '</div>' +
                    '<div class="col-6 col-md">' +
                        '<label class="form-label small text-muted mb-1">Agent ID</label>' +
                        '<input type="text" class="form-control form-control-sm" disabled value="' + escAttr(selectedUser.agent_id || 'default') + '">' +
                    '</div>' +
                    '<div class="col-12 col-md-auto">' +
                        '<div class="d-flex gap-2">' +
                            '<button class="btn btn-sm btn-dark" onclick="loadMemoryContexts(1)">筛选</button>' +
                            '<button class="btn btn-sm btn-outline-secondary" onclick="resetMemoryCtxFilters()">重置</button>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +
        // Card grid
        '<div id="mCtxCardGrid" class="row g-3"></div>' +
        // Pagination
        '<div id="mCtxPaginationArea" class="mt-3"></div>';

    // Populate type dropdown
    populateMemoryCtxTypeDropdown();
    loadMemoryContexts(1);
}

async function populateMemoryCtxTypeDropdown() {
    var select = document.getElementById('mCtxFilterType');
    if (!select) return;
    try {
        var params = getMemoryCtxFilterParams();
        params.set('limit', '1');
        var resp = await fetch('/api/contexts?' + params);
        var data = await resp.json();
        if (resp.ok && data.data && data.data.context_types) {
            var types = data.data.context_types;
            var html = '<option value="">全部</option>';
            for (var i = 0; i < types.length; i++) {
                html += '<option value="' + escAttr(types[i]) + '">' + esc(types[i]) + '</option>';
            }
            select.innerHTML = html;
        }
    } catch (e) {
        // Silently fail
    }
}

function resetMemoryCtxFilters() {
    var type = document.getElementById('mCtxFilterType');
    var level = document.getElementById('mCtxFilterLevel');
    var sd = document.getElementById('mCtxFilterStartDate');
    var ed = document.getElementById('mCtxFilterEndDate');
    if (type) type.value = '';
    if (level) level.value = '';
    if (sd) sd.value = '';
    if (ed) ed.value = '';
    loadMemoryContexts(1);
}

function getMemoryCtxFilterParams() {
    var params = new URLSearchParams();
    params.set('user_id', selectedUser.user_id);
    params.set('device_id', selectedUser.device_id || 'default');
    params.set('agent_id', selectedUser.agent_id || 'default');
    var type = document.getElementById('mCtxFilterType');
    var level = document.getElementById('mCtxFilterLevel');
    var startDate = document.getElementById('mCtxFilterStartDate');
    var endDate = document.getElementById('mCtxFilterEndDate');
    if (type && type.value) params.set('type', type.value);
    if (level && level.value !== '') params.set('hierarchy_level', level.value);
    if (startDate && startDate.value) params.set('start_date', startDate.value);
    if (endDate && endDate.value) params.set('end_date', endDate.value);
    return params;
}

async function loadMemoryContexts(page) {
    mCtxPage = page;
    var params = getMemoryCtxFilterParams();
    params.set('page', page);
    params.set('limit', mCtxLimit);

    var grid = document.getElementById('mCtxCardGrid');
    var pagArea = document.getElementById('mCtxPaginationArea');
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
            var linkHtml = agentLink(ctx.agent_id);
            return '<div class="col-md-6 col-lg-4">' + renderContextCardHTML(ctx) +
                (linkHtml ? '<div class="mt-1 ms-1">' + linkHtml + '</div>' : '') +
                '</div>';
        }).join('');

        if (pagArea) {
            pagArea.innerHTML = renderPaginationHTML(page, totalPages, total, mCtxLimit, 'loadMemoryContexts');
        }

        refreshIcons();
    } catch (e) {
        grid.innerHTML = '<div class="col-12 text-center text-danger py-4">加载失败: ' + esc(e.message) + '</div>';
    }
}
