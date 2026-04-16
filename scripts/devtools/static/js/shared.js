/**
 * MineContext DevTools Console — shared utilities
 *
 * Provides common helpers used by both agent_console and memory_explorer pages:
 * escaping, clipboard, date formatting, badge rendering, pagination, context
 * cards, and icon refresh.
 */

/* ============================================================
   HTML Escaping
   ============================================================ */

function esc(str) {
    if (str == null) return '';
    var div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function escAttr(str) {
    return esc(str).replace(/'/g, '&#39;');
}

/* ============================================================
   Clipboard
   ============================================================ */

function fallbackCopyText(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
}

function copyToClipboard(text, feedbackEl) {
    var done = function () {
        if (feedbackEl) {
            var orig = feedbackEl.textContent;
            feedbackEl.textContent = '已复制';
            setTimeout(function () { feedbackEl.textContent = orig; }, 1200);
        }
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
            fallbackCopyText(text);
            done();
        });
    } else {
        fallbackCopyText(text);
        done();
    }
}

/* ============================================================
   Date Formatting
   ============================================================ */

function formatDatetime(dtStr) {
    if (!dtStr) return '';
    return String(dtStr).substring(0, 19).replace('T', ' ');
}

function formatDateShort(dtStr) {
    if (!dtStr) return '';
    return String(dtStr).substring(0, 16).replace('T', ' ');
}

/* ============================================================
   Context Type Helpers
   ============================================================ */

var TYPE_COLORS = {
    event:          'warning',
    knowledge:      'purple',
    document:       'success',
    profile:        'primary',
    agent_profile:  'info'
};

function typeBadgeClass(type) {
    return 'badge-' + (type || 'event');
}

/* ============================================================
   Hierarchy Level Helpers
   ============================================================ */

var LEVEL_LABELS = { 0: '原始事件', 1: '日摘要', 2: '周摘要', 3: '月摘要' };
var LEVEL_SHORT  = { 0: 'L0', 1: 'L1', 2: 'L2', 3: 'L3' };
var LEVEL_BADGE_CLASS = { 0: 'badge-l0', 1: 'badge-l1', 2: 'badge-l2', 3: 'badge-l3' };

function levelBadge(level) {
    var l = (level != null) ? Number(level) : 0;
    var cls = LEVEL_BADGE_CLASS[l] || 'bg-secondary';
    var label = LEVEL_SHORT[l] || ('L' + l);
    return '<span class="badge ' + cls + '">' + esc(label) + '</span>';
}

/* ============================================================
   View Context Detail (opens full-page detail view)
   ============================================================ */

function viewContext(contextId, contextType) {
    fetch('/contexts/detail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: contextId, context_type: contextType || 'event' })
    }).then(function (resp) {
        if (!resp.ok) throw new Error('请求失败');
        return resp.text();
    }).then(function (html) {
        document.open();
        document.write(html);
        document.close();
    }).catch(function (err) {
        console.error('viewContext failed:', err);
        alert('查看失败');
    });
}

/* ============================================================
   Pagination
   ============================================================ */

function renderPaginationHTML(page, totalPages, total, limit, onPageClick) {
    if (totalPages <= 1) return '';

    var html = '<nav><ul class="pagination pagination-sm justify-content-center mb-0">';

    // Previous
    html += '<li class="page-item ' + (page <= 1 ? 'disabled' : '') + '">';
    html += '<a class="page-link" href="#" onclick="' + onPageClick + '(' + (page - 1) + ');return false;">&laquo;</a></li>';

    // Page numbers — show at most 5 around current
    var start = Math.max(1, page - 2);
    var end = Math.min(totalPages, page + 2);
    if (start > 1) {
        html += '<li class="page-item"><a class="page-link" href="#" onclick="' + onPageClick + '(1);return false;">1</a></li>';
        if (start > 2) html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
    }
    for (var p = start; p <= end; p++) {
        html += '<li class="page-item ' + (p === page ? 'active' : '') + '">';
        html += '<a class="page-link" href="#" onclick="' + onPageClick + '(' + p + ');return false;">' + p + '</a></li>';
    }
    if (end < totalPages) {
        if (end < totalPages - 1) html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
        html += '<li class="page-item"><a class="page-link" href="#" onclick="' + onPageClick + '(' + totalPages + ');return false;">' + totalPages + '</a></li>';
    }

    // Next
    html += '<li class="page-item ' + (page >= totalPages ? 'disabled' : '') + '">';
    html += '<a class="page-link" href="#" onclick="' + onPageClick + '(' + (page + 1) + ');return false;">&raquo;</a></li>';

    html += '</ul>';
    html += '<div class="text-center text-muted small mt-1">共 ' + total + ' 条，每页 ' + limit + ' 条</div>';
    html += '</nav>';
    return html;
}

/* ============================================================
   Collapsible Card Wrapper
   ============================================================ */

function wrapCard(title, icon, content, sectionId) {
    var collapseId = 'collapse-' + (sectionId || Math.random().toString(36).substring(2, 8));
    return '<div class="card mb-3">' +
        '<div class="card-header" style="cursor:pointer;" data-bs-toggle="collapse" data-bs-target="#' + collapseId + '">' +
            '<h6 class="mb-0 d-flex align-items-center">' +
                (icon ? '<i data-feather="' + escAttr(icon) + '" style="width:16px;height:16px;" class="me-2"></i>' : '') +
                esc(title) +
                '<i class="bi bi-chevron-down ms-auto"></i>' +
            '</h6>' +
        '</div>' +
        '<div class="collapse show" id="' + collapseId + '">' +
            '<div class="card-body">' + content + '</div>' +
        '</div>' +
    '</div>';
}

/* ============================================================
   Context Card Renderer
   ============================================================ */

function renderContextCardHTML(ctx) {
    var id = ctx.id || '';
    var type = ctx.context_type || 'event';
    var title = ctx.title || '未命名';
    var summary = ctx.summary || '';
    var level = ctx.hierarchy_level;
    var truncId = id.length > 12 ? id.substring(0, 12) + '...' : id;

    // Media icons
    var mediaHtml = '';
    var refs = (ctx.metadata && ctx.metadata.media_refs) || [];
    if (refs.length > 0) {
        var hasImage = refs.some(function (r) { return r.type === 'image'; });
        var hasVideo = refs.some(function (r) { return r.type === 'video'; });
        if (hasImage) mediaHtml += '<i data-feather="image" style="width:14px;height:14px;" class="text-muted me-1"></i>';
        if (hasVideo) mediaHtml += '<i data-feather="video" style="width:14px;height:14px;" class="text-muted me-1"></i>';
    }

    var html = '<div class="ctx-card" id="ctx-card-' + escAttr(id) + '">';

    // Header row: type badge, level badge, media icons, truncated ID
    html += '<div class="d-flex align-items-center gap-1 mb-1">';
    html += '<span class="badge ' + typeBadgeClass(type) + '">' + esc(type) + '</span>';
    if (level != null && type === 'event') {
        html += ' ' + levelBadge(level);
    }
    html += mediaHtml;
    html += '<small class="text-muted ms-auto" title="' + escAttr(id) + '">' + esc(truncId) + '</small>';
    html += '</div>';

    // Title
    html += '<div class="fw-semibold mb-1">' + esc(title) + '</div>';

    // Summary (truncated)
    if (summary) {
        var short = summary.length > 150 ? summary.substring(0, 150) + '...' : summary;
        html += '<div class="text-muted small mb-2">' + esc(short) + '</div>';
    }

    // Actions
    html += '<div class="d-flex gap-2">';
    html += '<a href="#" class="btn btn-outline-primary btn-sm" onclick="viewContext(\'' + escAttr(id) + '\',\'' + escAttr(type) + '\');return false;">查看</a>';
    html += '<a href="#" class="btn btn-outline-danger btn-sm" onclick="deleteContextItem(\'' + escAttr(id) + '\',\'' + escAttr(type) + '\',this);return false;">删除</a>';
    html += '</div>';

    html += '</div>';
    return html;
}

/* ============================================================
   Delete Context
   ============================================================ */

async function deleteContextItem(id, type, el) {
    if (!confirm('确认删除此上下文？此操作不可撤销。')) return;
    try {
        var resp = await fetch('/contexts/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: id, context_type: type })
        });
        if (resp.ok) {
            var card = document.getElementById('ctx-card-' + id);
            if (card) {
                card.style.transition = 'opacity 0.3s';
                card.style.opacity = '0';
                setTimeout(function () { card.remove(); }, 300);
            }
        } else {
            alert('删除失败');
        }
    } catch (err) {
        console.error('deleteContextItem failed:', err);
        alert('删除失败');
    }
}

/* ============================================================
   Feather Icons Refresh
   ============================================================ */

function refreshIcons() {
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}
