// Settings page JavaScript

let currentPrompts = {};
let selectedPromptKey = '';

// ==================== Utility ====================

function showToast(message, isError = false) {
    const toastEl = document.getElementById('settingsToast');
    const toastBody = document.getElementById('toastMessage');
    toastBody.textContent = message;
    toastEl.classList.remove('bg-success', 'bg-danger');
    toastEl.classList.add(isError ? 'bg-danger' : 'bg-success');
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
}

// Safe element value helpers
function setVal(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value ?? '';
}

function setChecked(id, value) {
    const el = document.getElementById(id);
    if (el) el.checked = !!value;
}

function getVal(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
}

function getInt(id) {
    return parseInt(getVal(id)) || 0;
}

function getFloat(id) {
    return parseFloat(getVal(id)) || 0;
}

function getChecked(id) {
    const el = document.getElementById(id);
    return el ? el.checked : false;
}

// ==================== Tab 1: Model Config ====================

async function loadModelSettings() {
    try {
        const response = await fetch('/api/model_settings/get');
        const data = await response.json();

        if (data.code === 0 && data.data) {
            const d = data.data;

            // LLM
            if (d.llm) {
                setVal('llm_provider', d.llm.provider);
                setVal('llm_model', d.llm.model);
                setVal('llm_base_url', d.llm.base_url);
                setVal('llm_api_key', d.llm.api_key);
                setVal('llm_max_concurrent', d.llm.max_concurrent);
            }

            // VLM
            if (d.vlm_model) {
                setVal('vlm_provider', d.vlm_model.provider);
                setVal('vlm_model', d.vlm_model.model);
                setVal('vlm_base_url', d.vlm_model.base_url);
                setVal('vlm_api_key', d.vlm_model.api_key);
                setVal('vlm_max_concurrent', d.vlm_model.max_concurrent);
            }

            // Embedding
            if (d.embedding_model) {
                setVal('emb_provider', d.embedding_model.provider);
                setVal('emb_model', d.embedding_model.model);
                setVal('emb_base_url', d.embedding_model.base_url);
                setVal('emb_api_key', d.embedding_model.api_key);
                setVal('emb_max_concurrent', d.embedding_model.max_concurrent);
                setVal('emb_output_dim', d.embedding_model.output_dim);
            }
        }
    } catch (error) {
        console.error('Failed to load model settings:', error);
        showToast('Failed to load model settings', true);
    }
}

function _collectModelSection(section) {
    if (section === 'llm') {
        return {
            provider: getVal('llm_provider'),
            model: getVal('llm_model'),
            base_url: getVal('llm_base_url'),
            api_key: getVal('llm_api_key'),
            max_concurrent: getInt('llm_max_concurrent') || null,
        };
    } else if (section === 'vlm_model') {
        return {
            provider: getVal('vlm_provider'),
            model: getVal('vlm_model'),
            base_url: getVal('vlm_base_url'),
            api_key: getVal('vlm_api_key'),
            max_concurrent: getInt('vlm_max_concurrent') || null,
        };
    } else if (section === 'embedding_model') {
        return {
            provider: getVal('emb_provider'),
            model: getVal('emb_model'),
            base_url: getVal('emb_base_url'),
            api_key: getVal('emb_api_key'),
            max_concurrent: getInt('emb_max_concurrent') || null,
            output_dim: getInt('emb_output_dim') || null,
        };
    }
    return null;
}

async function validateModelSection(section) {
    try {
        showToast('Testing connection...');
        const payload = {};
        payload[section] = _collectModelSection(section);

        const response = await fetch('/api/model_settings/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Connection test passed!');
        } else {
            showToast('Connection test failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Validation failed:', error);
        showToast('Connection test failed', true);
    }
}

async function saveAllModels() {
    try {
        const payload = {
            llm: _collectModelSection('llm'),
            vlm_model: _collectModelSection('vlm_model'),
            embedding_model: _collectModelSection('embedding_model'),
        };

        const response = await fetch('/api/model_settings/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Model settings saved successfully');
        } else {
            showToast('Save failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to save model settings:', error);
        showToast('Save failed', true);
    }
}

// ==================== Tab 2: Capture Config ====================

function populateCaptureSettings(allData) {
    const cap = allData.capture || {};

    setChecked('capture_enabled', cap.enabled);

    // text_chat
    const tc = cap.text_chat || {};
    setChecked('text_chat_enabled', tc.enabled);
    setVal('text_chat_buffer_size', tc.buffer_size);

    // folder_monitor
    const fm = cap.folder_monitor || {};
    setChecked('folder_monitor_enabled', fm.enabled);
    setVal('folder_monitor_interval', fm.monitor_interval);
    setVal('folder_monitor_max_file_size', fm.max_file_size);
    setChecked('folder_monitor_recursive', fm.recursive);
    setChecked('folder_monitor_initial_scan', fm.initial_scan);
    const paths = Array.isArray(fm.watch_folder_paths) ? fm.watch_folder_paths.join('\n') : '';
    setVal('folder_monitor_paths', paths);

    // vault_document_monitor
    const vdm = cap.vault_document_monitor || {};
    setChecked('vault_enabled', vdm.enabled);
    setVal('vault_interval', vdm.monitor_interval);
    setChecked('vault_initial_scan', vdm.initial_scan);
}

async function loadCaptureSettings() {
    try {
        const response = await fetch('/api/settings/general');
        const data = await response.json();
        if (data.code === 0 && data.data) populateCaptureSettings(data.data);
    } catch (error) {
        console.error('Failed to load capture settings:', error);
        showToast('Failed to load capture settings', true);
    }
}

document.getElementById('captureForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const pathsText = getVal('folder_monitor_paths').trim();
    const watchPaths = pathsText ? pathsText.split('\n').map(p => p.trim()).filter(Boolean) : [];

    const settings = {
        capture: {
            enabled: getChecked('capture_enabled'),
            text_chat: {
                enabled: getChecked('text_chat_enabled'),
                buffer_size: getInt('text_chat_buffer_size'),
            },
            folder_monitor: {
                enabled: getChecked('folder_monitor_enabled'),
                monitor_interval: getInt('folder_monitor_interval'),
                max_file_size: getInt('folder_monitor_max_file_size'),
                recursive: getChecked('folder_monitor_recursive'),
                initial_scan: getChecked('folder_monitor_initial_scan'),
                watch_folder_paths: watchPaths,
            },
            vault_document_monitor: {
                enabled: getChecked('vault_enabled'),
                monitor_interval: getInt('vault_interval'),
                initial_scan: getChecked('vault_initial_scan'),
            },
        }
    };

    try {
        const response = await fetch('/api/settings/general', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Capture settings saved');
        } else {
            showToast('Save failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to save capture settings:', error);
        showToast('Save failed', true);
    }
});

// ==================== Tab 3: Processing Config ====================

function populateProcessingSettings(allData) {
    const proc = allData.processing || {};

    setChecked('processing_enabled', proc.enabled);

    // text_chat_processor
    setChecked('text_chat_proc_enabled', proc.text_chat_processor?.enabled);

    // document_processor
    const dp = proc.document_processor || {};
    setChecked('doc_proc_enabled', dp.enabled);
    setVal('doc_proc_batch_size', dp.batch_size);
    setVal('doc_proc_batch_timeout', dp.batch_timeout);

    // screenshot_processor
    const sp = proc.screenshot_processor || {};
    setChecked('screenshot_proc_enabled', sp.enabled);
    setVal('screenshot_hash_threshold', sp.similarity_hash_threshold);
    setVal('screenshot_batch_size', sp.batch_size);
    setVal('screenshot_batch_timeout', sp.batch_timeout);
    setVal('screenshot_max_raw_props', sp.max_raw_properties);
    setVal('screenshot_max_image_size', sp.max_image_size);
    setVal('screenshot_resize_quality', sp.resize_quality);
    setChecked('screenshot_enabled_delete', sp.enabled_delete);

    // context_merger
    const cm = proc.context_merger || {};
    setChecked('merger_enabled', cm.enabled);
    setVal('merger_threshold', cm.similarity_threshold);
    setChecked('merger_intelligent', cm.use_intelligent_merging);
    setChecked('merger_memory_mgmt', cm.enable_memory_management);
    setVal('merger_retention_days', cm.knowledge_retention_days);
    setVal('merger_sim_threshold', cm.knowledge_similarity_threshold);
    setVal('merger_max_merge_count', cm.knowledge_max_merge_count);

    // document_processing (separate top-level key)
    const docProc = allData.document_processing || {};
    setVal('docproc_batch_size', docProc.batch_size);
    setVal('docproc_dpi', docProc.dpi);
    setVal('docproc_text_threshold', docProc.text_threshold_per_page);
}

async function loadProcessingSettings() {
    try {
        const response = await fetch('/api/settings/general');
        const data = await response.json();
        if (data.code === 0 && data.data) populateProcessingSettings(data.data);
    } catch (error) {
        console.error('Failed to load processing settings:', error);
        showToast('Failed to load processing settings', true);
    }
}

document.getElementById('processingForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const settings = {
        processing: {
            enabled: getChecked('processing_enabled'),
            text_chat_processor: {
                enabled: getChecked('text_chat_proc_enabled'),
            },
            document_processor: {
                enabled: getChecked('doc_proc_enabled'),
                batch_size: getInt('doc_proc_batch_size'),
                batch_timeout: getInt('doc_proc_batch_timeout'),
            },
            screenshot_processor: {
                enabled: getChecked('screenshot_proc_enabled'),
                similarity_hash_threshold: getInt('screenshot_hash_threshold'),
                batch_size: getInt('screenshot_batch_size'),
                batch_timeout: getInt('screenshot_batch_timeout'),
                max_raw_properties: getInt('screenshot_max_raw_props'),
                max_image_size: getInt('screenshot_max_image_size'),
                resize_quality: getInt('screenshot_resize_quality'),
                enabled_delete: getChecked('screenshot_enabled_delete'),
            },
            context_merger: {
                enabled: getChecked('merger_enabled'),
                similarity_threshold: getFloat('merger_threshold'),
                use_intelligent_merging: getChecked('merger_intelligent'),
                enable_memory_management: getChecked('merger_memory_mgmt'),
                knowledge_retention_days: getInt('merger_retention_days'),
                knowledge_similarity_threshold: getFloat('merger_sim_threshold'),
                knowledge_max_merge_count: getInt('merger_max_merge_count'),
            },
        },
        document_processing: {
            batch_size: getInt('docproc_batch_size'),
            dpi: getInt('docproc_dpi'),
            text_threshold_per_page: getInt('docproc_text_threshold'),
        },
    };

    try {
        const response = await fetch('/api/settings/general', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Processing settings saved');
        } else {
            showToast('Save failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to save processing settings:', error);
        showToast('Save failed', true);
    }
});

// ==================== Tab 4: Scheduler & Cache ====================

function populateSchedulerCacheSettings(allData) {
    const sched = allData.scheduler || {};
    setChecked('scheduler_enabled', sched.enabled);

    // executor
    const exec = sched.executor || {};
    setVal('executor_check_interval', exec.check_interval);
    setVal('executor_max_concurrent', exec.max_concurrent);
    setVal('executor_lock_timeout', exec.lock_timeout);

    // tasks
    const tasks = sched.tasks || {};

    // memory_compression
    const mc = tasks.memory_compression || {};
    setChecked('task_mc_enabled', mc.enabled);
    setVal('task_mc_trigger', mc.trigger_mode);
    setVal('task_mc_interval', mc.interval);
    setVal('task_mc_ttl', mc.task_ttl);

    // data_cleanup
    const dc = tasks.data_cleanup || {};
    setChecked('task_dc_enabled', dc.enabled);
    setVal('task_dc_trigger', dc.trigger_mode);
    setVal('task_dc_interval', dc.interval);
    setVal('task_dc_timeout', dc.timeout);
    setVal('task_dc_retention', dc.retention_days);

    // hierarchy_summary
    const hs = tasks.hierarchy_summary || {};
    setChecked('task_hs_enabled', hs.enabled);
    setVal('task_hs_trigger', hs.trigger_mode);
    setVal('task_hs_interval', hs.interval);
    setVal('task_hs_timeout', hs.timeout);
    setVal('task_hs_ttl', hs.task_ttl);

    // memory_cache
    const cache = allData.memory_cache || {};
    setVal('cache_snapshot_ttl', cache.snapshot_ttl);
    setVal('cache_recent_days', cache.recent_days);
    setVal('cache_max_recently', cache.max_recently_accessed);
    setVal('cache_max_today_events', cache.max_today_events);
    setVal('cache_max_documents', cache.max_recent_documents);
    setVal('cache_max_knowledge', cache.max_recent_knowledge);
    setVal('cache_accessed_ttl', cache.accessed_ttl);
    setVal('cache_max_entities', cache.max_entities);
}

async function loadSchedulerCacheSettings() {
    try {
        const response = await fetch('/api/settings/general');
        const data = await response.json();
        if (data.code === 0 && data.data) populateSchedulerCacheSettings(data.data);
    } catch (error) {
        console.error('Failed to load scheduler/cache settings:', error);
        showToast('Failed to load scheduler/cache settings', true);
    }
}

document.getElementById('schedulerCacheForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const settings = {
        scheduler: {
            enabled: getChecked('scheduler_enabled'),
            executor: {
                check_interval: getInt('executor_check_interval'),
                max_concurrent: getInt('executor_max_concurrent'),
                lock_timeout: getInt('executor_lock_timeout'),
            },
            tasks: {
                memory_compression: {
                    enabled: getChecked('task_mc_enabled'),
                    trigger_mode: getVal('task_mc_trigger'),
                    interval: getInt('task_mc_interval'),
                    task_ttl: getInt('task_mc_ttl'),
                },
                data_cleanup: {
                    enabled: getChecked('task_dc_enabled'),
                    trigger_mode: getVal('task_dc_trigger'),
                    interval: getInt('task_dc_interval'),
                    timeout: getInt('task_dc_timeout'),
                    retention_days: getInt('task_dc_retention'),
                },
                hierarchy_summary: {
                    enabled: getChecked('task_hs_enabled'),
                    trigger_mode: getVal('task_hs_trigger'),
                    interval: getInt('task_hs_interval'),
                    timeout: getInt('task_hs_timeout'),
                    task_ttl: getInt('task_hs_ttl'),
                },
            },
        },
        memory_cache: {
            snapshot_ttl: getInt('cache_snapshot_ttl'),
            recent_days: getInt('cache_recent_days'),
            max_recently_accessed: getInt('cache_max_recently'),
            max_today_events: getInt('cache_max_today_events'),
            max_recent_documents: getInt('cache_max_documents'),
            max_recent_knowledge: getInt('cache_max_knowledge'),
            accessed_ttl: getInt('cache_accessed_ttl'),
            max_entities: getInt('cache_max_entities'),
        },
    };

    try {
        const response = await fetch('/api/settings/general', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Scheduler & cache settings saved');
        } else {
            showToast('Save failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to save scheduler/cache settings:', error);
        showToast('Save failed', true);
    }
});

// ==================== Tab 5: Prompts ====================

function getPromptValue(key) {
    const keys = key.split('.');
    let value = currentPrompts;
    for (const k of keys) {
        value = value?.[k];
    }
    return typeof value === 'string' ? value : '';
}

function setPromptValue(key, newValue) {
    const keys = key.split('.');
    let current = currentPrompts;
    for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) current[keys[i]] = {};
        current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = newValue;
}

function buildPromptTree(obj, prefix, parentEl) {
    for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            // Group node
            const groupBtn = document.createElement('button');
            groupBtn.className = 'list-group-item list-group-item-action fw-bold';
            groupBtn.textContent = key;
            groupBtn.onclick = function() {
                const children = this.nextElementSibling;
                if (children) {
                    children.style.display = children.style.display === 'none' ? 'block' : 'none';
                }
            };
            parentEl.appendChild(groupBtn);

            const childContainer = document.createElement('div');
            childContainer.style.paddingLeft = '20px';
            childContainer.style.display = 'none';

            // Check for leaf strings inside
            let hasLeaves = false;
            for (const [subKey, subValue] of Object.entries(value)) {
                if (typeof subValue === 'string') {
                    hasLeaves = true;
                    const btn = document.createElement('button');
                    btn.className = 'list-group-item list-group-item-action';
                    btn.textContent = subKey;
                    btn.onclick = () => selectPrompt(`${fullKey}.${subKey}`);
                    childContainer.appendChild(btn);
                }
            }

            // Recurse into nested objects
            for (const [subKey, subValue] of Object.entries(value)) {
                if (typeof subValue === 'object' && subValue !== null && !Array.isArray(subValue)) {
                    buildPromptTree({ [subKey]: subValue }, fullKey, childContainer);
                }
            }

            parentEl.appendChild(childContainer);
        }
    }
}

function selectPrompt(key) {
    selectedPromptKey = key;
    document.getElementById('currentPromptKey').textContent = key;
    document.getElementById('promptEditor').value = getPromptValue(key);

    // Highlight the selected leaf
    document.querySelectorAll('#promptTree button').forEach(btn => {
        btn.classList.remove('active');
    });
    // Find the button whose onclick invokes this key
    document.querySelectorAll('#promptTree .list-group-item-action:not(.fw-bold)').forEach(btn => {
        if (btn.textContent === key.split('.').pop()) {
            btn.classList.add('active');
        }
    });
}

document.getElementById('promptEditor')?.addEventListener('change', (e) => {
    if (selectedPromptKey) {
        setPromptValue(selectedPromptKey, e.target.value);
    }
});

async function loadPrompts() {
    try {
        const response = await fetch('/api/settings/prompts');
        const data = await response.json();

        if (data.code === 0 && data.data) {
            currentPrompts = data.data.prompts;
            const treeEl = document.getElementById('promptTree');
            treeEl.innerHTML = '';
            buildPromptTree(currentPrompts, '', treeEl);
        }
    } catch (error) {
        console.error('Failed to load prompts:', error);
        showToast('Failed to load prompts', true);
    }
}

async function saveCurrentPrompt() {
    if (!selectedPromptKey) {
        showToast('No prompt selected', true);
        return;
    }
    // Sync editor value
    setPromptValue(selectedPromptKey, document.getElementById('promptEditor').value);
    await savePrompts();
}

async function saveAllPrompts() {
    // Sync current editor
    if (selectedPromptKey) {
        setPromptValue(selectedPromptKey, document.getElementById('promptEditor').value);
    }
    await savePrompts();
}

async function savePrompts() {
    try {
        const response = await fetch('/api/settings/prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompts: currentPrompts })
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Prompts saved successfully');
        } else {
            showToast('Save failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to save prompts:', error);
        showToast('Save failed', true);
    }
}

async function exportPrompts() {
    try {
        const response = await fetch('/api/settings/prompts/export');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `prompts_${new Date().getTime()}.yaml`;
        a.click();
        window.URL.revokeObjectURL(url);
        showToast('Export successful');
    } catch (error) {
        console.error('Failed to export prompts:', error);
        showToast('Export failed', true);
    }
}

async function importPrompts(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/settings/prompts/import', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Import successful');
            loadPrompts();
        } else {
            showToast('Import failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to import prompts:', error);
        showToast('Import failed', true);
    } finally {
        event.target.value = '';
    }
}

async function resetAllSettings() {
    if (!confirm('Are you sure you want to reset all settings? This will delete all custom configurations and prompts. This action cannot be undone!')) {
        return;
    }

    try {
        const response = await fetch('/api/settings/reset', { method: 'POST' });
        const data = await response.json();

        if (data.code === 0) {
            showToast('Settings reset successfully, reloading...');
            setTimeout(() => location.reload(), 2000);
        } else {
            showToast('Reset failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to reset settings:', error);
        showToast('Reset failed', true);
    }
}

// ==================== Prompt Language ====================

async function getCurrentPromptLanguage() {
    try {
        const response = await fetch('/api/settings/prompts/language');
        const data = await response.json();
        if (data.code === 0 && data.data) {
            return data.data.language || 'zh';
        }
    } catch (error) {
        console.error('Failed to get prompt language:', error);
    }
    return 'zh';
}

async function changePromptLanguage() {
    const language = document.getElementById('promptLanguage').value;

    if (!confirm(`Switch to ${language === 'zh' ? 'Chinese' : 'English'} prompts? Unsaved changes will be lost.`)) {
        const currentLang = await getCurrentPromptLanguage();
        document.getElementById('promptLanguage').value = currentLang;
        return;
    }

    try {
        const response = await fetch('/api/settings/prompts/language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: language })
        });

        const data = await response.json();
        if (data.code === 0) {
            showToast(`Switched to ${language === 'zh' ? 'Chinese' : 'English'} prompts`);
            await loadPrompts();
        } else {
            showToast('Language switch failed: ' + (data.message || 'Unknown error'), true);
        }
    } catch (error) {
        console.error('Failed to switch prompt language:', error);
        showToast('Language switch failed', true);
    }
}

// ==================== Bulk Load General Settings ====================

async function loadAllGeneralSettings() {
    try {
        const response = await fetch('/api/settings/general');
        const data = await response.json();

        if (data.code === 0 && data.data) {
            populateCaptureSettings(data.data);
            populateProcessingSettings(data.data);
            populateSchedulerCacheSettings(data.data);
        }
    } catch (error) {
        console.error('Failed to load general settings:', error);
        showToast('Failed to load settings', true);
    }
}

// ==================== Page Init ====================

document.addEventListener('DOMContentLoaded', async function() {
    // Load model settings + all general settings in parallel (single fetch)
    loadModelSettings();
    loadAllGeneralSettings();

    // Prompts lazy-load on tab switch
    let promptsLoaded = false;
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            const target = e.target.getAttribute('data-bs-target');
            if (target === '#prompts' && !promptsLoaded) {
                loadPrompts();
                promptsLoaded = true;
            }
        });
    });

    // Set language selector
    const currentLang = await getCurrentPromptLanguage();
    const langSelect = document.getElementById('promptLanguage');
    if (langSelect) {
        langSelect.value = currentLang;
    }
});
