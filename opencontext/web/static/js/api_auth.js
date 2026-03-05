/**
 * API Authentication utility for OpenContext
 * Handles API key management and request headers
 */

// Save original fetch before monkey patching
const originalFetch = window.fetch;

class APIAuth {
    constructor() {
        this.apiKey = null;
        this.authEnabled = false;
    }

    /**
     * Initialize API authentication
     */
    async init() {
        // Check auth status
        await this.checkAuthStatus();
        
        if (!this.authEnabled) {
            console.log('API authentication is disabled');
            return;
        }
        
        // Load API key from localStorage
        this.apiKey = localStorage.getItem('context_lab_api_key');
        
        if (!this.apiKey) {
            console.warn('API authentication is enabled but no API key found');
            this.showAPIKeyModal();
        }
    }

    /**
     * Check if authentication is enabled on the server
     */
    async checkAuthStatus() {
        try {
            const response = await originalFetch('/api/auth/status');
            if (response.ok) {
                const data = await response.json();
                this.authEnabled = data.data?.auth_enabled || false;
                console.log(`API authentication: ${this.authEnabled ? 'enabled' : 'disabled'}`);
            }
        } catch (error) {
            console.warn('Failed to check auth status:', error);
            this.authEnabled = false;
        }
    }

    /**
     * Set API key and save to localStorage
     */
    setAPIKey(apiKey) {
        this.apiKey = apiKey;
        if (apiKey) {
            localStorage.setItem('context_lab_api_key', apiKey);
        } else {
            localStorage.removeItem('context_lab_api_key');
        }
    }

    /**
     * Validate API key by making a test request
     */
    async validateAPIKey(apiKey) {
        try {
            const response = await originalFetch('/api/context_types', {
                headers: {
                    'X-API-Key': apiKey,
                    'Content-Type': 'application/json'
                }
            });
            return response.ok;
        } catch (error) {
            console.error('API key validation failed:', error);
            return false;
        }
    }

    /**
     * Enhanced fetch with automatic API key handling
     */
    async fetch(url, options = {}) {
        // Always include API key if available (don't wait for authEnabled check,
        // which may not have completed yet due to async init race condition)
        const apiKey = this.apiKey || localStorage.getItem('context_lab_api_key');

        // If no API key stored and auth not enabled, use original fetch
        if (!apiKey && !this.authEnabled) {
            return originalFetch(url, options);
        }

        // Prepare headers with API key
        const headers = {
            'Content-Type': 'application/json',
            ...(options.headers || {})
        };

        if (apiKey) {
            headers['X-API-Key'] = apiKey;
        }
        
        const requestOptions = {
            ...options,
            headers
        };

        try {
            const response = await originalFetch(url, requestOptions);
            
            // Handle authentication errors
            if (response.status === 401) {
                console.warn('Authentication failed. Please check your API key.');
                // Clear invalid API key
                this.setAPIKey(null);
                this.showAPIKeyModal();
            }

            return response;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * Show API key management modal
     */
    showAPIKeyModal() {
        return new Promise((resolve) => {
            // Check if modal already exists
            const existingModal = document.getElementById('apiKeyModal');
            if (existingModal) {
                existingModal.remove();
            }

            const modal = document.createElement('div');
            modal.id = 'apiKeyModal';
            modal.innerHTML = `
                <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 10000; display: flex; align-items: center; justify-content: center;">
                    <div style="background: white; padding: 30px; border-radius: 10px; max-width: 500px; width: 90%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h3 style="margin: 0 0 20px 0; color: #333;">API Key 设置</h3>
                        <p style="color: #666; margin-bottom: 20px;">
                            后端服务已启用API认证。请输入有效的API Key以继续访问。
                            <br><small style="color: #999;">API Key由系统管理员提供，用于保护后端服务安全。</small>
                        </p>
                        <div style="position: relative; margin-bottom: 20px;">
                            <input 
                                type="password" 
                                id="apiKeyInput" 
                                placeholder="请输入 API Key" 
                                value="${this.apiKey || ''}"
                                style="width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; box-sizing: border-box;">
                            <button 
                                id="toggleKeyVisibility"
                                style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: none; border: none; color: #666; cursor: pointer; padding: 5px;"
                                title="显示/隐藏">
                                👁️
                            </button>
                        </div>
                        <div id="apiKeyError" style="color: #dc3545; margin-bottom: 15px; font-size: 14px; display: none;"></div>
                        <div style="text-align: right; gap: 10px; display: flex; justify-content: flex-end;">
                            <button 
                                id="cancelBtn"
                                style="padding: 10px 20px; border: 1px solid #ddd; background: white; border-radius: 6px; cursor: pointer; font-size: 14px; color: #666;">
                                取消
                            </button>
                            <button 
                                id="validateBtn"
                                style="padding: 10px 20px; border: 1px solid #007bff; background: white; border-radius: 6px; cursor: pointer; font-size: 14px; color: #007bff;">
                                验证
                            </button>
                            <button 
                                id="saveBtn"
                                style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px;">
                                保存
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const input = document.getElementById('apiKeyInput');
            const errorDiv = document.getElementById('apiKeyError');
            const saveBtn = document.getElementById('saveBtn');
            const validateBtn = document.getElementById('validateBtn');
            const cancelBtn = document.getElementById('cancelBtn');
            const toggleBtn = document.getElementById('toggleKeyVisibility');
            
            // Focus input
            input.focus();
            
            // Toggle password visibility
            toggleBtn.addEventListener('click', () => {
                const type = input.type === 'password' ? 'text' : 'password';
                input.type = type;
                toggleBtn.textContent = type === 'password' ? '👁️' : '🔒';
            });
            
            // Validate API key
            const validateKey = async () => {
                const apiKey = input.value.trim();
                if (!apiKey) {
                    errorDiv.textContent = '请输入API Key';
                    errorDiv.style.display = 'block';
                    return false;
                }
                
                validateBtn.disabled = true;
                validateBtn.textContent = '验证中...';
                
                const isValid = await this.validateAPIKey(apiKey);
                
                validateBtn.disabled = false;
                validateBtn.textContent = '验证';
                
                if (isValid) {
                    errorDiv.textContent = '✅ API Key 验证成功';
                    errorDiv.style.color = '#28a745';
                    errorDiv.style.display = 'block';
                    return true;
                } else {
                    errorDiv.textContent = '❌ API Key 无效，请检查后重试';
                    errorDiv.style.color = '#dc3545';
                    errorDiv.style.display = 'block';
                    return false;
                }
            };
            
            // Handle validate button
            validateBtn.addEventListener('click', validateKey);
            
            // Handle save button
            saveBtn.addEventListener('click', async () => {
                const apiKey = input.value.trim();
                if (!apiKey) {
                    errorDiv.textContent = '请输入API Key';
                    errorDiv.style.display = 'block';
                    return;
                }
                
                // Validate before saving
                saveBtn.disabled = true;
                saveBtn.textContent = '验证中...';
                
                const isValid = await this.validateAPIKey(apiKey);
                
                saveBtn.disabled = false;
                saveBtn.textContent = '保存';
                
                if (isValid) {
                    this.setAPIKey(apiKey);
                    modal.remove();
                    resolve(true);
                } else {
                    errorDiv.textContent = '❌ API Key 无效，请检查后重试';
                    errorDiv.style.color = '#dc3545';
                    errorDiv.style.display = 'block';
                }
            });
            
            // Handle cancel button
            cancelBtn.addEventListener('click', () => {
                modal.remove();
                resolve(false);
            });
            
            // Handle Enter key
            input.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter') {
                    saveBtn.click();
                }
            });
        });
    }

    /**
     * Handle API key button click from header
     */
    showAPIKeyManager() {
        this.showAPIKeyModal();
    }
}

// Create global instance
window.apiAuth = new APIAuth();

// Monkey patch fetch for API calls that need authentication
window.fetch = function(url, options = {}) {
    // Only patch API calls that need authentication (not auth status check)
    if (typeof url === 'string' &&
        (url.startsWith('/api/') || url.startsWith('/contexts/')) &&
        !url.startsWith('/api/auth/')) {
        return window.apiAuth.fetch(url, options);
    }
    return originalFetch(url, options);
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => window.apiAuth.init());
} else {
    window.apiAuth.init();
}

console.log('API Auth utility loaded');