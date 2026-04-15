// ==================== GLOBAL CONFIG ====================
// Single shared script for landing + dashboard (analytics, monitoring, violations)

// Global configuration loaded from backend
let API_BASE_URL = 'http://localhost:8000'; // fallback
let WS_BASE_URL = 'ws://localhost:8000';   // fallback
let POLL_INTERVAL = 2000; // 2 seconds
let STREAM_POLL_INTERVAL = 1000; // 1 second for stream frames
let CONFIG_LOADED = false;

// Detect if running locally
function isLocalEnvironment() {
    const hostname = window.location.hostname;
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '';
}

// Load configuration from backend
async function loadConfig() {
    try {
        // Determine base URL for config fetch
        const configUrl = isLocalEnvironment() 
            ? 'http://localhost:8000/api/config' 
            : '/api/config';
        
        console.log('🔧 Loading config from:', configUrl);
        
        const response = await fetch(configUrl, {
            method: 'GET',
            cache: 'no-cache'
        });
        
        if (response.ok) {
            const config = await response.json();
            API_BASE_URL = config.api_base_url;
            WS_BASE_URL = config.ws_base_url;
            POLL_INTERVAL = config.poll_intervals?.stats || 2000;
            STREAM_POLL_INTERVAL = config.poll_intervals?.stream_frame || 1000;
            CONFIG_LOADED = true;
            console.log('✅ Config loaded from backend:', config);
            console.log('📍 Environment:', config.environment || 'unknown');
        } else {
            throw new Error(`Config endpoint returned ${response.status}`);
        }
    } catch (error) {
        console.warn('⚠️ Failed to load config from backend, using auto-detection:', error.message);
        
        // Fallback to auto-detection based on current location
        if (isLocalEnvironment()) {
            API_BASE_URL = 'http://localhost:8000';
            WS_BASE_URL = 'ws://localhost:8000';
            console.log('🏠 Detected local environment');
        } else {
            API_BASE_URL = 'https://traffic-monitoring-api.onrender.com';
            WS_BASE_URL = 'wss://traffic-monitoring-api.onrender.com';
            console.log('☁️ Detected production environment');
        }
        CONFIG_LOADED = true;
    }
}

// ==================== GLOBAL STATE ====================
let systemStats = {};
let activeStreams = new Set();
let pollInterval = null;
let streamPollIntervals = {};
let reconnectionInterval = null; // For continuous stream reconnection checks
let currentUser = null;
let authToken = null;
let violationsByTypeChart = null;
let streamsVehiclesChart = null;
let violationsOverTimeChart = null;
let analyticsDateRange = 'all';

// ==================== UTILITIES ====================
// Toast cooldown — violation toasts throttled to max one per 5 seconds.
let _lastViolationToastTime = 0;
let _pendingViolationCount = 0;
let _pendingViolationTimer = null;
const TOAST_COOLDOWN_MS = 5000;

function showToast(message, type = 'success') {
    const isViolation = type === 'error' && message.toLowerCase().includes('violation');

    if (isViolation) {
        const now = Date.now();
        if (now - _lastViolationToastTime < TOAST_COOLDOWN_MS) {
            _pendingViolationCount++;
            if (!_pendingViolationTimer) {
                _pendingViolationTimer = setTimeout(() => {
                    if (_pendingViolationCount > 0) {
                        _showToastDom(`+${_pendingViolationCount} more violation(s) detected`, 'error');
                        _pendingViolationCount = 0;
                    }
                    _pendingViolationTimer = null;
                }, TOAST_COOLDOWN_MS - (now - _lastViolationToastTime));
            }
            return;
        }
        _lastViolationToastTime = now;
    }

    _showToastDom(message, type);
}

function _showToastDom(message, type) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    return headers;
}

function formatTime(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return isNaN(date.getTime()) ? 'N/A' : date.toLocaleTimeString();
}

function formatSpeed(speed) {
    return typeof speed === 'number' && !isNaN(speed) ? `${speed.toFixed(1)} km/h` : 'N/A';
}

// ==================== LANDING PAGE AUTH (from landing.js) ====================
function showAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.add('active');
    const emailInput = document.getElementById('loginEmail');
    if (emailInput) emailInput.focus();
}

function hideAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.remove('active');
    clearAuthError();
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    if (loginForm) loginForm.reset();
    if (signupForm) signupForm.reset();
}

function switchAuthTab(tab) {
    const tabs = document.querySelectorAll('.auth-tab');
    const forms = document.querySelectorAll('.auth-form');

    tabs.forEach(t => t.classList.remove('active'));
    forms.forEach(f => f.classList.remove('active'));

    if (tab === 'login') {
        if (tabs[0]) tabs[0].classList.add('active');
        const form = document.getElementById('loginForm');
        if (form) form.classList.add('active');
    } else {
        if (tabs[1]) tabs[1].classList.add('active');
        const form = document.getElementById('signupForm');
        if (form) form.classList.add('active');
    }
    clearAuthError();
}

function openSignup() {
    switchAuthTab('signup');
    showAuthModal();
}

function openLogin() {
    switchAuthTab('login');
    showAuthModal();
}

function showAuthError(message) {
    const errorEl = document.getElementById('authError');
    const textEl = document.getElementById('authErrorText');
    if (!errorEl || !textEl) return;
    textEl.textContent = message;
    errorEl.classList.add('show');
}

function clearAuthError() {
    const errorEl = document.getElementById('authError');
    if (!errorEl) return;
    errorEl.classList.remove('show');
}

function setButtonLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (loading) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div><span>Please wait...</span>';
    } else {
        btn.disabled = false;
        btn.innerHTML = `<span>${btnId.includes('login') ? 'Login' : 'Create Account'}</span>`;
    }
}

async function handleLogin(event) {
    event.preventDefault();
    clearAuthError();
    setButtonLoading('loginSubmitBtn', true);

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Login failed');

        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        hideAuthModal();
        showToast(`Welcome back, ${currentUser.username}!`, 'success');

        // After successful login, send user to monitoring dashboard
        window.location.href = '/frontend/monitoring.html';
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setButtonLoading('loginSubmitBtn', false);
    }
}

async function handleSignup(event) {
    event.preventDefault();
    clearAuthError();
    setButtonLoading('signupSubmitBtn', true);

    const full_name = document.getElementById('signupName').value;
    const username = document.getElementById('signupUsername').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, full_name })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Signup failed');

        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        hideAuthModal();
        showToast(`Welcome to TMS, ${currentUser.username}!`, 'success');

        // After successful signup, send user to monitoring dashboard
        window.location.href = '/frontend/monitoring.html';
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setButtonLoading('signupSubmitBtn', false);
    }
}

// ==================== DASHBOARD AUTH (from app.js) ====================
async function logout() {
    try {
        if (authToken) {
            await fetch(`${API_BASE_URL}/api/auth/logout`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
        }
    } catch (error) {
        console.error('Logout error:', error);
    }

    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    authToken = null;
    currentUser = null;

    // Stop all stream polling
    for (let i = 0; i < 4; i++) {
        if (streamPollIntervals[i]) {
            clearInterval(streamPollIntervals[i]);
            delete streamPollIntervals[i];
        }
    }

    updateAuthUI();
    closeUserDropdown();
    showToast('You have been logged out', 'success');

    // Redirect to public landing page served from backend
    window.location.href = '/';
}

function updateAuthUI() {
    const userDropdown = document.getElementById('userDropdown');

    if (currentUser && authToken) {
        // Desktop: Show user dropdown
        if (userDropdown) userDropdown.classList.remove('hidden');

        // Desktop: Update Avatar
        if (currentUser.username) {
            const initials = currentUser.username.substring(0, 2).toUpperCase();
            const avatarEl = document.getElementById('userAvatar');
            if (avatarEl) avatarEl.textContent = initials;
        }
        
        // Determine display name
        const displayName = currentUser.full_name || currentUser.username || 'User';

        // Desktop: Update Name and Email
        const nameEl = document.getElementById('userDisplayName');
        const emailEl = document.getElementById('userDisplayEmail');
        if (nameEl) nameEl.textContent = displayName;
        if (emailEl) emailEl.textContent = currentUser.email || '';

        // Mobile: Update Name in Hamburger Menu
        const mobileNameEl = document.getElementById('mobileUserName');
        if (mobileNameEl) {
            mobileNameEl.textContent = displayName;
        }
    } else {
        // User not logged in - hide desktop dropdown
        if (userDropdown) userDropdown.classList.add('hidden');
        
        // Reset mobile menu to default
        const mobileNameEl = document.getElementById('mobileUserName');
        if (mobileNameEl) {
            mobileNameEl.textContent = 'Guest User';
        }
    }
}

function toggleUserDropdown() {
    const dropdown = document.getElementById('userDropdown');
    if (!dropdown) return;
    dropdown.classList.toggle('active');
}

function closeUserDropdown() {
    const dropdown = document.getElementById('userDropdown');
    if (!dropdown) return;
    dropdown.classList.remove('active');
}

async function verifyToken() {
    if (!authToken) return false;
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/verify`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (!response.ok) {
            localStorage.removeItem('authToken');
            localStorage.removeItem('currentUser');
            authToken = null;
            currentUser = null;
            return false;
        }
        return true;
    } catch (error) {
        console.error('Token verification error:', error);
        return false;
    }
}

function checkExistingAuthForDashboard() {
    const savedToken = localStorage.getItem('authToken');
    const savedUser = localStorage.getItem('currentUser');

    if (!savedToken || !savedUser) {
        // Not authenticated - redirect back to landing
        console.log('❌ Not authenticated, redirecting to landing page...');
        window.location.href = '/';
        return false;
    }

    authToken = savedToken;
    currentUser = JSON.parse(savedUser);
    console.log('✅ Authenticated as:', currentUser.username);

    verifyToken();
    updateAuthUI();
    return true;
}

// ==================== NAVIGATION (for single-page dashboard.html only) ====================
function navigateTo(pageId) {
    if (!currentUser || !authToken) {
        window.location.href = '/';
        return;
    }

    // Only used by the combined dashboard.html which uses data-page attributes
    document.querySelectorAll('.nav-link[data-page]').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.dashboard-page').forEach(p => p.classList.remove('active'));

    const navLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
    if (navLink) navLink.classList.add('active');

    const targetPage = document.getElementById(pageId);
    if (targetPage) targetPage.classList.add('active');

    closeUserDropdown();

    if (pageId === 'violations') updateViolationsPage();
    if (pageId === 'analytics' || pageId === 'monitoring') updateSystemStats();
}

// ==================== CHARTS (ANALYTICS) ====================
function initCharts() {
    const ctxViolationsByType = document.getElementById('violationsByTypeChart');
    const ctxStreamsVehicles = document.getElementById('streamsVehiclesChart');
    const ctxViolationsOverTime = document.getElementById('violationsOverTimeChart');

    if (ctxViolationsByType && window.Chart) {
        violationsByTypeChart = new Chart(ctxViolationsByType, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Violations',
                    data: [],
                    backgroundColor: 'rgba(6, 182, 212, 0.6)',
                    borderColor: 'rgba(6, 182, 212, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#cbd5e1' } },
                    y: { ticks: { color: '#cbd5e1' }, beginAtZero: true }
                }
            }
        });
    }

    if (ctxStreamsVehicles && window.Chart) {
        streamsVehiclesChart = new Chart(ctxStreamsVehicles, {
            type: 'doughnut',
            data: {
                labels: ['Active Streams', 'Total Vehicles'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(59, 130, 246, 0.7)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                }
            }
        });
    }

    if (ctxViolationsOverTime && window.Chart) {
        violationsOverTimeChart = new Chart(ctxViolationsOverTime, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Violations',
                    data: [],
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.3)',
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                },
                scales: {
                    x: { ticks: { color: '#cbd5e1' } },
                    y: { ticks: { color: '#cbd5e1' }, beginAtZero: true }
                }
            }
        });
    }
}

function updateChartsFromStats(data) {
    if (!data) return;

    if (violationsByTypeChart && data.violation_summary) {
        const labels = Object.keys(data.violation_summary);
        const values = labels.map(k => data.violation_summary[k] || 0);
        violationsByTypeChart.data.labels = labels.map(l => l.replace('_', ' ').toUpperCase());
        violationsByTypeChart.data.datasets[0].data = values;
        violationsByTypeChart.update();
    }

    if (streamsVehiclesChart) {
        streamsVehiclesChart.data.datasets[0].data = [
            data.active_streams || 0,
            data.total_vehicles || 0
        ];
        streamsVehiclesChart.update();
    }

    if (violationsOverTimeChart && Array.isArray(data.violations)) {
        const buckets = {};
        data.violations.forEach(v => {
            if (!v.timestamp) return;
            const d = new Date(v.timestamp);
            if (isNaN(d.getTime())) return;
            const key = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            buckets[key] = (buckets[key] || 0) + 1;
        });
        const labels = Object.keys(buckets).sort();
        const values = labels.map(k => buckets[k]);
        violationsOverTimeChart.data.labels = labels;
        violationsOverTimeChart.data.datasets[0].data = values;
        violationsOverTimeChart.update();
    }
}

// ==================== STREAM MANAGEMENT ====================
function startStream(index) {
    const urlInput = document.getElementById('streamUrl' + index);
    if (!urlInput) return; // Not on monitoring page
    const url = urlInput.value.trim();
    if (!url) {
        showToast('Please enter a stream URL', 'error');
        return;
    }

    // Check if it's a YouTube URL
    const isYouTube = url.includes('youtube.com') || url.includes('youtu.be');
    
    console.log(`Starting stream ${index} with URL:`, url, isYouTube ? '(YouTube)' : '');
    
    if (isYouTube) {
        showToast(`Processing YouTube URL for stream ${index + 1}... This may take a moment.`, 'info');
    } else {
        showToast(`Starting stream ${index + 1}...`, 'info');
    }

    fetch(`${API_BASE_URL}/api/start-stream/${index}?stream_url=${encodeURIComponent(url)}`, {
        method: 'POST',
        headers: getAuthHeaders()
    })
        .then(async r => {
            const data = await r.json();
            if (!r.ok) {
                // Provide more helpful error messages for YouTube URLs
                let errorMsg = data.detail || `Server returned ${r.status}`;
                if (isYouTube && (errorMsg.includes('extract') || errorMsg.includes('yt-dlp') || errorMsg.includes('Failed'))) {
                    errorMsg = 'YouTube URL processing failed. Please check:\n1. The URL is valid and accessible\n2. The video is not private/restricted\n3. Try a different YouTube video';
                }
                throw new Error(errorMsg);
            }
            return data;
        })
        .then((data) => {
            console.log(`Stream ${index} started successfully`, data);
            if (isYouTube) {
                showToast(`YouTube stream ${index + 1} started successfully! Processing video...`, 'success');
            } else {
                showToast(`Stream ${index + 1} started successfully!`, 'success');
            }
            updateStreamStatus(index);
            
            // Show loading state immediately
            const placeholder = document.getElementById('placeholder' + index);
            if (placeholder) {
                placeholder.style.display = 'flex';
                placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Processing video stream...</p>';
            }
            
            // Give YouTube streams more time before polling (extraction + processing takes time)
            // Regular streams: 500ms, YouTube: 5 seconds (allows time for extraction + initial frames)
            setTimeout(() => {
                console.log(`Starting frame polling for stream ${index}${isYouTube ? ' (YouTube)' : ''}`);
                pollStreamFrame(index);
            }, isYouTube ? 5000 : 500);
        })
        .catch(err => {
            console.error(`Failed to start stream ${index}:`, err);
            let errorMsg = err.message;
            if (isYouTube) {
                errorMsg = `YouTube stream failed: ${errorMsg}`;
            }
            showToast(errorMsg, 'error');
        });
}

function stopStream(index) {
    fetch(`${API_BASE_URL}/api/stop-stream/${index}`, {
        method: 'POST',
        headers: getAuthHeaders()
    })
        .then(r => r.json())
        .then(() => {
            showToast(`Stream ${index + 1} stopped`, 'success');
            
            // Remove from active streams set
            activeStreams.delete(index);
            
            const img = document.getElementById('stream' + index);
            const placeholder = document.getElementById('placeholder' + index);
            const statusEl = document.getElementById('status' + index);
            if (img) img.style.display = 'none';
            if (placeholder) {
                placeholder.style.display = 'flex';
                placeholder.innerHTML = '<i class="fas fa-video"></i><p>Stream Inactive</p>';
            }
            if (statusEl) {
                statusEl.textContent = 'Inactive';
                statusEl.className = 'status-badge status-inactive';
            }

            // Clear polling interval
            if (streamPollIntervals[index]) {
                clearInterval(streamPollIntervals[index]);
                streamPollIntervals[index] = null;
            }
        })
        .catch(err => showToast(`Failed to stop stream: ${err.message}`, 'error'));
}

function updateStreamStatus(index) {
    const statusEl = document.getElementById('status' + index);
    if (!statusEl) return;
    statusEl.textContent = 'Active';
    statusEl.className = 'status-badge status-active';
}

function pollStreamFrame(index) {
    const img = document.getElementById('stream' + index);
    const placeholder = document.getElementById('placeholder' + index);
    if (!img || !placeholder) {
        console.log(`⚠️ Cannot poll stream ${index}: elements not found`);
        return; // Not on monitoring UI
    }

    // Clear any existing interval for this stream to prevent duplicates
    if (streamPollIntervals[index]) {
        console.log(`🔄 Clearing existing interval for stream ${index}`);
        clearInterval(streamPollIntervals[index]);
        streamPollIntervals[index] = null;
    }

    // Mark stream as active
    activeStreams.add(index);
    
    console.log(`📹 Starting frame polling for stream ${index}`);
    
    let consecutiveErrors = 0;
    const maxConsecutiveErrors = 5;
    
    // Check if this is a YouTube stream by checking the URL input
    const urlInput = document.getElementById('streamUrl' + index);
    const isYouTubeStream = urlInput && (urlInput.value.includes('youtube.com') || urlInput.value.includes('youtu.be'));
    
    streamPollIntervals[index] = setInterval(async () => {
        try {
            // Add cache-busting timestamp to prevent browser caching
            const timestamp = new Date().getTime();
            const frameUrl = `${API_BASE_URL}/stream/${index}/frame?t=${timestamp}`;
            
            // YouTube streams may need longer timeout due to processing overhead
            const timeoutDuration = isYouTubeStream ? 10000 : 5000; // 10s for YouTube, 5s for others
            
            const response = await fetch(frameUrl, {
                headers: getAuthHeaders(),
                cache: 'no-store',
                signal: AbortSignal.timeout(timeoutDuration)
            });

            if (response.ok && response.headers.get('content-type')?.includes('image')) {
                // Success - reset error count
                consecutiveErrors = 0;
                
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                // Update image source
                img.onload = () => {
                    URL.revokeObjectURL(imageUrl); // Clean up old URL
                };
                img.src = imageUrl;
                img.style.display = 'block';
                placeholder.style.display = 'none';
            } else {
                consecutiveErrors++;
                console.warn(`Stream ${index} frame fetch failed (${consecutiveErrors}/${maxConsecutiveErrors})`);
                
                // Only check status after multiple failures to avoid excessive API calls
                if (consecutiveErrors >= 3) {
                    // Stream may have stopped - check status
                    const statusCheck = await fetch(`${API_BASE_URL}/api/stream-status/${index}`, {
                        headers: getAuthHeaders()
                    });
                    const status = await statusCheck.json();
                    
                    if (!status.processing) {
                        // Stream actually stopped, clean up
                        console.log(`❌ Stream ${index} stopped, clearing interval`);
                        clearInterval(streamPollIntervals[index]);
                        streamPollIntervals[index] = null;
                        activeStreams.delete(index);
                        img.style.display = 'none';
                        placeholder.style.display = 'flex';
                        placeholder.innerHTML = '<i class="fas fa-video"></i><p>Stream Inactive</p>';
                        
                        const statusEl = document.getElementById('status' + index);
                        if (statusEl) {
                            statusEl.textContent = 'Inactive';
                            statusEl.className = 'status-badge status-inactive';
                        }
                    } else {
                        // Stream is still processing, just waiting for frames
                        // Show different message for YouTube streams
                        const urlInput = document.getElementById('streamUrl' + index);
                        const isYouTube = urlInput && (urlInput.value.includes('youtube.com') || urlInput.value.includes('youtu.be'));
                        if (isYouTube) {
                            placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Processing YouTube video... This may take a moment.</p>';
                        } else {
                            placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Loading frames...</p>';
                        }
                    }
                }
            }
        } catch (err) {
            consecutiveErrors++;
            console.error(`❌ Error polling stream ${index} (${consecutiveErrors}/${maxConsecutiveErrors}):`, err.message);
            
            if (consecutiveErrors >= maxConsecutiveErrors) {
                // Too many errors, stop polling
                console.error(`🛑 Stream ${index} has too many errors, stopping polling`);
                clearInterval(streamPollIntervals[index]);
                streamPollIntervals[index] = null;
                activeStreams.delete(index);
                
                placeholder.style.display = 'flex';
                placeholder.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Connection lost</p>';
                img.style.display = 'none';
                
                showToast(`Stream ${index + 1} connection lost`, 'error');
            }
        }
    }, STREAM_POLL_INTERVAL);
}

// Check and reconnect to active streams on page load
async function reconnectToActiveStreams() {
    const stream0 = document.getElementById('stream0');
    const streamUrl0 = document.getElementById('streamUrl0');
    
    if (!stream0 && !streamUrl0) {
        console.log('⏭️ Not on monitoring page, skipping stream reconnection');
        return; // Not on monitoring page
    }

    console.log('🔄 Reconnecting to active streams...');
    
    try {
        // Fetch stats to see which streams are active
        const statsResponse = await fetch(`${API_BASE_URL}/api/stats`, {
            headers: getAuthHeaders(),
            cache: 'no-store'
        });
        
        if (!statsResponse.ok) {
            console.error('Failed to fetch stats:', statsResponse.status);
            return;
        }
        
        const stats = await statsResponse.json();
        
        if (!stats.streams || !Array.isArray(stats.streams)) {
            console.log('⚠️ No stream data available');
            return;
        }
        
        let reconnectedCount = 0;
        let alreadyConnectedCount = 0;
        
        // Process each stream
        for (const streamInfo of stats.streams) {
            const i = streamInfo.stream_id;
            
            if (i < 0 || i >= 4) continue; // Invalid stream ID
            
            const img = document.getElementById('stream' + i);
            const placeholder = document.getElementById('placeholder' + i);
            const statusEl = document.getElementById('status' + i);
            const urlInput = document.getElementById('streamUrl' + i);
            
            if (streamInfo.processing) {
                // Check if already connected
                if (streamPollIntervals[i] && activeStreams.has(i)) {
                    console.log(`✓ Stream ${i} already connected and polling`);
                    alreadyConnectedCount++;
                    continue;
                }
                
                console.log(`✅ Stream ${i} is active on backend, reconnecting...`);
                
                try {
                    // Clear any existing interval first
                    if (streamPollIntervals[i]) {
                        clearInterval(streamPollIntervals[i]);
                        streamPollIntervals[i] = null;
                    }
                    
                    // Mark as active immediately
                    activeStreams.add(i);
                    
                    // Update UI status
                    if (statusEl) {
                        statusEl.textContent = 'Active';
                        statusEl.className = 'status-badge status-active';
                    }
                    
                    // Restore stream URL in input
                    if (urlInput && streamInfo.stream_url) {
                        urlInput.value = streamInfo.stream_url;
                    }
                    
                    // Show placeholder while loading
                    if (placeholder) {
                        placeholder.style.display = 'flex';
                        placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Reconnecting...</p>';
                    }
                    if (img) {
                        img.style.display = 'none';
                    }
                    
                    // Start polling frames immediately
                    pollStreamFrame(i);
                    
                    reconnectedCount++;
                } catch (streamErr) {
                    console.error(`❌ Error reconnecting stream ${i}:`, streamErr);
                }
            } else {
                // Only update UI if stream was previously active
                if (activeStreams.has(i) || streamPollIntervals[i]) {
                    console.log(`⏸️ Stream ${i} stopped, updating UI`);
                    
                    // Ensure UI reflects inactive state
                    if (statusEl) {
                        statusEl.textContent = 'Inactive';
                        statusEl.className = 'status-badge status-inactive';
                    }
                    if (img) img.style.display = 'none';
                    if (placeholder) {
                        placeholder.style.display = 'flex';
                        placeholder.innerHTML = '<i class="fas fa-video"></i><p>Stream Inactive</p>';
                    }
                    
                    // Clear any existing polling
                    if (streamPollIntervals[i]) {
                        clearInterval(streamPollIntervals[i]);
                        streamPollIntervals[i] = null;
                    }
                    activeStreams.delete(i);
                }
            }
        }
        
        if (reconnectedCount > 0) {
            console.log(`✅ Successfully reconnected ${reconnectedCount} stream(s)`);
            showToast(`✅ Reconnected to ${reconnectedCount} active stream(s)`, 'success');
        } else if (alreadyConnectedCount > 0) {
            console.log(`✓ ${alreadyConnectedCount} stream(s) already connected`);
        } else {
            console.log('ℹ️ No active streams to reconnect');
        }
    } catch (err) {
        console.error('❌ Error in reconnectToActiveStreams:', err);
    }
}

// ==================== FILE UPLOAD ====================
function setupFileUpload(index) {
    const fileInput = document.getElementById('videoFile' + index);
    const uploadLabel = fileInput ? document.querySelector(`label[for="videoFile${index}"]`) : null;
    const selectedFile = document.getElementById('selectedFile' + index);
    const fileName = document.getElementById('fileName' + index);

    if (!fileInput || !uploadLabel || !selectedFile || !fileName) return; // Page does not have this stream

    const uploadBox = uploadLabel.querySelector('.upload-box');

    fileInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            selectedFile.style.display = 'flex';
        }
    });

    uploadLabel.addEventListener('dragover', e => {
        e.preventDefault();
        if (!uploadBox) return;
        uploadBox.style.borderColor = 'var(--accent)';
        uploadBox.style.background = 'rgba(6, 182, 212, 0.15)';
    });

    uploadLabel.addEventListener('dragleave', () => {
        if (!uploadBox) return;
        uploadBox.style.borderColor = 'var(--border-color)';
        uploadBox.style.background = 'rgba(6, 182, 212, 0.02)';
    });

    uploadLabel.addEventListener('drop', e => {
        e.preventDefault();
        if (uploadBox) {
            uploadBox.style.borderColor = 'var(--border-color)';
            uploadBox.style.background = 'rgba(6, 182, 212, 0.02)';
        }

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });
}

function clearStreamFile(index) {
    const input = document.getElementById('videoFile' + index);
    const selectedFile = document.getElementById('selectedFile' + index);
    if (input) input.value = '';
    if (selectedFile) selectedFile.style.display = 'none';
}

function uploadStreamVideo(index) {
    const fileInput = document.getElementById('videoFile' + index);
    if (!fileInput) return;
    const file = fileInput.files[0];

    if (!file) {
        showToast('Please select a video file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const headers = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

    fetch(`${API_BASE_URL}/api/upload-video/${index}`, {
        method: 'POST',
        headers,
        body: formData
    })
        .then(r => r.json())
        .then(() => {
            showToast(`Video uploading and processing started for Stream ${index + 1}`, 'success');
            clearStreamFile(index);
            updateStreamStatus(index);
            pollStreamFrame(index);
        })
        .catch(err => showToast(`Upload failed: ${err.message}`, 'error'));
}

// ==================== STATS AND VIOLATIONS ====================
async function updateSystemStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();
        
        // Store stats but preserve any existing violation data to prevent overwrites
        // when user has applied date filters
        const preservedViolations = systemStats?.violations || [];
        systemStats = data;
        
        // If violations exist from previous filter, keep them
        if (preservedViolations.length > 0 && !data.violations) {
            systemStats.violations = preservedViolations;
        }

        const activeStreamsEl = document.getElementById('activeStreams');
        const totalVehiclesEl = document.getElementById('totalVehicles');
        const totalViolationsEl = document.getElementById('totalViolations');
        const notificationBadgeEl = document.getElementById('notificationBadge');
        const speedViolationsEl = document.getElementById('speedViolations');
        const congestedStreams = data.congested_streams ?? 0;

        if (activeStreamsEl) activeStreamsEl.textContent = data.active_streams ?? 0;
        if (totalVehiclesEl) totalVehiclesEl.textContent = data.total_vehicles ?? 0;
        
        // Don't overwrite violation counts if we're on analytics page with a date filter applied
        const analyticsDateFilter = document.getElementById('analyticsDateFilter');
        const isFilterApplied = analyticsDateFilter && analyticsDateFilter.value !== 'all';
        
        if (!isFilterApplied && totalViolationsEl) {
            totalViolationsEl.textContent = data.total_violations ?? 0;
        }
        // Notification badge: show total violations, but if there are
        // congested streams, highlight by appending a marker.
        if (notificationBadgeEl) {
            const baseCount = data.total_violations ?? 0;
            notificationBadgeEl.textContent = baseCount;
            if (congestedStreams > 0) {
                notificationBadgeEl.classList.add('has-congestion');
            } else {
                notificationBadgeEl.classList.remove('has-congestion');
            }
        }

        const summary = data.violation_summary || {};
        
        // Don't overwrite speed violation count if date filter is applied
        if (!isFilterApplied && speedViolationsEl) {
            speedViolationsEl.textContent = summary.speed || 0;
        }

        // If any streams are congested, surface a toast once per update.
        if (congestedStreams > 0 && Array.isArray(data.streams)) {
            data.streams
                .filter(s => s.is_congested)
                .forEach(s => {
                    const count = s.current_vehicle_count ?? 0;
                    showToast(`High traffic on Stream ${s.stream_id}: ${count} vehicles detected`, 'error');
                });
        }

        // Only update violations table if no filter is applied
        // to prevent overwriting filtered results
        if (!isFilterApplied) {
            const violations = (data.violations || []).slice(0, 10);
            updateViolationsTable(violations);
            updateChartsFromStats(data);
        }
    } catch (err) {
        console.error('Error fetching stats:', err);
    }
}

function updateViolationsTable(violations) {
    const tbody = document.getElementById('violationsTableBody');
    if (!tbody) return; // Not on analytics/dashboard page

    if (!violations || violations.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="table-empty-state"><i class="fas fa-inbox"></i><p>No violations</p></td></tr>';
        return;
    }

    tbody.innerHTML = violations.map(v => `
        <tr>
            <td>Stream ${v.stream_id}</td>
            <td>${(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
            <td>${formatSpeed(v.speed_kmh)}</td>
            <td>${formatTime(v.timestamp)}</td>
        </tr>
    `).join('');
}

async function updateViolationsPage() {
    const dateFilterSelect = document.getElementById('violationDateFilter');
    const dateRange = dateFilterSelect ? dateFilterSelect.value : 'all';
    const datePicker = document.getElementById('violationDatePicker');
    
    // Build query string - support both date_range and specific_date
    let queryParams = 'limit=500';  // Increased limit for better historical data
    
    // Handle custom date picker
    if (dateRange === 'custom' && datePicker && datePicker.value) {
        queryParams += `&specific_date=${encodeURIComponent(datePicker.value)}`;
    } else if (dateRange !== 'all') {
        queryParams += `&date_range=${encodeURIComponent(dateRange)}`;
    }

    // Always try to fetch from database first
    try {
        const response = await fetch(`${API_BASE_URL}/api/db/violations?${queryParams}`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();

        // Use database results (even if empty array)
        const violations = data.violations || [];
        
        console.log(`📊 Fetched ${violations.length} violations from database for range: ${dateRange}`);
        
        // Count violations by type from the filtered results
        const counts = {
            speed: 0,
            red_light: 0,
            stop_line: 0,
            lane_change: 0,
            unsafe_distance: 0
        };

        violations.forEach(v => {
            const vtype = v.violation_type;
            if (counts.hasOwnProperty(vtype)) {
                counts[vtype]++;
            }
        });

        // Update count displays with filtered data
        const speedEl = document.getElementById('speedViolationCount');
        const redEl = document.getElementById('redLightCount');
        const stopEl = document.getElementById('stopLineCount');
        const laneEl = document.getElementById('laneChangeCount');

        if (speedEl) speedEl.textContent = counts.speed;
        if (redEl) redEl.textContent = counts.red_light;
        if (stopEl) stopEl.textContent = counts.stop_line || counts.unsafe_distance;
        if (laneEl) laneEl.textContent = counts.lane_change;

        // Update violations table
        const tbody = document.getElementById('allViolationsTable');
        if (tbody) {
            if (violations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="table-empty-state"><i class="fas fa-inbox"></i><p>No violations found for selected period</p></td></tr>';
            } else {
                tbody.innerHTML = violations.map(v => `
                    <tr>
                        <td>Stream ${v.stream_id}</td>
                        <td>${(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
                        <td>${formatSpeed(v.speed_kmh)}</td>
                        <td>${v.signal_state || 'N/A'}</td>
                        <td>${formatTime(v.timestamp)}</td>
                    </tr>
                `).join('');
            }
        }
        
        return; // Success - exit function
        
    } catch (err) {
        console.error('❌ Error fetching violations from database:', err);
        // Fall through to in-memory fallback
    }

    // FALLBACK: Use in-memory stats if database fetch failed
    console.log('⚠️ Using in-memory fallback for violations');
    
    if (!systemStats || !systemStats.violations) {
        // No data available at all - show zeros
        console.log('No system stats available');
        const speedEl = document.getElementById('speedViolationCount');
        const redEl = document.getElementById('redLightCount');
        const stopEl = document.getElementById('stopLineCount');
        const laneEl = document.getElementById('laneChangeCount');
        
        if (speedEl) speedEl.textContent = '0';
        if (redEl) redEl.textContent = '0';
        if (stopEl) stopEl.textContent = '0';
        if (laneEl) laneEl.textContent = '0';
        
        const tbody = document.getElementById('allViolationsTable');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="5" class="table-empty-state"><i class="fas fa-inbox"></i><p>No violations available</p></td></tr>';
        }
        return;
    }

    // Filter in-memory violations by date range
    let filteredViolations = [...systemStats.violations];  // Create a copy
    
    if (dateRange !== 'all') {
        const now = new Date();
        let startTime = null;
        let endTime = null;

        if (dateRange === 'custom' && datePicker && datePicker.value) {
            // Custom date from picker
            const targetDate = new Date(datePicker.value);
            startTime = new Date(targetDate.getFullYear(), targetDate.getMonth(), targetDate.getDate());
            endTime = new Date(startTime.getTime() + 24 * 60 * 60 * 1000);
        } else if (dateRange === 'today') {
            startTime = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        } else if (dateRange === 'yesterday') {
            const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            startTime = new Date(todayStart.getTime() - 24 * 60 * 60 * 1000);
            endTime = todayStart;
        } else if (dateRange === 'last_week') {
            startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        } else if (dateRange === 'last_month') {
            startTime = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        } else if (dateRange === 'last_year') {
            startTime = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
        }

        if (startTime) {
            filteredViolations = filteredViolations.filter(v => {
                if (!v.timestamp) return false;
                const d = new Date(v.timestamp);
                if (isNaN(d.getTime())) return false;
                if (endTime) return d >= startTime && d < endTime;
                return d >= startTime;
            });
        }
    }

    console.log(`📊 Filtered to ${filteredViolations.length} violations from memory`);

    // Count from filtered in-memory violations
    const counts = {
        speed: 0,
        red_light: 0,
        stop_line: 0,
        lane_change: 0,
        unsafe_distance: 0
    };
    
    filteredViolations.forEach(v => {
        const vtype = v.violation_type;
        if (counts.hasOwnProperty(vtype)) {
            counts[vtype]++;
        }
    });

    const speedEl = document.getElementById('speedViolationCount');
    const redEl = document.getElementById('redLightCount');
    const stopEl = document.getElementById('stopLineCount');
    const laneEl = document.getElementById('laneChangeCount');

    if (speedEl) speedEl.textContent = counts.speed;
    if (redEl) redEl.textContent = counts.red_light;
    if (stopEl) stopEl.textContent = counts.stop_line || counts.unsafe_distance;
    if (laneEl) laneEl.textContent = counts.lane_change;

    const tbody = document.getElementById('allViolationsTable');
    if (!tbody) return;

    if (!filteredViolations || filteredViolations.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="table-empty-state"><i class="fas fa-inbox"></i><p>No violations found for selected period</p></td></tr>';
        return;
    }

    tbody.innerHTML = filteredViolations.map(v => `
        <tr>
            <td>Stream ${v.stream_id}</td>
            <td>${(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
            <td>${formatSpeed(v.speed_kmh)}</td>
            <td>${v.signal_state || 'N/A'}</td>
            <td>${formatTime(v.timestamp)}</td>
        </tr>
    `).join('');
}

// ==================== WEBSOCKET CONNECTION ====================
function connectWebSocket() {
    const wsUrl = `${WS_BASE_URL}/ws`;
    console.log('🔌 Connecting to WebSocket:', wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        showToast('Connected to backend');
    };

    ws.onmessage = event => {
        const message = JSON.parse(event.data);

        if (message.type === 'violation') {
            updateSystemStats();
            showToast(`Violation detected: ${message.data.violation_type}`, 'error');
        } else if (message.type === 'stats_update') {
            systemStats = message.data;
            updateSystemStats();
        }
    };

    ws.onerror = error => {
        console.error('❌ WebSocket error:', error);
        showToast('Connection error', 'error');
    };

    ws.onclose = () => {
        console.log('⚠️ WebSocket disconnected, reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };
}

// ==================== ANALYTICS DATE RANGE ====================
async function updateAnalyticsRange() {
    const select = document.getElementById('analyticsDateFilter');
    analyticsDateRange = select ? select.value : 'all';
    const datePicker = document.getElementById('analyticsDatePicker');
    
    // Show/hide date picker based on selection
    if (datePicker) {
        if (analyticsDateRange === 'custom') {
            datePicker.style.display = 'block';
            // Don't proceed if no date selected yet
            if (!datePicker.value) {
                return;
            }
        } else {
            datePicker.style.display = 'none';
        }
    }
    
    const pickedDate = datePicker && datePicker.value ? datePicker.value : null;

    // Build query string based on range / specific date
    let query = '?limit=500';
    if (analyticsDateRange === 'custom' && pickedDate) {
        query += `&specific_date=${encodeURIComponent(pickedDate)}`;
    } else if (analyticsDateRange !== 'all') {
        query += `&date_range=${encodeURIComponent(analyticsDateRange)}`;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/db/violations${query}`, {
            headers: getAuthHeaders()
        });
        const data = await response.json();

        // Always use database results (even if empty) to ensure accurate counts
        const violations = data.violations || [];

        // Build summary from filtered violations
        const summary = {
            speed: 0,
            red_light: 0,
            stop_line: 0,
            lane_change: 0,
            unsafe_distance: 0
        };
        violations.forEach(v => {
            if (summary.hasOwnProperty(v.violation_type)) {
                summary[v.violation_type]++;
            }
        });

        // Update UI with filtered counts
        const totalViolationsEl = document.getElementById('totalViolations');
        const speedViolationsEl = document.getElementById('speedViolations');
        if (totalViolationsEl) totalViolationsEl.textContent = violations.length;
        if (speedViolationsEl) speedViolationsEl.textContent = summary.speed;

        // Update violations table with filtered data
        updateViolationsTable(violations.slice(0, 10));

        // Update charts with filtered data
        const statsLike = Object.assign({}, systemStats, {
            violation_summary: summary,
            violations: violations
        });
        updateChartsFromStats(statsLike);
        return; // Always return after database fetch (success or empty)
    } catch (err) {
        console.error('Error updating analytics range from DB:', err);
        // Fall through to in-memory fallback only on error
    }

    // Fallback to in-memory stats only if database fetch failed
    if (!systemStats.violations) {
        // No data available - set counts to 0
        const totalViolationsEl = document.getElementById('totalViolations');
        const speedViolationsEl = document.getElementById('speedViolations');
        if (totalViolationsEl) totalViolationsEl.textContent = '0';
        if (speedViolationsEl) speedViolationsEl.textContent = '0';
        updateViolationsTable([]);
        return;
    }

    let filteredViolations = systemStats.violations;
    if (analyticsDateRange !== 'all') {
        let startTime = null;
        let endTime = null;

        if (analyticsDateRange === 'today') {
            const now = new Date();
            startTime = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        } else if (analyticsDateRange === 'yesterday') {
            const now = new Date();
            const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            startTime = new Date(todayStart.getTime() - 24 * 60 * 60 * 1000);
            endTime = todayStart;
        } else if (analyticsDateRange === 'last_week') {
            const now = new Date();
            startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        } else if (analyticsDateRange === 'last_month') {
            const now = new Date();
            startTime = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        } else if (analyticsDateRange === 'last_year') {
            const now = new Date();
            startTime = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
        } else if (analyticsDateRange === 'custom' && pickedDate) {
            const d = new Date(pickedDate + 'T00:00:00');
            if (!isNaN(d.getTime())) {
                startTime = d;
                endTime = new Date(d.getTime() + 24 * 60 * 60 * 1000);
            }
        }

        if (startTime) {
            filteredViolations = filteredViolations.filter(v => {
                if (!v.timestamp) return false;
                const d = new Date(v.timestamp);
                if (isNaN(d.getTime())) return false;
                if (endTime) return d >= startTime && d < endTime;
                return d >= startTime;
            });
        }
    }

    // Count from filtered in-memory violations
    const summary = {
        speed: 0,
        red_light: 0,
        stop_line: 0,
        lane_change: 0,
        unsafe_distance: 0
    };
    filteredViolations.forEach(v => {
        if (summary.hasOwnProperty(v.violation_type)) {
            summary[v.violation_type]++;
        }
    });

    // Update UI with filtered counts
    const totalViolationsEl = document.getElementById('totalViolations');
    const speedViolationsEl = document.getElementById('speedViolations');
    if (totalViolationsEl) totalViolationsEl.textContent = filteredViolations.length;
    if (speedViolationsEl) speedViolationsEl.textContent = summary.speed;

    updateViolationsTable(filteredViolations.slice(0, 10));

    const statsLike = Object.assign({}, systemStats, {
        violation_summary: summary,
        violations: filteredViolations
    });
    updateChartsFromStats(statsLike);
}

// ==================== INITIALIZATION ====================
function initLandingPage() {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.addEventListener('click', function (event) {
            if (event.target === this) hideAuthModal();
        });
    }

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') hideAuthModal();
    });

    const savedToken = localStorage.getItem('authToken');
    const savedUser = localStorage.getItem('currentUser');
    if (savedToken && savedUser) {
        authToken = savedToken;
        currentUser = JSON.parse(savedUser);
    }
}

function initDashboardPages() {
    // Only enforce auth on dashboard-style pages
    if (!checkExistingAuthForDashboard()) return;

    // Setup dropdown outside-click close
    document.addEventListener('click', function (event) {
        const dropdown = document.getElementById('userDropdown');
        if (dropdown && !dropdown.contains(event.target)) closeUserDropdown();
    });

    // Setup nav links for SPA dashboard.html if data-page is used
    const spaNavLinks = document.querySelectorAll('.nav-link[data-page]');
    spaNavLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const pageId = this.getAttribute('data-page');
            navigateTo(pageId);
        });
    });

    // Initialize charts if present (analytics or dashboard)
    if (document.getElementById('violationsByTypeChart')) {
        initCharts();
    }

    // Setup file uploads only where stream inputs exist (monitoring or dashboard monitoring section)
    for (let i = 0; i < 4; i++) {
        if (document.getElementById('videoFile' + i)) {
            setupFileUpload(i);
        }
    }

    // Initial stats and periodic polling (for analytics/dashboard/monitoring)
    if (document.getElementById('activeStreams') || document.getElementById('violationsTableBody')) {
        const statsPromise = updateSystemStats();
        pollInterval = setInterval(updateSystemStats, POLL_INTERVAL);
        connectWebSocket();
        
        // If on analytics page, load violations from database immediately
        // to prevent showing 0 counts
        const analyticsDateFilter = document.getElementById('analyticsDateFilter');
        if (analyticsDateFilter) {
            // Load data from database on page load
            setTimeout(() => {
                updateAnalyticsRange();
            }, 100);
        }
    }

    // Check for monitoring page elements - check multiple times to ensure DOM is ready
    const monitoringCheck = () => {
        const stream0 = document.getElementById('stream0');
        const streamUrl0 = document.getElementById('streamUrl0');
        
        console.log('🔍 Checking for monitoring page elements...', {
            stream0: !!stream0,
            streamUrl0: !!streamUrl0,
            bodyClass: document.body.className
        });
        
        if (stream0 || streamUrl0) {
            console.log('🎬 Monitoring page detected, setting up stream reconnection...');
            
            // Initial reconnection attempts
            setTimeout(() => {
                console.log('🔄 Initial reconnection attempt...');
                reconnectToActiveStreams();
            }, 500);
            
            // Continuous reconnection check every 5 seconds
            // This ensures streams reconnect even after page refresh or navigation
            reconnectionInterval = setInterval(() => {
                reconnectToActiveStreams();
            }, 5000); // Check every 5 seconds
            
            console.log('⏰ Continuous stream monitoring enabled (checks every 5s)');
            return true;
        }
        return false;
    };

    // Try to detect monitoring page immediately
    if (!monitoringCheck()) {
        // If not found, try again after a short delay (for slower loading)
        setTimeout(() => {
            if (!monitoringCheck()) {
                console.log('ℹ️ Not on monitoring page - stream reconnection disabled');
            }
        }, 300);
    }

    // Keyboard shortcuts for closing dropdown / modal
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            hideAuthModal();
            closeUserDropdown();
        }
    });

    window.addEventListener('beforeunload', () => {
        // Clean up all intervals before page unload
        console.log('🧹 Cleaning up intervals before page unload...');
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        
        // Clear reconnection interval
        if (reconnectionInterval) {
            clearInterval(reconnectionInterval);
            reconnectionInterval = null;
        }
        
        // Clear all stream polling intervals
        for (let i = 0; i < 4; i++) {
            if (streamPollIntervals[i]) {
                clearInterval(streamPollIntervals[i]);
                streamPollIntervals[i] = null;
            }
        }
        
        // Clear active streams set
        activeStreams.clear();
    });
}

/* ============================================
   STREAM SWAP LOGIC - FULL IMPLEMENTATION
   ============================================ */

let currentMainStreamIndex = 2; // Stream 3 is main by default

/**
 * Swap a small stream with the main stream
 * Physically moves DOM elements (content, videos, uploads, everything)
 */
function swapToMainScreen(smallStreamIndex) {
  if (smallStreamIndex === currentMainStreamIndex) return;
  
  const smallCard = document.querySelector(`.stream-card-wrapper.small[data-stream="${smallStreamIndex}"]`);
  const mainCard = document.querySelector('.stream-card-wrapper.main-stream');
  
  if (!smallCard || !mainCard) return;
  
  const oldMainIndex = parseInt(mainCard.dataset.stream);

  // Swap the actual content (DOM elements)
  const smallContent = smallCard.querySelector('.stream-card-content').innerHTML;
  const mainContent = mainCard.querySelector('.stream-card-content').innerHTML;

  smallCard.querySelector('.stream-card-content').innerHTML = mainContent;
  mainCard.querySelector('.stream-card-content').innerHTML = smallContent;

  // Swap data-stream attributes
  smallCard.dataset.stream = String(oldMainIndex);
  mainCard.dataset.stream = String(smallStreamIndex);

  currentMainStreamIndex = smallStreamIndex;

  // Re-init uploads after DOM swap
  for (let i = 0; i < 4; i++) {
    setupFileUploadForStream(i);
  }

  // Ensure main card visual state
  document.querySelectorAll('.stream-card-wrapper.main-stream')
    .forEach(c => c.classList.remove('main-stream'));
  mainCard.classList.add('main-stream');

  // >>> AUTO REFRESH PART <<<
  // 1. Refresh stats panel and charts
  if (typeof updateSystemStats === 'function') {
    updateSystemStats();
  }

  // 2. Restart frame polling for all streams so the new main one
  //    starts getting fresh frames immediately
  if (typeof pollStreamFrame === 'function') {
    for (let i = 0; i < 4; i++) {
      // Only poll if that stream is currently active
      const statusEl = document.getElementById(`status${i}`);
      if (statusEl && statusEl.textContent.trim().toLowerCase() === 'active') {
        pollStreamFrame(i);
      }
    }
  }

  console.log(`Swapped Stream ${smallStreamIndex} to main and refreshed streams.`);
}


/**
 * Enhanced file upload setup - call this after DOM swaps
 */
function setupFileUploadForStream(streamIndex) {
  const fileInput = document.getElementById(`videoFile${streamIndex}`);
  const uploadLabel = fileInput ? document.querySelector(`label[for="videoFile${streamIndex}"]`) : null;
  const selectedFile = document.getElementById(`selectedFile${streamIndex}`);
  const fileName = document.getElementById(`fileName${streamIndex}`);
  
  if (!fileInput || !uploadLabel || !selectedFile || !fileName) return;
  
  const uploadBox = uploadLabel.querySelector('.upload-box');
  
  // Remove old listeners
  const newFileInput = fileInput.cloneNode(true);
  fileInput.parentNode.replaceChild(newFileInput, fileInput);
  
  // Reattach listeners
  const freshInput = document.getElementById(`videoFile${streamIndex}`);
  
  freshInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
      fileName.textContent = file.name;
      selectedFile.style.display = 'flex';
    } else {
      selectedFile.style.display = 'none';
    }
  });
  
  uploadLabel.addEventListener('dragover', function(e) {
    e.preventDefault();
    if (uploadBox) {
      uploadBox.style.borderColor = 'var(--primary-green)';
      uploadBox.style.background = 'rgba(124, 179, 66, 0.08)';
    }
  });
  
  uploadLabel.addEventListener('dragleave', function(e) {
    if (uploadBox) {
      uploadBox.style.borderColor = 'var(--light-gray)';
      uploadBox.style.background = 'rgba(0, 0, 0, 0.02)';
    }
  });
  
  uploadLabel.addEventListener('drop', function(e) {
    e.preventDefault();
    if (uploadBox) {
      uploadBox.style.borderColor = 'var(--light-gray)';
      uploadBox.style.background = 'rgba(0, 0, 0, 0.02)';
    }
    if (e.dataTransfer.files.length > 0) {
      freshInput.files = e.dataTransfer.files;
      freshInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
}

/**
 * Original clearStreamFile - still works with new layout
 */
function clearStreamFileMonitoring(index) {
  const input = document.getElementById(`videoFile${index}`);
  const selectedFile = document.getElementById(`selectedFile${index}`);
  if (input) input.value = '';
  if (selectedFile) selectedFile.style.display = 'none';
}

// Call this in your existing init code:
// for (let i = 0; i < 4; i++) {
//   setupFileUploadForStream(i);
// }


// Main entry
async function initializeApp() {
    // Load configuration first
    await loadConfig();

    console.log('🚀 Traffic Monitoring System');
    console.log('📡 API URL:', API_BASE_URL);
    console.log('🔌 WebSocket URL:', WS_BASE_URL);

    const isDashboard = document.body.classList.contains('dashboard-body');
    if (isDashboard) {
        initDashboardPages();
    } else {
        initLandingPage();
    }
}

// Global helper function for manual stream reconnection
window.reconnectStreams = function() {
    console.log('🔧 Manual reconnection triggered...');
    reconnectToActiveStreams();
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Hamburger Menu Logic
document.addEventListener('DOMContentLoaded', () => {
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const navbarMenu = document.getElementById('navbar-menu');

    if (mobileBtn && navbarMenu) {
        mobileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            navbarMenu.classList.toggle('active');
            
            // Icon Toggle
            const icon = mobileBtn.querySelector('i');
            if (navbarMenu.classList.contains('active')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (navbarMenu.classList.contains('active') && 
                !navbarMenu.contains(e.target) && 
                !mobileBtn.contains(e.target)) {
                closeMobileMenu();
            }
        });
    }
});

function closeMobileMenu() {
    const navbarMenu = document.getElementById('navbar-menu');
    const mobileBtn = document.getElementById('mobile-menu-btn');
    
    if (navbarMenu) {
        navbarMenu.classList.remove('active');
        if (mobileBtn) {
            const icon = mobileBtn.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        }
    }
}


/* ==========================================================================
   APPEND THIS TO THE BOTTOM OF SCRIPT.JS
   (This overrides previous functions to add Pagination & New Charts)
   ========================================================================== */

// --- New Global Variables for Pagination & Charts ---
var violationsByStreamChart = null;
var speedDistributionChart = null;
var currentViolations = []; // Stores data for pagination
var currentPage = 1;
const ROWS_PER_PAGE = 5;

// --- 1. Updated Chart Initialization (Adds 2 New Graphs) ---
function initCharts() {
    // Canvas Elements
    const ctxType = document.getElementById('violationsByTypeChart');
    const ctxStreams = document.getElementById('streamsVehiclesChart');
    const ctxTime = document.getElementById('violationsOverTimeChart');
    const ctxStreamDist = document.getElementById('violationsByStreamChart'); // New
    const ctxSpeed = document.getElementById('speedDistributionChart');       // New

    // 1. Violations Type (Bar)
    if (ctxType && window.Chart) {
        if (violationsByTypeChart) violationsByTypeChart.destroy();
        violationsByTypeChart = new Chart(ctxType, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Violations', data: [], backgroundColor: 'rgba(6, 182, 212, 0.6)' }] },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
    }
    
    // 2. Streams vs Vehicles (Doughnut)
    if (ctxStreams && window.Chart) {
        if (streamsVehiclesChart) streamsVehiclesChart.destroy();
        streamsVehiclesChart = new Chart(ctxStreams, {
            type: 'doughnut',
            data: { labels: ['Active Streams', 'Total Vehicles'], datasets: [{ data: [0, 0], backgroundColor: ['#10b981', '#3b82f6'] }] },
            options: { responsive: true }
        });
    }

    // 3. Violations Over Time (Line)
    if (ctxTime && window.Chart) {
        if (violationsOverTimeChart) violationsOverTimeChart.destroy();
        violationsOverTimeChart = new Chart(ctxTime, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Violations', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.2)', fill: true }] },
            options: { responsive: true }
        });
    }

    // 4. NEW: Violations by Stream (Pie)
    if (ctxStreamDist && window.Chart) {
        if (violationsByStreamChart) violationsByStreamChart.destroy();
        violationsByStreamChart = new Chart(ctxStreamDist, {
            type: 'pie',
            data: { 
                labels: [], 
                datasets: [{ 
                    data: [], 
                    backgroundColor: ['#f87171', '#fb923c', '#fbbf24', '#a3e635'] 
                }] 
            },
            options: { responsive: true }
        });
    }

    // 5. NEW: Speed Distribution (Bar)
    if (ctxSpeed && window.Chart) {
        if (speedDistributionChart) speedDistributionChart.destroy();
        speedDistributionChart = new Chart(ctxSpeed, {
            type: 'bar',
            data: {
                labels: ['0-30', '30-60', '60-90', '90+'],
                datasets: [{
                    label: 'Vehicle Count',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(139, 92, 246, 0.6)'
                }]
            },
            options: { responsive: true }
        });
    }
}

// --- 2. Updated Chart Data Logic ---
function updateChartsFromStats(data) {
    if (!data) return;

    // Update Type Chart
    if (violationsByTypeChart && data.violation_summary) {
        const labels = Object.keys(data.violation_summary);
        const values = labels.map(k => data.violation_summary[k]);
        violationsByTypeChart.data.labels = labels.map(l => l.replace('_', ' ').toUpperCase());
        violationsByTypeChart.data.datasets[0].data = values;
        violationsByTypeChart.update();
    }

    // Update Stream/Vehicle Chart
    if (streamsVehiclesChart) {
        streamsVehiclesChart.data.datasets[0].data = [data.active_streams || 0, data.total_vehicles || 0];
        streamsVehiclesChart.update();
    }

    const violationList = data.violations || [];

    // Update Timeline Chart
    if (violationsOverTimeChart && violationList.length > 0) {
        const buckets = {};
        violationList.forEach(v => {
            if (!v.timestamp) return;
            const d = new Date(v.timestamp);
            if (isNaN(d.getTime())) return;
            const key = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            buckets[key] = (buckets[key] || 0) + 1;
        });
        const sortedLabels = Object.keys(buckets).sort();
        const values = sortedLabels.map(k => buckets[k]);
        violationsOverTimeChart.data.labels = sortedLabels;
        violationsOverTimeChart.data.datasets[0].data = values;
        violationsOverTimeChart.update();
    }

    // Update New Stream Pie Chart
    if (violationsByStreamChart && violationList.length > 0) {
        const streamCounts = {};
        violationList.forEach(v => {
            const key = `Stream ${v.stream_id}`;
            streamCounts[key] = (streamCounts[key] || 0) + 1;
        });
        violationsByStreamChart.data.labels = Object.keys(streamCounts);
        violationsByStreamChart.data.datasets[0].data = Object.values(streamCounts);
        violationsByStreamChart.update();
    }

    // Update New Speed Bar Chart
    if (speedDistributionChart && violationList.length > 0) {
        const speedBuckets = [0, 0, 0, 0]; // 0-30, 30-60, 60-90, 90+
        violationList.forEach(v => {
            const speed = v.speed_kmh || 0;
            if (speed < 30) speedBuckets[0]++;
            else if (speed < 60) speedBuckets[1]++;
            else if (speed < 90) speedBuckets[2]++;
            else speedBuckets[3]++;
        });
        speedDistributionChart.data.datasets[0].data = speedBuckets;
        speedDistributionChart.update();
    }
}

// --- 3. New Pagination Logic ---
function renderPaginationTable() {
    const tbody = document.getElementById('violationsTableBody') || document.getElementById('allViolationsTable');
    if (!tbody) return;

    const validData = currentViolations || [];
    const totalItems = validData.length;
    
    // Slice data for current page
    const startIndex = (currentPage - 1) * ROWS_PER_PAGE;
    const endIndex = Math.min(startIndex + ROWS_PER_PAGE, totalItems);
    const pageData = validData.slice(startIndex, endIndex);

    // Update Footer Info
    const paginationContainer = document.getElementById('tablePagination');
    if (paginationContainer) {
        if (totalItems === 0) {
            paginationContainer.style.display = 'none';
        } else {
            paginationContainer.style.display = 'flex';
            document.getElementById('pageStart').textContent = startIndex + 1;
            document.getElementById('pageEnd').textContent = endIndex;
            document.getElementById('totalRows').textContent = totalItems;
        }
    }

    // Render Table Rows
    if (totalItems === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="table-empty-state"><i class="fas fa-inbox"></i><p>No data available</p></td></tr>';
        return;
    }

    tbody.innerHTML = pageData.map(v => `
        <tr>
            <td>Stream ${v.stream_id}</td>
            <td>${(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
            <td>${formatSpeed(v.speed_kmh)}</td>
            <td>${v.signal_state || 'N/A'}</td>
            <td>${formatTime(v.timestamp)}</td>
        </tr>
    `).join('');

    renderPaginationControls(totalItems);
}

function renderPaginationControls(totalItems) {
    const controlsContainer = document.getElementById('paginationControls');
    if (!controlsContainer) return;

    const totalPages = Math.ceil(totalItems / ROWS_PER_PAGE);
    let buttonsHtml = '';

    // Prev Button
    buttonsHtml += `<button class="page-btn" onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}><i class="fas fa-chevron-left"></i></button>`;

    // Simple Logic: 1, 2, 3 ... Last
    if (totalPages <= 7) {
        for (let i = 1; i <= totalPages; i++) {
            buttonsHtml += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
        }
    } else {
        // Complex logic for many pages
        if (currentPage <= 4) {
            for (let i = 1; i <= 5; i++) buttonsHtml += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
            buttonsHtml += `<span class="page-dots">...</span>`;
            buttonsHtml += `<button class="page-btn" onclick="changePage(${totalPages})">${totalPages}</button>`;
        } else if (currentPage >= totalPages - 3) {
            buttonsHtml += `<button class="page-btn" onclick="changePage(1)">1</button>`;
            buttonsHtml += `<span class="page-dots">...</span>`;
            for (let i = totalPages - 4; i <= totalPages; i++) buttonsHtml += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
        } else {
            buttonsHtml += `<button class="page-btn" onclick="changePage(1)">1</button>`;
            buttonsHtml += `<span class="page-dots">...</span>`;
            for (let i = currentPage - 1; i <= currentPage + 1; i++) buttonsHtml += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
            buttonsHtml += `<span class="page-dots">...</span>`;
            buttonsHtml += `<button class="page-btn" onclick="changePage(${totalPages})">${totalPages}</button>`;
        }
    }

    // Next Button
    buttonsHtml += `<button class="page-btn" onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}><i class="fas fa-chevron-right"></i></button>`;

    controlsContainer.innerHTML = buttonsHtml;
}

window.changePage = function(newPage) {
    const totalPages = Math.ceil((currentViolations.length || 0) / ROWS_PER_PAGE);
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        renderPaginationTable();
    }
};

// --- 4. Overridden Data Fetching (Links Data to Pagination) ---
async function updateSystemStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();
        systemStats = data;

        // Update Counts
        if(document.getElementById('activeStreams')) document.getElementById('activeStreams').textContent = data.active_streams ?? 0;
        if(document.getElementById('totalVehicles')) document.getElementById('totalVehicles').textContent = data.total_vehicles ?? 0;
        
        // If NO filter is active, update dashboard table/charts
        const isFiltered = document.getElementById('analyticsDateFilter') && document.getElementById('analyticsDateFilter').value !== 'all';
        if (!isFiltered) {
            if(document.getElementById('totalViolations')) document.getElementById('totalViolations').textContent = data.total_violations ?? 0;
            if(document.getElementById('speedViolations')) document.getElementById('speedViolations').textContent = data.violation_summary?.speed ?? 0;
            
            // Set global data and render table
            currentViolations = data.violations || [];
            
            // Only update table if on a page that uses it
            if (document.getElementById('violationsTableBody') || document.getElementById('allViolationsTable')) {
                renderPaginationTable();
            }
            updateChartsFromStats(data);
        }
    } catch (e) { console.error('Stats fetch error', e); }
}

async function updateAnalyticsRange() {
    const select = document.getElementById('analyticsDateFilter');
    if (!select) return;
    
    analyticsDateRange = select.value;
    const datePicker = document.getElementById('analyticsDatePicker');
    
    if (analyticsDateRange === 'custom') {
        datePicker.style.display = 'block';
        if (!datePicker.value) return; 
    } else {
        datePicker.style.display = 'none';
    }
    
    // Build Query
    let query = '?limit=500';
    if (analyticsDateRange === 'custom' && datePicker.value) {
        query += `&specific_date=${encodeURIComponent(datePicker.value)}`;
    } else if (analyticsDateRange !== 'all') {
        query += `&date_range=${encodeURIComponent(analyticsDateRange)}`;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/db/violations${query}`, { headers: getAuthHeaders() });
        const data = await response.json();
        const violations = data.violations || [];

        // Update Summaries
        const summary = { speed: 0, red_light: 0, stop_line: 0, lane_change: 0 };
        violations.forEach(v => {
            if (summary.hasOwnProperty(v.violation_type)) summary[v.violation_type]++;
        });

        if(document.getElementById('totalViolations')) document.getElementById('totalViolations').textContent = violations.length;
        if(document.getElementById('speedViolations')) document.getElementById('speedViolations').textContent = summary.speed;
        if(document.getElementById('redLightCount')) document.getElementById('redLightCount').textContent = summary.red_light;
        if(document.getElementById('laneChangeCount')) document.getElementById('laneChangeCount').textContent = summary.lane_change;

        // Set Pagination Data & Render
        currentViolations = violations;
        currentPage = 1; 
        renderPaginationTable();

        // Update Charts
        const statsLike = { violation_summary: summary, violations: violations };
        updateChartsFromStats(statsLike);

    } catch (err) { console.error('Analytics fetch error', err); }
}
// --- MONITORING LAYOUT SWAP LOGIC ---

// Track which stream index is currently in the big card UI
let bigStreamIndex = 2; // default: Stream 3 in big card

function swapToBigScreen(smallIndex) {
  // smallIndex is the logical stream index (0,1,3) from the small cards
  if (smallIndex === bigStreamIndex) return;

  const bigCard = document.querySelector('.big-stream-card');
  const smallCard = document.querySelector(`.small-stream-card[data-stream="${smallIndex}"]`);
  if (!bigCard || !smallCard) return;

  const oldBigIndex = bigStreamIndex;
  bigStreamIndex = smallIndex;

  // Update header titles
  const bigTitleEl = bigCard.querySelector('h3');
  const smallTitleEl = smallCard.querySelector('h3');
  if (bigTitleEl && smallTitleEl) {
    const bigBase = smallTitleEl.textContent.replace(' (Main)', '');
    const smallBase = bigTitleEl.textContent.replace(' (Main)', '');
    bigTitleEl.textContent = `${bigBase} (Main)`;
    smallTitleEl.textContent = smallBase;
  }

  // Swap DOM IDs for all elements associated with the streams
  swapStreamElements(oldBigIndex, smallIndex);

  // Update data-stream attribute so next click still works
  smallCard.dataset.stream = String(oldBigIndex);

  // Visual active state
  document.querySelectorAll('.big-stream-card').forEach(c => c.classList.remove('active-big'));
  bigCard.classList.add('active-big');
}

/**
 * Swap all UI elements (ids + event bindings) between two stream indices
 * without changing your backend logic.
 */
function swapStreamElements(a, b) {
  const idBases = ['stream', 'placeholder', 'status', 'streamUrl', 'videoFile', 'selectedFile', 'fileName'];

  idBases.forEach(base => {
    const elA = document.getElementById(`${base}${a}`);
    const elB = document.getElementById(`${base}${b}`);

    if (!elA && !elB) return;

    // Temporary id for A so we can swap
    if (elA) elA.id = `${base}${a}_tmp`;

    if (elB) elB.id = `${base}${a}`;
    if (elA) elA.id = `${base}${b}`;
  });

  // Fix upload buttons / labels for these two indices
  setupFileUpload(a);
  setupFileUpload(b);

  // Update buttons' onclick handlers
  fixStreamButtonsForIndex(a);
  fixStreamButtonsForIndex(b);
}

function fixStreamButtonsForIndex(index) {
  const card = document.querySelector(
    `.small-stream-card[data-stream="${index}"], .big-stream-card[data-stream="${index}"]`
  );
  if (!card) return;

  const startBtn = card.querySelector('button.btn.btn-primary');
  const stopBtn = card.querySelector('button.btn.btn-danger');
  const uploadBtn = card.querySelector('button.btn.btn-accent');

  if (startBtn) startBtn.onclick = (e) => { e.stopPropagation(); startStream(index); };
  if (stopBtn) stopBtn.onclick = (e) => { e.stopPropagation(); stopStream(index); };
  if (uploadBtn) uploadBtn.onclick = (e) => { e.stopPropagation(); uploadStreamVideo(index); };
}

/**
 * Extend existing setupFileUpload so it can be safely called multiple times
 * and used with the new layout.
 */
function setupFileUpload(index) {
  const fileInput = document.getElementById(`videoFile${index}`);
  const uploadLabel = fileInput ? document.querySelector(`label[for="videoFile${index}"]`) : null;
  const selectedFile = document.getElementById(`selectedFile${index}`);
  const fileName = document.getElementById(`fileName${index}`);

  if (!fileInput || !uploadLabel || !selectedFile || !fileName) return;

  const uploadBox = uploadLabel.querySelector('.upload-box');

  fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
      fileName.textContent = file.name;
      selectedFile.style.display = 'flex';
    } else {
      selectedFile.style.display = 'none';
    }
  };

  uploadLabel.ondragover = (e) => {
    e.preventDefault();
    if (!uploadBox) return;
    uploadBox.style.borderColor = 'var(--primary-green)';
    uploadBox.style.background = 'rgba(124,179,66,0.08)';
  };

  uploadLabel.ondragleave = () => {
    if (!uploadBox) return;
    uploadBox.style.borderColor = 'var(--light-gray)';
    uploadBox.style.background = 'rgba(0,0,0,0.02)';
  };

  uploadLabel.ondrop = (e) => {
    e.preventDefault();
    if (uploadBox) {
      uploadBox.style.borderColor = 'var(--light-gray)';
      uploadBox.style.background = 'rgba(0,0,0,0.02)';
    }
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
  };
}
