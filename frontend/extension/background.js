// MedGuard AI Chrome Extension — background.js
// Service worker that maintains WebSocket connection and fires notifications

const WS_URL = 'ws://localhost:8000/ws/events';
let socket = null;
let reconnectTimeout = null;
let alertCount = 0;

const SEVERITY_COLORS = {
  critical: '#ff2d55',
  high: '#ff9f0a',
  medium: '#ffd60a',
  low: '#34c759',
};

const EVENT_ICONS = {
  fall: '⬇️',
  seizure: '⚡',
  cardiac: '❤️',
  unconscious: '💤',
  facial_distress: '😟',
};

function connect() {
  if (socket?.readyState === WebSocket.OPEN) return;

  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log('[MedGuard] WebSocket connected');
    chrome.storage.local.set({ wsStatus: 'connected', lastConnected: Date.now() });
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'medical_event') {
        handleMedicalEvent(data);
      }
    } catch (e) {
      console.error('[MedGuard] Parse error:', e);
    }
  };

  socket.onclose = () => {
    console.log('[MedGuard] WebSocket disconnected, reconnecting...');
    chrome.storage.local.set({ wsStatus: 'disconnected' });
    reconnectTimeout = setTimeout(connect, 5000);
  };

  socket.onerror = (err) => {
    console.error('[MedGuard] WebSocket error:', err);
    chrome.storage.local.set({ wsStatus: 'error' });
  };
}

function handleMedicalEvent(data) {
  alertCount++;

  // Store in history
  chrome.storage.local.get(['alertHistory'], ({ alertHistory = [] }) => {
    alertHistory.unshift({
      ...data,
      receivedAt: Date.now(),
    });
    // Keep last 50 alerts
    chrome.storage.local.set({
      alertHistory: alertHistory.slice(0, 50),
      alertCount,
    });
  });

  // Update badge
  chrome.action.setBadgeText({ text: alertCount.toString() });
  chrome.action.setBadgeBackgroundColor({
    color: SEVERITY_COLORS[data.severity] || '#ff2d55'
  });

  // Fire browser notification
  if (data.severity === 'critical' || data.severity === 'high') {
    const icon = EVENT_ICONS[data.event_type] || '⚠️';
    chrome.notifications.create(`medguard_${Date.now()}`, {
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: `${icon} MedGuard: ${data.event_type.replace('_', ' ').toUpperCase()}`,
      message: `Camera: ${data.camera_id} | Severity: ${data.severity.toUpperCase()} | Confidence: ${(data.confidence * 100).toFixed(0)}%`,
      priority: 2,
      requireInteraction: data.severity === 'critical',
    });
  }
}

// Click notification → open dashboard
chrome.notifications.onClicked.addListener(() => {
  chrome.tabs.create({ url: 'http://localhost:3000' });
});

// Reset badge when popup opens
chrome.action.onClicked.addListener(() => {
  chrome.action.setBadgeText({ text: '' });
  alertCount = 0;
  chrome.storage.local.set({ alertCount: 0 });
});

// Start connection
connect();

// Keepalive ping every 25s
setInterval(() => {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send('ping');
  } else {
    connect();
  }
}, 25000);
