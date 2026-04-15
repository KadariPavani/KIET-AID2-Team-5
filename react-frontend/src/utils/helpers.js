// ── Toast system ────────────────────────────────────────────
// No batching — every violation shown individually.
// Slides in from the right, stacks upward at bottom-right.

function _ensureStyles() {
  if (document.getElementById('toast-styles')) return;
  const style = document.createElement('style');
  style.id = 'toast-styles';
  style.textContent = `
    .toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      padding: 1rem 1.5rem;
      background: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 10000;
      max-width: 350px;
      font-size: 0.9rem;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #212121;
      animation: slideIn 0.3s ease forwards;
      transition: bottom 0.3s ease;
    }
    .toast.toast-exit {
      animation: slideOut 0.3s ease forwards;
    }
    .toast.success { border-left: 4px solid #4caf50; }
    .toast.error   { border-left: 4px solid #ef4444; }
    .toast.info    { border-left: 4px solid #42a5f5; }
    @keyframes slideIn {
      from { transform: translateX(400px); opacity: 0; }
      to   { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
      from { transform: translateX(0); opacity: 1; }
      to   { transform: translateX(400px); opacity: 0; }
    }
  `;
  document.head.appendChild(style);
}

// Track active toasts — max 3 visible at a time
const MAX_TOASTS = 3;
const _activeToasts = [];

function _repositionToasts() {
  let offset = 2; // rem from bottom
  for (let i = _activeToasts.length - 1; i >= 0; i--) {
    _activeToasts[i].style.bottom = offset + 'rem';
    offset += _activeToasts[i].offsetHeight / 16 + 0.6;
  }
}

function _killToast(toast) {
  // Remove from array immediately (synchronous)
  const idx = _activeToasts.indexOf(toast);
  if (idx !== -1) _activeToasts.splice(idx, 1);
  // Animate out then remove DOM node
  toast.classList.add('toast-exit');
  toast.addEventListener('animationend', () => {
    toast.remove();
    _repositionToasts();
  }, { once: true });
  // Fallback: force remove if animationend never fires
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 400);
}

function _showToastDom(message, type) {
  _ensureStyles();

  // Remove oldest if at max
  while (_activeToasts.length >= MAX_TOASTS) {
    _killToast(_activeToasts[0]);
  }

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  _activeToasts.push(toast);
  _repositionToasts();

  setTimeout(() => {
    if (_activeToasts.includes(toast)) {
      _killToast(toast);
      _repositionToasts();
    }
  }, 4000);
}

export const showToast = (message, type = 'success') => {
  _showToastDom(message, type);
};

export const formatTime = (isoString) => {
  if (!isoString) return 'N/A';
  const date = new Date(isoString);
  return isNaN(date.getTime()) ? 'N/A' : date.toLocaleTimeString();
};

export const formatSpeed = (speed) => {
  return typeof speed === 'number' && !isNaN(speed) ? `${speed.toFixed(1)} km/h` : 'N/A';
};

export const formatDate = (isoString) => {
  if (!isoString) return 'N/A';
  const date = new Date(isoString);
  return isNaN(date.getTime()) ? 'N/A' : date.toLocaleDateString();
};
