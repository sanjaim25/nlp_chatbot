/* ═══════════════════════════════════════════════════════════════
   NeuralBot — Main JavaScript (Redesigned)
   ═══════════════════════════════════════════════════════════════ */

const chatBox  = document.getElementById('chat-box');
const inputEl  = document.getElementById('user-input');
const sendBtn  = document.getElementById('send-btn');
const charCnt  = document.getElementById('char-count');

const MAX_CHARS = 300;
let msgCount = 0, totalConf = 0, totalLat = 0;
let isBusy = false;

/* ── Auto-resize textarea ──────────────────────────────────── */
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  const len = inputEl.value.length;
  charCnt.textContent = `${len}/${MAX_CHARS}`;
  charCnt.className = 'char-count' + (len > 270 ? ' danger' : len > 230 ? ' warn' : '');
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* ── Tab switching ──────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('pane-' + name).classList.add('active');
}

/* ── Helpers ────────────────────────────────────────────────── */
function confClass(c) {
  if (c >= 70) return 'conf-high';
  if (c >= 35) return 'conf-mid';
  return 'conf-low';
}
function timeStr() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function removeWelcome() {
  const w = document.getElementById('welcome');
  if (w) { w.style.animation = 'fadeUp 0.2s ease reverse'; setTimeout(() => w.remove(), 180); }
}

/* ── Update radial gauge ────────────────────────────────────── */
function updateRadial(conf) {
  const arc   = document.getElementById('radial-arc');
  const label = document.getElementById('radial-label');
  if (!arc || !label) return;
  const circumference = 2 * Math.PI * 34; // r=34
  const fill = (conf / 100) * circumference;
  arc.setAttribute('stroke-dasharray', `${fill} ${circumference - fill}`);
  // colour
  if (conf >= 70) arc.style.stroke = '#34d399';
  else if (conf >= 35) arc.style.stroke = '#fbbf24';
  else arc.style.stroke = '#f87171';
  label.textContent = conf + '%';
}

/* ── Append message ─────────────────────────────────────────── */
function buildAvatar(role) {
  const av = document.createElement('div');
  av.className = `avatar ${role}`;
  if (role === 'bot') {
    av.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/><circle cx="12" cy="16" r="1"/></svg>`;
  } else {
    av.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
  }
  return av;
}

function appendMsg(text, role, meta = null) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const av   = buildAvatar(role);
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);

  // meta row (timestamp + confidence pill)
  const metaEl = document.createElement('div');
  metaEl.className = 'meta';
  if (meta) {
    metaEl.innerHTML = `
      <span class="conf-pill ${confClass(meta.confidence)}">${meta.confidence}%</span>
      <span>${meta.latency_ms}ms</span>
      <span>${timeStr()}</span>`;
  } else {
    metaEl.innerHTML = `<span>${timeStr()}</span>`;
  }
  wrap.appendChild(metaEl);

  row.appendChild(av);
  row.appendChild(wrap);
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
  return row;
}

/* ── Typing indicator ───────────────────────────────────────── */
function showTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="avatar bot">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/><circle cx="12" cy="16" r="1"/></svg>
    </div>
    <div class="bubble-wrap">
      <div class="bubble bot">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>`;
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function removeTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}

/* ── Update sidebar stats ───────────────────────────────────── */
function updateStats(conf, lat) {
  msgCount++;
  totalConf += conf;
  totalLat  += lat;

  const ac = (totalConf / msgCount).toFixed(1);
  const al = (totalLat  / msgCount).toFixed(0);

  document.getElementById('msg-count').textContent = msgCount;
  document.getElementById('avg-conf').textContent  = ac;
  document.getElementById('avg-lat').textContent   = al;
  document.getElementById('conf-pct').textContent  = ac + '%';
  document.getElementById('conf-bar').style.width  = ac + '%';

  // Right panel
  document.getElementById('r-conf').textContent = conf + '%';
  document.getElementById('r-lat').textContent  = lat  + 'ms';
  updateRadial(conf);
}

/* ── Update right panel intent ──────────────────────────────── */
function updatePanel(data) {
  const pill = document.getElementById('r-intent');
  if (pill) pill.textContent = data.intent && data.intent !== 'unknown' ? data.intent : '—';
}

/* ── Preset send ────────────────────────────────────────────── */
function send(text) {
  inputEl.value = text;
  inputEl.dispatchEvent(new Event('input'));
  sendMessage();
}

/* ── Clear chat ─────────────────────────────────────────────── */
function clearChat() {
  chatBox.innerHTML = '';
  msgCount = 0; totalConf = 0; totalLat = 0;

  ['msg-count','avg-conf','avg-lat'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = id === 'msg-count' ? '0' : '—';
  });
  const pct = document.getElementById('conf-pct'); if (pct) pct.textContent = '0%';
  const bar = document.getElementById('conf-bar'); if (bar) bar.style.width = '0';
  const rc  = document.getElementById('r-conf');  if (rc)  rc.textContent = '—';
  const rl  = document.getElementById('r-lat');   if (rl)  rl.textContent = '— ms';
  const ri  = document.getElementById('r-intent');if (ri)  ri.textContent = '—';
  updateRadial(0);

  // Re-inject welcome
  const w = document.createElement('div');
  w.className = 'welcome-screen';
  w.id = 'welcome';
  w.innerHTML = `
    <div class="welcome-glow"></div>
    <div class="welcome-icon-wrap">
      <div class="welcome-icon">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>
      </div>
    </div>
    <h2 class="welcome-title">Hello, I'm NeuralBot</h2>
    <p class="welcome-desc">Powered by a Bidirectional LSTM model trained with NLP. Ask me about AI, machine learning, deep learning — or just say hi!</p>
    <div class="starter-grid">
      <button class="starter-card" onclick="send('What is deep learning?')">
        <span class="starter-icon">🧠</span><span class="starter-text">What is deep learning?</span>
      </button>
      <button class="starter-card" onclick="send('Explain neural networks')">
        <span class="starter-icon">🔗</span><span class="starter-text">Explain neural networks</span>
      </button>
      <button class="starter-card" onclick="send('Tell me a joke')">
        <span class="starter-icon">😄</span><span class="starter-text">Tell me a joke</span>
      </button>
      <button class="starter-card" onclick="send('What is NLP?')">
        <span class="starter-icon">💬</span><span class="starter-text">What is NLP?</span>
      </button>
    </div>`;
  chatBox.appendChild(w);
}

/* ── Main send logic ────────────────────────────────────────── */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isBusy) return;

  isBusy = true;
  sendBtn.disabled = true;

  removeWelcome();

  // Switch to Response tab
  switchTab('response');

  appendMsg(text, 'user');
  inputEl.value = '';
  inputEl.style.height = 'auto';
  charCnt.textContent = `0/${MAX_CHARS}`;
  charCnt.className = 'char-count';

  showTyping();

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeTyping();
    appendMsg(data.response, 'bot', { confidence: data.confidence, latency_ms: data.latency_ms });
    updateStats(data.confidence, data.latency_ms);
    updatePanel(data);
  } catch (err) {
    removeTyping();
    const row = appendMsg('⚠️  Could not reach the server.', 'bot');
    row.querySelector('.bubble').classList.add('error');
  }

  isBusy = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

/* ── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  inputEl.focus();
  charCnt.textContent = `0/${MAX_CHARS}`;
  updateRadial(0);
});
