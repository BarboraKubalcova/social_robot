const messagesDiv = document.getElementById('messages');
const chatWindow = document.getElementById('chat-window');

const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const debugContent = document.getElementById('debug-content');

const API_URL = "http://localhost:8000/chat";
const APPOINTMENTS_URL = "http://localhost:8000/chat/appointments";

const statusDot = document.querySelector('.status-pill .dot');
const statusText = document.getElementById('status-text');

const HEALTH_URL = "http://localhost:8000/health"; 

// Autoscroll behavior:
// - If the user is already near the bottom, keep them pinned to the bottom.
// - If they scroll up to read, don't yank them back down.
function isNearBottom(el, thresholdPx = 120) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < thresholdPx;
}

function scrollToBottom(el) {
  el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
}


// async function checkServerStatus() {
//   try {
//     const res = await fetch(HEALTH_URL, { method: "GET" });

//     if (!res.ok) throw new Error();

//     statusDot.style.background = "#27f59b";
//     statusText.textContent = "Online";

//   } catch (err) {

//     statusDot.style.background = "#ff4d4d";
//     statusText.textContent = "Offline";

//   }
// }

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, 'user');
  userInput.value = '';
  userInput.focus();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // 'X-API-Key': 'patient_123' // Optional auth
      },
      body: JSON.stringify({ message: text })
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    addMessage(data.response, 'assistant');

    debugContent.textContent = JSON.stringify({
      intent: data.intent,
      history_length: data.history?.length ?? 0
    }, null, 2);

  } catch (error) {
    addMessage("Sorry, something went wrong. Please try again.", 'assistant');
    console.error(error);
  }
}

function addMessage(text, role) {
  const shouldStickToBottom = isNearBottom(chatWindow);

  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = text;
  messagesDiv.appendChild(div);

  // Wait for layout, then scroll (prevents occasional off-by-a-bit issues)
  if (shouldStickToBottom) {
    requestAnimationFrame(() => scrollToBottom(chatWindow));
  }
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});

// Simple speech-to-text stub
const micBtn = document.getElementById('mic-btn');
if ('webkitSpeechRecognition' in window) {
  const recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    sendMessage();
  };

  micBtn.addEventListener('click', () => {
    recognition.start();
  });
} else {
  micBtn.style.display = 'none';
}

// run immediately
// checkServerStatus();
// setInterval(checkServerStatus, 5000);

// ── Timetable ────────────────────────────────────────
const timetableGrid = document.getElementById('timetable');
const refreshBtn = document.getElementById('refresh-btn');

async function loadTimetable() {
  try {
    const res = await fetch(APPOINTMENTS_URL);
    if (!res.ok) throw new Error(res.statusText);
    const slots = await res.json();

    // Group slots by date
    const days = new Map();
    const times = new Set();
    for (const s of slots) {
      if (!days.has(s.date)) days.set(s.date, {});
      days.get(s.date)[s.time] = s;
      times.add(s.time);
    }

    const sortedTimes = [...times].sort();
    const dayEntries = [...days.entries()]; // [[date, slotMap], ...]
    timetableGrid.style.gridTemplateColumns = `auto repeat(${dayEntries.length}, 1fr)`;

    timetableGrid.innerHTML = '';

    // Header row: empty corner + day labels
    const corner = document.createElement('div');
    corner.className = 'tt-cell tt-header';
    timetableGrid.appendChild(corner);

    for (const [dateStr, slotMap] of dayEntries) {
      const firstSlot = Object.values(slotMap)[0];
      const th = document.createElement('div');
      th.className = 'tt-cell tt-header';
      th.textContent = `${firstSlot.day.charAt(0).toUpperCase() + firstSlot.day.slice(1, 3)}\n${dateStr.slice(5)}`; // e.g. "Mo\n06-01"
      timetableGrid.appendChild(th);
    }

    // One row per time slot
    for (const t of sortedTimes) {
      const label = document.createElement('div');
      label.className = 'tt-cell tt-day-label';
      label.textContent = t;
      timetableGrid.appendChild(label);

      for (const [dateStr, slotMap] of dayEntries) {
        const cell = document.createElement('div');
        const slot = slotMap[t];
        if (slot) {
          cell.className = `tt-cell ${slot.status}`;
          cell.textContent = slot.status === 'free' ? '✓' : '✗';
          cell.title = `${slot.id} — ${slot.date} ${slot.time} — ${slot.status}`;
        } else {
          cell.className = 'tt-cell';
          cell.textContent = '—';
        }
        timetableGrid.appendChild(cell);
      }
    }
  } catch (err) {
    console.error('Failed to load timetable:', err);
    timetableGrid.innerHTML = '<p style="color:var(--muted);padding:8px;">Could not load appointments.</p>';
  }
}

refreshBtn.addEventListener('click', loadTimetable);

// Load timetable on start
loadTimetable();

// Auto-refresh timetable after every chat message
const _origSendMessage = sendMessage;
sendMessage = async function () {
  await _origSendMessage();
  loadTimetable();
};

// On load: keep the greeting visible at the bottom
requestAnimationFrame(() => scrollToBottom(chatWindow));