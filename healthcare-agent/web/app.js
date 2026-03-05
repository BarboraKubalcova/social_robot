const messagesDiv = document.getElementById('messages');
const chatWindow = document.getElementById('chat-window');

const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const debugContent = document.getElementById('debug-content');

const API_URL = "http://localhost:8000/chat";

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

// On load: keep the greeting visible at the bottom
requestAnimationFrame(() => scrollToBottom(chatWindow));