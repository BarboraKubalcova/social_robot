const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const debugContent = document.getElementById('debug-content');

const API_URL = "http://localhost:8000/chat";

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message
    addMessage(text, 'user');
    userInput.value = '';

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
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();

        // Add assistant response
        addMessage(data.response, 'assistant');

        // Update debug
        debugContent.textContent = JSON.stringify({
            intent: data.intent,
            history_length: data.history.length
        }, null, 2);

    } catch (error) {
        addMessage("Sorry, something went wrong. Please try again.", 'assistant');
        console.error(error);
    }
}

function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Simple speech to text stub
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
