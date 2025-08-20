document.getElementById('eventSelector').addEventListener('change', handleFileSelect);
document.getElementById('timeSlider').addEventListener('input', () => renderAtTime(Number(timeSlider.value)));

let events = [];
let devices = {};
let maxTime = 0;
let interval = null;
let speed = 1;
let playing = false;

const eventDisplay = document.getElementById('eventDisplay');
const eventSelector = document.getElementById('eventSelector');
const timeSlider = document.getElementById('timeSlider');
const speedControl = document.getElementById('speedControl');

// Add current time display
let currentTimeDiv = document.createElement('div');
currentTimeDiv.className = 'current-time';
currentTimeDiv.innerHTML = 'Current Time: <span id="currentTimeValue">0.000</span> s';
eventDisplay.parentNode.insertBefore(currentTimeDiv, eventDisplay);

// Add play/pause controls in a row
let controlsDiv = document.createElement('div');
controlsDiv.className = 'controls-row';
controlsDiv.innerHTML = `
    <label for="speedControl">Speed:</label>
    <input type="range" id="speedControl" min="0" max="10000" step="1" value="0" style="width:400px;">
    <span id="speedValue">1.00x</span>
`;
eventDisplay.parentNode.insertBefore(controlsDiv, currentTimeDiv.nextSibling);

const playBtn = document.getElementById('playBtn');
const pauseBtn = document.getElementById('pauseBtn');
const speedValue = document.getElementById('speedValue');
const currentTimeValue = document.getElementById('currentTimeValue');

// Remove old speed control (if present)
if (speedControl) speedControl.parentNode.removeChild(speedControl);

// Logarithmic speed mapping: slider 0-100 maps to 0.01x - 10x
function sliderToSpeed(sliderVal) {
    if (sliderVal == 0) {
        return 0;
    }
    // log10(0.01) = -2, log10(10) = 1
    let min = -2, max = 1;
    let logSpeed = min + (max - min) * (sliderVal / 10_000);
    return Math.pow(10, logSpeed);
}

function updateSpeedDisplay() {
    speed = sliderToSpeed(Number(document.getElementById('speedControl').value));
    speedValue.textContent = speed.toFixed(3) + 'x';
}

document.getElementById('speedControl').addEventListener('input', updateSpeedDisplay);
updateSpeedDisplay();

eventSelector.addEventListener('change', handleFileSelect);
timeSlider.addEventListener('input', () => renderAtTime(Number(timeSlider.value)));

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (evt) {
        events = [];
        devices = {};
        maxTime = 0;
        evt.target.result.split('\n').forEach(line => {
            line = line.trim();
            if (!line) return;
            try {
                const obj = JSON.parse(line);
                events.push(obj);
                if (obj.time !== undefined && obj.time > maxTime) maxTime = obj.time;
            } catch { }
        });
        // Find all devices
        events.forEach(ev => {
            if (ev.device && !(ev.device in devices)) {
                devices[ev.device] = {
                    state: 'unknown',
                    tokens: [],
                    lastToken: '',
                    lastTime: 0
                };
            }
        });
        // Set slider range
        maxTime = Math.max(...events.filter(ev => ev.time !== undefined).map(ev => ev.time), 0);
        timeSlider.min = 0;
        timeSlider.max = Math.ceil(maxTime * 1_000) + 0.1;
        timeSlider.step = 0.001;
        timeSlider.value = 0;
        renderAtTime(0);
    };
    reader.readAsText(file);
}

function renderAtTime(us) {
    // Reset device states
    for (const dev in devices) {
        devices[dev].state = 'unknown';
        devices[dev].tokens = [];
        devices[dev].lastToken = '';
        devices[dev].lastTime = 0;
        devices[dev].tokensPerSecond = 0;
        devices[dev].idleTime = 0;
    }
    // Apply events up to us/1000
    const t = us / 1000;
    events.forEach(ev => {
        if (ev.time !== undefined && ev.time > t) return;
        if (!ev.device) return;
        if (ev.action === 'running' || ev.action === 'idle' || ev.action === 'terminated' || ev.action === 'finished') {
            devices[ev.device].state = ev.action;
            devices[ev.device].lastTime = ev.time ?? 0;
        }
        if (ev.action === 'generate') {
            devices[ev.device].tokens.push(ev.token);
            devices[ev.device].lastToken = ev.token;
            devices[ev.device].lastTime = ev.time ?? 0;
            devices[ev.device].tokensPerSecond = (devices[ev.device].tokens.length / (t - (devices[ev.device].lastTime / 1000))).toFixed(2);
        }
    });
    // Render
    eventDisplay.innerHTML = '';
    let deviceList = document.createElement('div');
    deviceList.className = 'device-list';
    for (const dev in devices) {
        const d = devices[dev];
        const card = document.createElement('div');
        card.className = 'device-card';
        card.innerHTML = `
            <h2>${dev}</h2>
            <div class="state ${d.state}">${d.state.toUpperCase()}</div>
            <table class="device-stats-table">
                <tr>
                    <td><strong>Last Token:</strong></td>
                    <td>${d.lastToken}</td>
                </tr>
                <tr>
                    <td><strong>Tokens Per Second:</strong></td>
                    <td>${d.tokensPerSecond} tok/s</td>
                </tr>
            </table>
            <hr style="margin: 10px 0;">
            <div class="tokens-section">
                <strong>Tokens:</strong>
                <div>
                    ${d.tokens.join('')}
                </div>
            </div>
        `;
        deviceList.appendChild(card);
    }
    eventDisplay.appendChild(deviceList);

    // Update current time display
    currentTimeValue.textContent = (us / 1_000).toFixed(6);
}

let lastTimestamp = performance.now();
function step() {
    let now = performance.now();
    let elapsed = (now - lastTimestamp) * speed;
    lastTimestamp = now;
    let next = Number(timeSlider.value) + elapsed;
    if (next > timeSlider.max) {
        next = timeSlider.max;
        playing = false;
    }
    timeSlider.value = next;
    renderAtTime(next);
    requestAnimationFrame(step);
}
requestAnimationFrame(step);

