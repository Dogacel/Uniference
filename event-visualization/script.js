document.getElementById('eventSelector').addEventListener('change', handleFileSelect);
document.getElementById('timeSlider').addEventListener('input', () => renderAtTime(Number(timeSlider.value)));

let events = [];
let devices = {};
let chans = {};
let maxTime = 0;
let interval = null;
let speed = 1;
let playing = false;

const chanDisplay = document.getElementById('chanDisplay');
const eventDisplay = document.getElementById('eventDisplay');
const eventSelector = document.getElementById('eventSelector');
const timeSlider = document.getElementById('timeSlider');
const speedControl = document.getElementById('speedControl');

const graph = new graphology.MultiGraph();
const sigmaInstance = new Sigma(graph, document.getElementById("graphDisplay"));

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

// Logarithmic speed mapping: slider 0-100 maps to 0.001x - 1x
function sliderToSpeed(sliderVal) {
    if (sliderVal == 0) {
        return 0;
    }
    // log10(0.001) = -3, log10(1) = 0
    let min = -3, max = 0;
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
        chans = {};
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

            if (ev.chan && !(ev.chan in chans)) {
                chans[ev.chan] = {};
            }
        });
        // Set slider range
        maxTime = Math.max(...events.filter(ev => ev.time !== undefined).map(ev => ev.time), 0);
        timeSlider.min = 0;
        timeSlider.max = Math.ceil(maxTime * 10_000) + 0.1;
        timeSlider.step = 0.001;
        timeSlider.value = 0;

        // Create a graphology graph
        graph.clear();
        x = 0;
        for (const device in devices) {
            graph.addNode(device, {
                label: "Dev/" + device,
                x: x,
                y: 0.5,
                size: 20,
            });
            x += 0.2;
        }

        x = 0;
        for (const chan in chans) {
            graph.addNode(chan, {
                label: "Chan/" + chan,
                x: x,
                y: 0.3,
                size: 15,
            });
            x += 0.2;
        }

        renderAtTime(0);
    };
    reader.readAsText(file);
}

function pointOnEdge(edgeKey, t) {
    const [a, b] = graph.extremities(edgeKey);
    const { x: ax, y: ay } = graph.getNodeAttributes(a);
    const { x: bx, y: by } = graph.getNodeAttributes(b);
    return { x: ax + (bx - ax) * t, y: ay + (by - ay) * t };
}

function ensurePacket(edgeKey) {
    const pid = `packet:${edgeKey}`;
    if (!graph.hasNode(pid)) {
        graph.addNode(pid, { size: 5, color: "#111827", zIndex: 10 });
    }
    return pid;
}

function setPacketPercent(edgeKey, t) {
    const pid = ensurePacket(edgeKey);
    const p = pointOnEdge(edgeKey, Math.max(0, Math.min(1, t)));
    graph.setNodeAttribute(pid, "x", p.x);
    graph.setNodeAttribute(pid, "y", p.y);
}

function removePackage(edgeKey) {
    const pid = `packet:${edgeKey}`;
    if (graph.hasNode(pid)) {
        graph.dropNode(pid);
    }
}

function dropNodesByPrefix(prefix) {
  const toDrop = [];
  graph.forEachNode((key) => {
    if (String(key).startsWith(prefix)) toDrop.push(key);
  });
  for (const k of toDrop) graph.dropNode(k); // also removes attached edges
}

function clearPackages() {
    dropNodesByPrefix("packet:");
    graph.clearEdges();
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

    for (const chan in chans) {
        chans[chan] = {};
    }

    clearPackages();

    const stateColors = {
        'idle': '#1b5e20',
        'running': '#bfa600',
        'terminated': '#bf360c',
        'finished': '#1a237e',
        'pending': '#bfa600',
        'arrived': '#1b5e20',
    }

    // Apply events up to us/10000
    const t = us / 10000;
    events.forEach(ev => {
        if (ev.time !== undefined && ev.time > t) return;
        if (ev.device) {
            if (ev.action === 'running' || ev.action === 'idle' || ev.action === 'terminated' || ev.action === 'finished') {
                devices[ev.device].state = ev.action;
                devices[ev.device].lastTime = ev.time ?? 0;
                graph.setNodeAttribute(ev.device, "color", stateColors[ev.action]);
            }

            if (ev.action === 'generate') {
                devices[ev.device].tokens.push(ev.token);
                devices[ev.device].lastToken = ev.token;
                devices[ev.device].lastTime = ev.time ?? 0;
                devices[ev.device].tokensPerSecond = (devices[ev.device].tokens.length / (t - (devices[ev.device].lastTime / 1000))).toFixed(2);
            }

        }

        if (ev.chan) {
            if (ev.action === 'synchronize') {
                const state = ev.arrive_at >= t ? 'pending' : 'arrived';
                const progress = Math.min(((t - ev.time) / (ev.arrive_at - ev.time)) * 100, 100);
                chans[ev.chan][ev.id] = {
                    device: ev.device,
                    operation: 'synchronize',
                    latency: ev.arrive_at - ev.time,
                    progress: progress,
                    state
                };
                if (!graph.hasEdge(ev.id)) {
                    graph.addDirectedEdgeWithKey(ev.id, ev.device, ev.chan, { size: 5, color: stateColors[state] });
                } else {
                    graph.setEdgeAttribute(ev.id, "color", stateColors[state]);
                }

                const key = graph.edge(ev.device, ev.chan);
                setPacketPercent(key, 1 - (Math.abs(progress - 50) / 50));
            }

            if (ev.action === 'desynchronize') {
                delete chans[ev.chan][ev.id];
                if (graph.hasEdge(ev.id)) {
                    graph.dropEdge(ev.id);
                    removePackage(ev.id);
                }
            }

            if (ev.action == 'send') {
                const state = ev.arrive_at >= t ? 'pending' : 'arrived';
                const progress = Math.min(((t - ev.time) / (ev.arrive_at - ev.time)) * 100, 100);
                chans[ev.chan][ev.id] = {
                    device: ev.device,
                    operation: 'send',
                    latency: ev.arrive_at - ev.time,
                    progress: progress,
                    state,
                };

                if (state == 'pending') {
                    if (!graph.hasEdge(ev.id)) {
                        graph.addDirectedEdgeWithKey(ev.id, ev.device, ev.chan, { size: 5, color: stateColors[state] });
                    } else {
                        graph.setEdgeAttribute(ev.id, "color", stateColors[state]);
                    }

                    setPacketPercent(ev.id, (progress / 100));
                } else {
                    if (graph.hasEdge(ev.id)) {
                        graph.dropEdge(ev.id);
                        removePackage(ev.id)
                    }
                }
            }

            // if (ev.action == 'receive') {
            //     const state = ev.arrive_at >= t ? 'pending' : 'arrived';
            //     chans[ev.chan][ev.device] = {
            //         operation: 'receive',
            //         latency: ev.arrive_at - ev.time,
            //         progress: Math.min(((t - ev.time) / (ev.arrive_at - ev.time)) * 100, 100),
            //         state,
            //     };

            //     if (state == 'pending') {
            //         if (!graph.hasEdge(ev.device, ev.chan)) {
            //             graph.addEdge(ev.device, ev.chan, { size: 5, color: stateColors[state] });
            //         } else {
            //             graph.setEdgeAttribute(ev.device, ev.chan, "color", stateColors[state]);
            //         }

            //         const key = graph.edge(ev.device, ev.chan);
            //         setPacketPercent(key, 1 - (Math.abs(chans[ev.chan][ev.device].progress - 50) / 50));
            //     } else {
            //         if (graph.hasEdge(ev.device, ev.chan)) {
            //             const key = graph.edge(ev.device, ev.chan);
            //             graph.dropEdge(ev.device, ev.chan);
            //             removePackage(key)
            //         }
            //     }
            // }
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

    // Render sidepanel
    let html = `<h3>Channel States</h3>`;
    let hasAny = false;
    for (const chan in chans) {
        if (Object.keys(chans[chan]).length === 0) continue;

        html += `<div class="chan-wrap" id="chan-${chan}"><h4>Channel: ${chan}</h4>`;
        for (const id in chans[chan]) {
            const ch = chans[chan][id]
            if (ch.operation == 'synchronize') {
                hasAny = true;
                html += `<div class="chan-msg chan-${ch.state}">
                <b>Device:</b> ${ch.device}<br>
                <b>Operation:</b> ${ch.operation}<br>
                <b>Latency:</b> ${ch.latency?.toFixed(3)} s <br>
                <b>Progress:</b> ${ch.progress?.toFixed(0)}%
                </div>`;
            }
        }
        html += `</div>`;
    }
    if (!hasAny) html += `<div style="color:#888;">No active channels found.</div>`;
    chanDisplay.innerHTML = html;

    // Update current time display
    currentTimeValue.textContent = (us / 10_000).toFixed(6);
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
    if (elapsed != 0) {
        renderAtTime(next);
    }
    requestAnimationFrame(step);
}
requestAnimationFrame(step);


// Instantiate sigma.js and render the graph
