function buildTimelineData(events) {
  const deviceStates = {};
  const items = [];
  let id = 1;

  // Sort events by time (if present)
  events = events.filter(e => e.device && e.time !== undefined)
                 .sort((a, b) => a.time - b.time);

  for (const event of events) {
    const { device, action, time } = event;
    if (!deviceStates[device]) deviceStates[device] = [];

    // Only track running/idle for timeline
    if (action === 'running' || action === 'idle') {
      deviceStates[device].push({ action, time });
    }
  }

  // Build intervals for each device
  for (const [device, states] of Object.entries(deviceStates)) {
    for (let i = 0; i < states.length - 1; i++) {
      const curr = states[i];
      const next = states[i + 1];
      // Only create interval if state changes
      if (curr.action !== next.action) {
        let color = curr.action === 'running' ? '#4caf50' : (curr.action === 'idle' ? '#ffd600' : undefined);
        items.push({
          id: id++,
          group: device,
          start: new Date(curr.time * 1000), // convert to ms
          end: new Date(next.time * 1000),
          content: curr.action,
          style: color ? `background-color: ${color}; color: #222; border: 1px solid #888;` : undefined
        });
      }
    }
    // Optionally, add last state as a point
    const last = states[states.length - 1];
    if (last) {
      let color = last.action === 'running' ? '#4caf50' : (last.action === 'idle' ? '#ffd600' : undefined);
      items.push({
        id: id++,
        group: device,
        start: new Date(last.time * 1000),
        content: last.action,
        style: color ? `background-color: ${color}; color: #222; border: 1px solid #888;` : undefined
      });
    }
  }

  // Groups for each device
  const groups = Object.keys(deviceStates).map(device => ({
    id: device,
    content: device
  }));

  return { items, groups };
}


function renderTimeline(events) {
  const { items, groups } = buildTimelineData(events);
  const container = document.getElementById('visualization');
  container.innerHTML = '';
  const timeline = new vis.Timeline(container, items, groups, {
    stack: false,
    showCurrentTime: true,
    orientation: 'top',
    margin: { item: 10, axis: 5 }
  });
}

document.getElementById('fileInput').addEventListener('change', function(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const lines = text.trim().split('\n');
    let events = [];
    for (const line of lines) {
      try {
        events.push(JSON.parse(line));
      } catch (err) {
        // skip invalid lines
      }
    }
    renderTimeline(events);
  };
  reader.readAsText(file);
});