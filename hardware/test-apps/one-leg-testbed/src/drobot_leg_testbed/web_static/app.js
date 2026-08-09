"use strict";

const token = document.body.dataset.controlToken;
const grid = document.querySelector("#motorGrid");
const notice = document.querySelector("#notice");
const safetyAck = document.querySelector("#safetyAck");
const connection = document.querySelector("#connection");
const connectionText = document.querySelector("#connectionText");
const disarmAll = document.querySelector("#disarmAll");
const cards = new Map();
let latestState = null;
let connected = false;

async function api(path, method = "GET", body = undefined, keepalive = false) {
  const response = await fetch(path, {
    method,
    headers: {
      "X-Control-Token": token,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
    keepalive,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function showNotice(message, isError = false) {
  notice.textContent = message;
  notice.classList.toggle("error", isError);
}

function formatAngle(value) {
  return value == null ? "—" : `${Number(value).toFixed(2)}°`;
}

function motorCard(motor) {
  const article = document.createElement("article");
  article.className = "motor-card";
  article.dataset.motor = motor.number;
  article.innerHTML = `
    <div class="card-title">
      <span class="motor-number">${motor.number}</span>
      <h3></h3>
      <span class="arm-state">DISARMED</span>
    </div>
    <div class="angle-block">
      <span class="input-label">Current position</span>
      <div class="angle-line">
        <div class="angle-value"><span class="measured">0.00</span><small>deg</small></div>
        <div class="target-readout">
          <span class="input-label">Destination</span>
          <strong class="desired">—</strong>
        </div>
      </div>
      <input class="slider" type="range" step="0.5" aria-label="Target angle slider">
      <div class="range-labels"><span class="minimum"></span><span class="maximum"></span></div>
    </div>
    <label class="input-label">Exact destination angle</label>
    <div class="target-entry">
      <input class="angle-input" type="number" step="0.5" inputmode="decimal">
      <button class="set-button" type="button">SET</button>
    </div>
    <div class="quick-row" aria-label="Quick angle controls">
      <button class="nudge-button" data-delta="-5" type="button">−5°</button>
      <button class="nudge-button" data-delta="-1" type="button">−1°</button>
      <button class="nudge-button home" data-home type="button">ZERO</button>
      <button class="nudge-button" data-delta="1" type="button">+1°</button>
      <button class="nudge-button" data-delta="5" type="button">+5°</button>
    </div>
    <button class="arm-button" type="button">ARM MOTOR</button>
    <div class="telemetry">
      <div><span class="telemetry-label">Voltage</span><strong class="voltage">—</strong></div>
      <div><span class="telemetry-label">Temp</span><strong class="temperature">—</strong></div>
      <div><span class="telemetry-label">Current</span><strong class="current">—</strong></div>
      <div><span class="telemetry-label">Raw</span><strong class="raw">—</strong></div>
    </div>
  `;

  article.querySelector("h3").textContent = motor.label;
  const slider = article.querySelector(".slider");
  const angleInput = article.querySelector(".angle-input");
  const setButton = article.querySelector(".set-button");
  const armButton = article.querySelector(".arm-button");
  slider.min = motor.min_deg;
  slider.max = motor.max_deg;
  angleInput.min = motor.min_deg;
  angleInput.max = motor.max_deg;
  article.querySelector(".minimum").textContent = `${motor.min_deg}°`;
  article.querySelector(".maximum").textContent = `+${motor.max_deg}°`;

  const sendTarget = async (degrees) => {
    try {
      await api("/api/target", "POST", {
        motor: motor.number,
        degrees: Number(degrees),
      });
      showNotice(`Motor #${motor.number} destination: ${Number(degrees).toFixed(1)}°`);
      await refresh();
    } catch (error) {
      showNotice(error.message, true);
    }
  };

  let sliderTimer = null;
  slider.addEventListener("input", () => {
    article.querySelector(".desired").textContent = formatAngle(slider.value);
    angleInput.value = Number(slider.value).toFixed(1);
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(() => sendTarget(slider.value), 90);
  });

  setButton.addEventListener("click", () => sendTarget(angleInput.value));
  angleInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      sendTarget(angleInput.value);
    }
  });

  article.querySelectorAll("[data-delta]").forEach((button) => {
    button.addEventListener("click", () => {
      const current =
        latestState?.motors.find((item) => item.number === motor.number)?.desired_deg ??
        latestState?.motors.find((item) => item.number === motor.number)?.measured_deg ??
        0;
      const value = Math.max(
        motor.min_deg,
        Math.min(motor.max_deg, Number(current) + Number(button.dataset.delta)),
      );
      sendTarget(value);
    });
  });

  article.querySelector("[data-home]").addEventListener("click", () => sendTarget(0));

  armButton.addEventListener("click", async () => {
    const current = latestState?.motors.find((item) => item.number === motor.number);
    try {
      if (current?.armed) {
        await api("/api/disarm", "POST", { motor: motor.number });
        showNotice(`Motor #${motor.number} disarmed`);
      } else {
        await api("/api/arm", "POST", {
          motor: motor.number,
          safety_ack: safetyAck.checked,
        });
        showNotice(`Motor #${motor.number} armed at its current position`);
      }
      await refresh();
    } catch (error) {
      showNotice(error.message, true);
    }
  });

  grid.appendChild(article);
  cards.set(motor.number, article);
  return article;
}

function updateCard(motor) {
  const article = cards.get(motor.number) || motorCard(motor);
  article.classList.toggle("armed", motor.armed);
  article.querySelector(".arm-state").textContent = motor.armed ? "ARMED" : "DISARMED";
  article.querySelector(".arm-button").textContent = motor.armed
    ? "DISARM MOTOR"
    : "ARM MOTOR";
  article.querySelector(".measured").textContent = motor.measured_deg.toFixed(2);
  article.querySelector(".desired").textContent = formatAngle(motor.desired_deg);
  article.querySelector(".voltage").textContent = `${motor.voltage_v.toFixed(1)} V`;
  article.querySelector(".temperature").textContent = `${motor.temperature_c} °C`;
  article.querySelector(".current").textContent = `${motor.current_ma.toFixed(1)} mA`;
  article.querySelector(".raw").textContent = motor.raw_position;

  const controls = article.querySelectorAll(
    ".slider, .angle-input, .set-button, .nudge-button",
  );
  controls.forEach((control) => {
    control.disabled = !motor.armed;
  });

  const slider = article.querySelector(".slider");
  const angleInput = article.querySelector(".angle-input");
  if (document.activeElement !== slider) {
    slider.value = motor.desired_deg ?? motor.measured_deg;
  }
  if (document.activeElement !== angleInput) {
    angleInput.value = Number(motor.desired_deg ?? motor.measured_deg).toFixed(1);
  }
}

async function refresh() {
  try {
    latestState = await api("/api/state");
    latestState.motors.forEach(updateCard);
    document.querySelector("#rampRate").textContent =
      `${latestState.ramp_rate_deg_s.toFixed(0)}°/s`;
    document.querySelector("#watchdog").textContent =
      `${latestState.heartbeat_timeout_s.toFixed(1)} s auto-disarm`;
    document.querySelector("#lastEvent").textContent = latestState.last_event;
    if (latestState.fault) {
      showNotice(latestState.fault, true);
    }
    connected = true;
    connection.className = "connection online";
    connectionText.textContent = latestState.any_armed ? "LIVE · MOTOR ARMED" : "LIVE · SAFE";
  } catch (error) {
    connected = false;
    connection.className = "connection offline";
    connectionText.textContent = "CONNECTION LOST";
    showNotice(error.message, true);
  }
}

disarmAll.addEventListener("click", async () => {
  try {
    await api("/api/disarm-all", "POST", {});
    showNotice("All motors disarmed");
    await refresh();
  } catch (error) {
    showNotice(error.message, true);
  }
});

setInterval(() => {
  if (connected && latestState?.any_armed) {
    api("/api/heartbeat", "POST", {}).catch(() => {});
  }
}, 750);

setInterval(refresh, 400);

window.addEventListener("pagehide", () => {
  fetch("/api/disarm-all", {
    method: "POST",
    headers: {
      "X-Control-Token": token,
      "Content-Type": "application/json",
    },
    body: "{}",
    keepalive: true,
  }).catch(() => {});
});

refresh();
