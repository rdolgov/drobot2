"use strict";

const token = document.body.dataset.controlToken;
const safetyAck = document.querySelector("#safetyAck");
const legGrid = document.querySelector("#legGrid");
const notice = document.querySelector("#notice");
const connection = document.querySelector("#connection");
const connectionText = document.querySelector("#connectionText");
const alertPanel = document.querySelector("#alertPanel");
const alertList = document.querySelector("#alertList");
const disarmAll = document.querySelector("#disarmAll");
const zeroAll = document.querySelector("#zeroAll");
const centerAll = document.querySelector("#centerAll");
const captureZeroAll = document.querySelector("#captureZeroAll");
const gaitPanel = document.querySelector(".gait-panel");
const walkForward = document.querySelector("#walkForward");
const stopWalk = document.querySelector("#stopWalk");
const gaitStage = document.querySelector("#gaitStage");
const gaitPhase = document.querySelector("#gaitPhase");
const gaitProgress = document.querySelector("#gaitProgress");
const gaitDetail = document.querySelector("#gaitDetail");
const legNodes = new Map();
const motorNodes = new Map();
const sliderTimers = new Map();
let latestState = null;
let connected = false;
let refreshing = false;

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

function formatCurrent(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(2)} A` : `${value.toFixed(0)} mA`;
}

function motorKey(legNumber, motorNumber) {
  return `${legNumber}:${motorNumber}`;
}

function stateMotor(legNumber, motorNumber) {
  return latestState?.legs
    .find((leg) => leg.number === legNumber)
    ?.motors.find((motor) => motor.number === motorNumber);
}

async function postAction(path, body, successMessage) {
  try {
    await api(path, "POST", body);
    showNotice(successMessage);
    await refresh();
  } catch (error) {
    showNotice(error.message, true);
  }
}

function createMotorRow(leg, motor) {
  const row = document.createElement("article");
  row.className = "motor-row";
  row.dataset.leg = leg.number;
  row.dataset.motor = motor.number;
  row.innerHTML = `
    <div class="motor-identity">
      <span class="motor-id"></span>
      <div>
        <h3></h3>
        <p class="direction-note"></p>
      </div>
      <span class="arm-state">DISARMED</span>
    </div>
    <div class="position-block">
      <span class="micro-label">Current</span>
      <strong class="measured">0.00°</strong>
      <span class="target-line">Target <b class="desired">—</b></span>
    </div>
    <div class="motion-controls">
      <input class="slider" type="range" step="0.5" aria-label="Target angle">
      <div class="range-labels"><span class="minimum"></span><span class="maximum"></span></div>
      <div class="entry-row">
        <input class="angle-input" type="number" step="0.5" inputmode="decimal" aria-label="Exact target angle">
        <button class="set-button" type="button">SET</button>
      </div>
      <div class="quick-row" aria-label="Quick test angles">
        <button data-delta="-15" type="button">−15°</button>
        <button data-delta="-5" type="button">−5°</button>
        <button data-home type="button">ZERO</button>
        <button data-delta="5" type="button">+5°</button>
        <button data-delta="15" type="button">+15°</button>
      </div>
    </div>
    <div class="motor-actions">
      <button class="arm-button" type="button">ARM</button>
      <div class="telemetry">
        <span><i>V</i><b class="voltage">—</b></span>
        <span><i>°C</i><b class="temperature">—</b></span>
        <span><i>mA</i><b class="current">—</b></span>
        <span><i>RAW</i><b class="raw">—</b></span>
      </div>
    </div>
  `;

  row.querySelector(".motor-id").textContent = `ID ${motor.id}`;
  row.querySelector("h3").textContent = motor.label;
  row.querySelector(".direction-note").textContent =
    `Positive moves ${motor.positive_motion} · direction ${motor.direction > 0 ? "+1" : "−1"}`;
  const slider = row.querySelector(".slider");
  const angleInput = row.querySelector(".angle-input");
  slider.min = motor.min_deg;
  slider.max = motor.max_deg;
  angleInput.min = motor.min_deg;
  angleInput.max = motor.max_deg;
  row.querySelector(".minimum").textContent = `${motor.min_deg}°`;
  row.querySelector(".maximum").textContent = `+${motor.max_deg}°`;

  const sendTarget = async (degrees) => {
    const value = Number(degrees);
    await postAction(
      "/api/target",
      { leg: leg.number, motor: motor.number, degrees: value },
      `${leg.label} · ${motor.label} → ${value.toFixed(1)}°`,
    );
  };

  slider.addEventListener("input", () => {
    row.querySelector(".desired").textContent = formatAngle(slider.value);
    angleInput.value = Number(slider.value).toFixed(1);
    const key = motorKey(leg.number, motor.number);
    clearTimeout(sliderTimers.get(key));
    sliderTimers.set(key, setTimeout(() => sendTarget(slider.value), 120));
  });

  row.querySelector(".set-button").addEventListener("click", () => {
    sendTarget(angleInput.value);
  });
  angleInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      sendTarget(angleInput.value);
    }
  });

  row.querySelectorAll("[data-delta]").forEach((button) => {
    button.addEventListener("click", () => {
      const current = stateMotor(leg.number, motor.number);
      const basis = current?.desired_deg ?? current?.measured_deg ?? 0;
      const value = Math.max(
        motor.min_deg,
        Math.min(motor.max_deg, Number(basis) + Number(button.dataset.delta)),
      );
      sendTarget(value);
    });
  });
  row.querySelector("[data-home]").addEventListener("click", () => sendTarget(0));

  row.querySelector(".arm-button").addEventListener("click", async () => {
    const current = stateMotor(leg.number, motor.number);
    if (current?.armed) {
      await postAction(
        "/api/disarm",
        { leg: leg.number, motor: motor.number },
        `${leg.label} · ${motor.label} disarmed`,
      );
    } else {
      await postAction(
        "/api/arm",
        {
          leg: leg.number,
          motor: motor.number,
          safety_ack: safetyAck.checked,
        },
        `${leg.label} · ${motor.label} armed at measured position`,
      );
    }
  });

  motorNodes.set(motorKey(leg.number, motor.number), row);
  return row;
}

function createLegPanel(leg) {
  const section = document.createElement("section");
  section.className = "leg-panel";
  section.dataset.leg = leg.number;
  section.innerHTML = `
    <header class="leg-header">
      <div class="leg-title">
        <span class="leg-number"></span>
        <div><p>THREE-MOTOR BRANCH</p><h2></h2></div>
      </div>
      <div class="leg-summary">
        <span><b class="leg-current">—</b> diagnostic</span>
        <span><b class="leg-armed">0</b> armed</span>
      </div>
      <div class="leg-actions">
        <button class="arm-leg" type="button">ARM 3 MOTORS</button>
        <button class="zero-leg" type="button">ZERO ARMED</button>
        <button class="disarm-leg" type="button">DISARM LEG</button>
      </div>
    </header>
    <div class="motor-list"></div>
  `;
  section.querySelector(".leg-number").textContent = leg.number;
  section.querySelector("h2").textContent = leg.label;
  const list = section.querySelector(".motor-list");
  leg.motors.forEach((motor) => list.appendChild(createMotorRow(leg, motor)));

  section.querySelector(".arm-leg").addEventListener("click", () => {
    postAction(
      "/api/arm-leg",
      { leg: leg.number, safety_ack: safetyAck.checked },
      `${leg.label} armed at measured positions`,
    );
  });
  section.querySelector(".zero-leg").addEventListener("click", () => {
    postAction(
      "/api/zero-leg",
      { leg: leg.number },
      `${leg.label} armed motors returning to zero`,
    );
  });
  section.querySelector(".disarm-leg").addEventListener("click", () => {
    postAction(
      "/api/disarm-leg",
      { leg: leg.number },
      `${leg.label} disarmed`,
    );
  });

  legGrid.appendChild(section);
  legNodes.set(leg.number, section);
  return section;
}

function updateMotor(leg, motor) {
  const key = motorKey(leg.number, motor.number);
  const row = motorNodes.get(key) || createMotorRow(leg, motor);
  row.classList.toggle("armed", motor.armed);
  row.classList.toggle("torque-mismatch", motor.torque_enabled !== motor.armed);
  row.querySelector(".arm-state").textContent = motor.armed ? "ARMED" : "DISARMED";
  row.querySelector(".arm-button").textContent = motor.armed ? "DISARM" : "ARM";
  row.querySelector(".measured").textContent = formatAngle(motor.measured_deg);
  row.querySelector(".desired").textContent = formatAngle(motor.desired_deg);
  row.querySelector(".voltage").textContent = motor.voltage_v.toFixed(1);
  row.querySelector(".temperature").textContent = motor.temperature_c;
  row.querySelector(".current").textContent = motor.current_ma.toFixed(0);
  row.querySelector(".raw").textContent = motor.raw_position;

  row.querySelectorAll(
    ".slider, .angle-input, .set-button, .quick-row button",
  ).forEach((control) => {
    control.disabled = !motor.armed || latestState?.crawl?.active;
  });
  row.querySelector(".arm-button").disabled = latestState?.crawl?.active;

  const value = motor.desired_deg ?? motor.measured_deg;
  const slider = row.querySelector(".slider");
  const angleInput = row.querySelector(".angle-input");
  if (document.activeElement !== slider) {
    slider.value = value;
  }
  if (document.activeElement !== angleInput) {
    angleInput.value = Number(value).toFixed(1);
  }
}

function updateLeg(leg) {
  const panel = legNodes.get(leg.number) || createLegPanel(leg);
  panel.classList.toggle("active", leg.armed_count > 0);
  panel.querySelector(".leg-current").textContent = formatCurrent(leg.current_ma);
  panel.querySelector(".leg-armed").textContent = leg.armed_count;
  panel.querySelector(".arm-leg").disabled = latestState?.crawl?.active;
  panel.querySelector(".zero-leg").disabled = latestState?.crawl?.active;
  leg.motors.forEach((motor) => updateMotor(leg, motor));
}

function updateSummary(state) {
  const { summary, settings, crawl } = state;
  document.querySelector("#onlineCount").textContent = `${summary.online_count} / 12`;
  document.querySelector("#healthLabel").textContent =
    summary.health === "nominal" ? "All telemetry nominal" : "Attention required";
  document.querySelector("#voltageRange").textContent =
    `${summary.voltage_min_v.toFixed(1)}–${summary.voltage_max_v.toFixed(1)} V`;
  document.querySelector("#voltageSpread").textContent =
    `${summary.voltage_spread_v.toFixed(1)} V spread across bus`;
  document.querySelector("#totalCurrent").textContent =
    formatCurrent(summary.total_current_ma);
  document.querySelector("#maxTemperature").textContent =
    `${summary.max_temperature_c} °C`;
  document.querySelector("#temperatureLimit").textContent =
    `Warning at ${settings.temperature_warning_c} °C`;
  document.querySelector("#armedCount").textContent = `${summary.armed_count} / 12`;
  document.querySelector("#watchdog").textContent =
    `${settings.heartbeat_timeout_s.toFixed(1)} s auto-disarm`;
  document.querySelector("#torqueLimit").textContent =
    `${(settings.torque_limit / 10).toFixed(0)}%`;
  document.querySelector("#servoSpeed").textContent = settings.speed;
  document.querySelector("#acceleration").textContent = settings.acceleration;
  document.querySelector("#rampRate").textContent =
    `${settings.ramp_rate_deg_s.toFixed(0)}°/s`;
  document.querySelector("#lastEvent").textContent = state.last_event;

  alertPanel.hidden = summary.warnings.length === 0 && !state.fault;
  alertList.innerHTML = "";
  [...summary.warnings, ...(state.fault ? [state.fault] : [])].forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    alertList.appendChild(item);
  });

  document.body.classList.toggle("has-warning", summary.health === "warning");
  document.body.classList.toggle("has-armed", state.any_armed);

  const phaseText = crawl.phase.replaceAll("_", " ").toUpperCase();
  const swingText = crawl.swing_corner
    ? ` / ${crawl.swing_corner.replaceAll("_", " ").toUpperCase()}`
    : "";
  gaitPanel.classList.toggle("active", crawl.active);
  gaitStage.textContent = crawl.active
    ? `${crawl.stage.toUpperCase()} / ${(crawl.progress * 100).toFixed(0)}%`
    : crawl.stage === "complete"
      ? "COMPLETE / HOLDING"
      : "READY / DISARMED";
  gaitPhase.textContent = `${phaseText}${swingText}`;
  gaitProgress.style.width = `${Math.max(0, Math.min(100, crawl.progress * 100))}%`;
  gaitDetail.textContent =
    `${crawl.stride_mm.toFixed(0)} mm stride / ` +
    `${crawl.lift_mm.toFixed(0)} mm lift / ${crawl.duration_s.toFixed(0)} s`;

  walkForward.disabled =
    crawl.active ||
    state.any_armed ||
    summary.health !== "nominal" ||
    !safetyAck.checked;
  stopWalk.disabled = !state.any_armed;
  zeroAll.disabled = crawl.active;
  centerAll.disabled = crawl.active;
  captureZeroAll.disabled = crawl.active || state.any_armed;
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    latestState = await api("/api/state");
    latestState.legs.forEach(updateLeg);
    updateSummary(latestState);
    connected = true;
    connection.className = latestState.crawl.active
      ? "connection armed"
      : latestState.any_armed
      ? "connection armed"
      : "connection online";
    connectionText.textContent = latestState.crawl.active
      ? "LIVE / CRAWL ACTIVE"
      : latestState.any_armed
      ? "LIVE · TORQUE ARMED"
      : "LIVE · ALL DISARMED";
  } catch (error) {
    connected = false;
    connection.className = "connection offline";
    connectionText.textContent = "CONNECTION LOST";
    showNotice(error.message, true);
  } finally {
    refreshing = false;
  }
}

disarmAll.addEventListener("click", () => {
  postAction("/api/disarm-all", {}, "All 12 motors disarmed");
});

zeroAll.addEventListener("click", () => {
  postAction("/api/zero-all", {}, "All armed motors returning to zero");
});

centerAll.addEventListener("click", () => {
  if (!safetyAck.checked) {
    showNotice("Confirm support, clearance, and cutoff before centering", true);
    return;
  }
  const accepted = window.confirm(
    "CENTER ALL 12 will arm every motor and ramp every joint to calibrated 0°. " +
      "Keep the robot supported and the power cutoff ready. Continue?",
  );
  if (!accepted) {
    return;
  }
  postAction(
    "/api/center-all",
    { safety_ack: true, confirmation: "CENTER ALL 12" },
    "All 12 motors returning to calibrated zero; torque remains armed",
  );
});

captureZeroAll.addEventListener("click", () => {
  if (!safetyAck.checked) {
    showNotice("Confirm support, clearance, and cutoff before capture", true);
    return;
  }
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before capturing zero", true);
    return;
  }
  const accepted = window.confirm(
    "CAPTURE ZERO ALL will replace the four software calibration files with " +
      "the current manually positioned pose. Backups will be saved first. Continue?",
  );
  if (!accepted) {
    return;
  }
  postAction(
    "/api/capture-zero-all",
    { safety_ack: true, confirmation: "CAPTURE ZERO ALL" },
    "Current pose saved as calibrated zero for all 12 motors",
  );
});

walkForward.addEventListener("click", () => {
  if (!safetyAck.checked) {
    showNotice(
      "Confirm support, clearance, corner map, and cutoff before walking",
      true,
    );
    return;
  }
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before starting the crawl", true);
    return;
  }
  const accepted = window.confirm(
    "WALK FORWARD will arm all 12 motors and run two slow crawl cycles " +
      "using the displayed corner map. Start on blocks for the first test, " +
      "keep the physical cutoff ready, and stop on slip or unexpected motion. Continue?",
  );
  if (!accepted) {
    return;
  }
  postAction(
    "/api/crawl-forward",
    { safety_ack: true, confirmation: "WALK FORWARD" },
    "Moving to crawl stance; rear-right foot will move first",
  );
});

stopWalk.addEventListener("click", () => {
  postAction("/api/crawl-stop", {}, "Crawl stopped; all 12 motors disarmed");
});

safetyAck.addEventListener("change", () => {
  if (latestState) updateSummary(latestState);
});

setInterval(() => {
  if (connected && latestState?.any_armed) {
    api("/api/heartbeat", "POST", {}).catch(() => {});
  }
}, 700);

setInterval(refresh, 900);

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
