"use strict";

const token = document.body.dataset.controlToken;
const legGrid = document.querySelector("#legGrid");
const notice = document.querySelector("#notice");
const connection = document.querySelector("#connection");
const connectionText = document.querySelector("#connectionText");
const modeBadge = document.querySelector("#modeBadge");
const alertPanel = document.querySelector("#alertPanel");
const alertList = document.querySelector("#alertList");
const errorLog = document.querySelector("#errorLog");
const errorList = document.querySelector("#errorList");
const disarmAll = document.querySelector("#disarmAll");
const settingsButton = document.querySelector("#settingsButton");
const settingsDialog = document.querySelector("#settingsDialog");
const settingsClose = document.querySelector("#settingsClose");
const clearErrorLog = document.querySelector("#clearErrorLog");
const zeroAll = document.querySelector("#zeroAll");
const centerAll = document.querySelector("#centerAll");
const captureZeroAll = document.querySelector("#captureZeroAll");
const gaitPanel = document.querySelector(".gait-panel");
const setCrawlStance = document.querySelector("#setCrawlStance");
const walkForward = document.querySelector("#walkForward");
const walkDiagonalPair = document.querySelector("#walkDiagonalPair");
const stopWalk = document.querySelector("#stopWalk");
const gaitStage = document.querySelector("#gaitStage");
const gaitPhase = document.querySelector("#gaitPhase");
const gaitProgress = document.querySelector("#gaitProgress");
const gaitDetail = document.querySelector("#gaitDetail");
const rlPanel = document.querySelector("#rlPanel");
const rlSpeed = document.querySelector("#rlSpeed");
const rlDuration = document.querySelector("#rlDuration");
const rlSafetyAck = document.querySelector("#rlSafetyAck");
const startRl = document.querySelector("#startRl");
const stopRl = document.querySelector("#stopRl");
const rlStatus = document.querySelector("#rlStatus");
const rlModel = document.querySelector("#rlModel");
const rlGravity = document.querySelector("#rlGravity");
const rlTargets = document.querySelector("#rlTargets");
const rlDetail = document.querySelector("#rlDetail");
const resetPower = document.querySelector("#resetPower");
const powerLine = document.querySelector("#powerLine");
const legNodes = new Map();
const motorNodes = new Map();
const sliderTimers = new Map();
let latestState = null;
let connected = false;
let refreshing = false;
let heartbeatInFlight = false;
let reloadingAfterServerRestart = false;
const legacyErrorStorageKey = "drobot-four-leg-permanent-errors-v1";
const errorStorageKey = "drobot-four-leg-recent-errors-v2";
const errorRetentionMs = 10 * 60 * 1000;
let recentErrors = loadRecentErrors();

async function api(
  path,
  method = "GET",
  body = undefined,
  keepalive = false,
  signal = undefined,
) {
  const response = await fetch(path, {
    method,
    headers: {
      "X-Control-Token": token,
      "X-Drobot-Client-Version": "2",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
    keepalive,
    signal,
  });
  const payload = await response.json();
  if (response.status === 403 && !reloadingAfterServerRestart) {
    reloadingAfterServerRestart = true;
    window.location.reload();
    return new Promise(() => {});
  }
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function normalizeErrorMessage(message) {
  const original = String(message || "Unknown dashboard error").trim();
  const transportFault =
    original.includes("Serial bus is not open") ||
    original.includes("No likely USB serial adapter found") ||
    original.includes("Servo bus unavailable") ||
    original.includes("automatic reconnect failed");
  return transportFault
    ? "Servo USB adapter disconnected; waiting for automatic recovery. " +
      "Controls remain unavailable until all 12 motors reconnect."
    : original;
}

function isTransientNetworkError(message) {
  const text = String(message || "").trim().toLowerCase();
  return (
    text === "load failed" ||
    text === "failed to fetch" ||
    text.includes("networkerror when attempting to fetch resource") ||
    text.includes("the network connection was lost")
  );
}

function isStateBackedError(message) {
  const text = normalizeErrorMessage(message).toLowerCase();
  return (
    text.includes("servo usb adapter") ||
    text.includes("serial bus") ||
    text.includes("reconnect") ||
    text.includes("bno") ||
    text.includes("imu") ||
    text.includes("enable feature") ||
    text.includes("rl control") ||
    text.includes("rl policy") ||
    text.includes("joint feedback")
  );
}

function loadRecentErrors() {
  try {
    localStorage.removeItem(legacyErrorStorageKey);
    const stored = JSON.parse(localStorage.getItem(errorStorageKey) || "[]");
    if (!Array.isArray(stored)) return [];
    const messages = new Set();
    const oldestAllowed = Date.now() - errorRetentionMs;
    return stored
      .filter((entry) => entry && typeof entry.message === "string")
      .filter((entry) => !isTransientNetworkError(entry.message))
      .filter((entry) => Date.parse(entry.createdAt) >= oldestAllowed)
      .map((entry) => ({
        ...entry,
        message: normalizeErrorMessage(entry.message),
      }))
      .filter((entry) => {
        if (messages.has(entry.message)) return false;
        messages.add(entry.message);
        return true;
      });
  } catch (_error) {
    return [];
  }
}

function storeRecentErrors() {
  try {
    localStorage.setItem(errorStorageKey, JSON.stringify(recentErrors));
  } catch (_error) {
    // The on-screen recent log still remains available for this page session.
  }
}

function renderRecentErrors() {
  errorList.innerHTML = "";
  recentErrors.forEach((entry) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.dateTime = entry.createdAt;
    time.textContent = new Date(entry.createdAt).toLocaleTimeString();
    const message = document.createElement("span");
    message.textContent = entry.message;
    item.append(time, message);
    errorList.append(item);
  });
  errorLog.hidden = recentErrors.length === 0;
}

function addRecentError(message, createdAt = new Date().toISOString()) {
  if (isTransientNetworkError(message)) return;
  const text = normalizeErrorMessage(message);
  if (recentErrors.some((entry) => entry.message === text)) return;
  recentErrors.push({ message: text, createdAt });
  storeRecentErrors();
  renderRecentErrors();
}

function refreshRecentErrors(state) {
  const previous = JSON.stringify(recentErrors);
  const activeMessages = [state?.fault, state?.rl_policy?.error]
    .filter((message) => typeof message === "string" && message.trim())
    .map(normalizeErrorMessage);
  const activeSet = new Set(activeMessages);
  const oldestAllowed = Date.now() - errorRetentionMs;
  recentErrors = recentErrors.filter((entry) => {
    const createdAt = Date.parse(entry.createdAt);
    if (isStateBackedError(entry.message)) {
      return activeSet.has(normalizeErrorMessage(entry.message));
    }
    return Number.isFinite(createdAt) && createdAt >= oldestAllowed;
  });
  activeMessages.forEach((message) => {
    if (!recentErrors.some((entry) => entry.message === message)) {
      recentErrors.push({ message, createdAt: new Date().toISOString() });
    }
  });
  if (JSON.stringify(recentErrors) !== previous) {
    storeRecentErrors();
    renderRecentErrors();
  }
}

function showNotice(message, isError = false) {
  if (isError) {
    notice.textContent = "";
    addRecentError(message);
    return;
  }
  notice.textContent = message;
}

function formatAngle(value) {
  return value == null ? "—" : `${Number(value).toFixed(2)}°`;
}

function formatCurrent(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(2)} A` : `${value.toFixed(0)} mA`;
}

function formatPower(value) {
  return `${Number(value).toFixed(2)} W`;
}

function renderPowerHistory(power) {
  const history = Array.isArray(power?.history) ? power.history : [];
  if (history.length < 2) {
    powerLine.setAttribute("points", "");
    document.querySelector("#powerScale").textContent = "Waiting for samples";
    return;
  }
  const width = 600;
  const height = 98;
  const peak = Math.max(1, ...history.map((sample) => Number(sample.power_w)));
  const windowSeconds = Number(power.window_s) || 60;
  const points = history.map((sample) => {
    const ageFraction = Math.min(windowSeconds, Number(sample.age_s)) / windowSeconds;
    const x = width - ageFraction * width;
    const y = height - Number(sample.power_w) / peak * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  powerLine.setAttribute("points", points.join(" "));
  document.querySelector("#powerScale").textContent = `0–${peak.toFixed(1)} W`;
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
    if (isTransientNetworkError(error.message)) {
      showNotice("Dashboard connection unavailable; retrying automatically.");
    } else {
      showNotice(error.message, true);
    }
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
        <span><i>W</i><b class="power">—</b></span>
        <span><i>ERR</i><b class="tracking-error">—</b></span>
        <span><i>SPD</i><b class="speed">—</b></span>
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
          safety_ack: true,
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
      { leg: leg.number, safety_ack: true },
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
  row.classList.toggle("possible-stall", motor.possible_stall);
  row.querySelector(".arm-state").textContent = motor.armed ? "ARMED" : "DISARMED";
  row.querySelector(".arm-button").textContent = motor.armed ? "DISARM" : "ARM";
  row.querySelector(".measured").textContent = formatAngle(motor.measured_deg);
  row.querySelector(".desired").textContent = formatAngle(motor.desired_deg);
  row.querySelector(".voltage").textContent = motor.voltage_v.toFixed(1);
  row.querySelector(".temperature").textContent = motor.temperature_c;
  row.querySelector(".current").textContent = motor.current_ma.toFixed(0);
  row.querySelector(".power").textContent = motor.power_w.toFixed(2);
  row.querySelector(".tracking-error").textContent =
    `${motor.tracking_error_deg.toFixed(1)}°`;
  row.querySelector(".speed").textContent = Math.abs(motor.speed);
  row.querySelector(".raw").textContent = motor.raw_position;

  row.querySelectorAll(
    ".slider, .angle-input, .set-button, .quick-row button",
  ).forEach((control) => {
    control.disabled =
      !motor.armed ||
      latestState?.crawl?.active ||
      latestState?.rl_policy?.active;
  });
  row.querySelector(".arm-button").disabled =
    latestState?.crawl?.active || latestState?.rl_policy?.active;

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
  panel.querySelector(".leg-current").textContent =
    `${formatCurrent(leg.current_ma)} / ${formatPower(leg.power_w)}`;
  panel.querySelector(".leg-armed").textContent = leg.armed_count;
  const autonomousMotion =
    latestState?.crawl?.active || latestState?.rl_policy?.active;
  panel.querySelector(".arm-leg").disabled = autonomousMotion;
  panel.querySelector(".zero-leg").disabled = autonomousMotion;
  leg.motors.forEach((motor) => updateMotor(leg, motor));
}

function updateSummary(state) {
  const { summary, settings, crawl, runtime, power } = state;
  const rl = state.rl_policy || {};
  const demoMode = runtime?.mode === "demo";
  modeBadge.textContent = demoMode
    ? "DEMO / NO MOTOR OUTPUT"
    : `HARDWARE / ${(runtime?.port || "SERIAL").toUpperCase()}`;
  modeBadge.className = `mode-badge ${demoMode ? "demo" : "hardware"}`;
  document.body.classList.toggle("demo-mode", demoMode);
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
  const browserHeartbeat = state.browser_heartbeat;
  document.querySelector("#watchdog").textContent =
    browserHeartbeat.recent
      ? `Browser ${browserHeartbeat.age_s.toFixed(1)} s ago · warning only`
      : `STALE ${browserHeartbeat.age_s.toFixed(1)} s · motion continues`;
  document.querySelector("#torqueLimit").textContent =
    `${(settings.torque_limit / 10).toFixed(0)}%`;
  document.querySelector("#servoSpeed").textContent = settings.speed;
  document.querySelector("#acceleration").textContent = settings.acceleration;
  document.querySelector("#rampRate").textContent =
    `${settings.ramp_rate_deg_s.toFixed(0)}°/s`;
  document.querySelector("#lastEvent").textContent = state.last_event;
  document.querySelector("#powerNow").textContent = formatPower(power.instantaneous_w);
  document.querySelector("#powerAverage").textContent =
    `60 s average ${formatPower(power.average_w_60s)}`;
  document.querySelector("#powerPeak").textContent = formatPower(power.peak_w_60s);
  document.querySelector("#currentPeak").textContent =
    `Current peak ${power.peak_current_a_60s.toFixed(2)} A`;
  document.querySelector("#voltageSag").textContent = power.voltage_sag_v == null
    ? "--"
    : `${power.voltage_sag_v.toFixed(2)} V`;
  document.querySelector("#idleVoltage").textContent =
    power.idle_reference_voltage_v == null
      ? "Collecting idle reference"
      : `Idle ${power.idle_reference_voltage_v.toFixed(2)} V / ` +
        `60 s low ${power.minimum_voltage_v_60s.toFixed(2)} V`;
  const batteryCharge = power.battery_charge;
  const batteryChargeNode = document.querySelector("#batteryCharge");
  document.querySelector("#batteryChargeStatus").textContent =
    batteryCharge.status.toUpperCase();
  document.querySelector("#batteryLiveVoltage").textContent =
    power.bus_voltage_v == null
      ? " · -- V"
      : ` · ${power.bus_voltage_v.toFixed(2)} V`;
  batteryChargeNode.dataset.level = batteryCharge.status;
  document.querySelector("#batteryVoltage").textContent =
    batteryCharge.average_cell_voltage_v == null
      ? `${batteryCharge.series_cells}S / collect idle reference`
      : `${batteryCharge.idle_pack_voltage_v.toFixed(2)} V pack / ` +
        `${batteryCharge.average_cell_voltage_v.toFixed(2)} V average cell`;
  document.querySelector("#energyUsed").textContent =
    `${power.energy_wh.toFixed(4)} Wh`;
  document.querySelector("#stallWatch").textContent = power.possible_stall_ids.length
    ? `ID ${power.possible_stall_ids.join(", ")}`
    : "NONE";
  document.querySelector("#powerAssessment").textContent = power.assessment;
  resetPower.disabled = state.any_armed;
  renderPowerHistory(power);

  alertPanel.hidden = summary.warnings.length === 0 && !state.fault;
  alertList.innerHTML = "";
  [...summary.warnings, ...(state.fault ? [state.fault] : [])].forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    alertList.appendChild(item);
  });
  document.body.classList.toggle("has-warning", summary.health === "warning");
  document.body.classList.toggle("has-armed", state.any_armed);

  rlPanel.classList.toggle("active", Boolean(rl.active));
  rlStatus.textContent = String(rl.status || "unavailable").toUpperCase();
  rlModel.textContent = rl.model || "Not configured";
  const gravity = rl.imu?.projected_gravity;
  rlGravity.textContent = Array.isArray(gravity)
    ? gravity.map((value) => Number(value).toFixed(2)).join(" / ")
    : "Waiting";
  rlTargets.textContent = `${Array.isArray(rl.targets) ? rl.targets.length : 0} / 12`;
  const temperatureVerification = Array.isArray(rl.temperature_verification)
    ? rl.temperature_verification[0]
    : null;
  rlDetail.textContent = rl.error
    ? `FAULT: ${rl.error}`
    : temperatureVerification
      ? `VERIFYING TEMP ID ${temperatureVerification.motor_id}: ` +
        `${temperatureVerification.temperature_c} C / ` +
        `${Number(temperatureVerification.elapsed_s).toFixed(1)} / ` +
        `${Number(temperatureVerification.required_s).toFixed(1)} s / ` +
        `${temperatureVerification.high_sample_count} readings`
      : rl.active
        ? `${Number(rl.elapsed_s || 0).toFixed(1)} / ` +
          `${Number(rl.duration_s || 5).toFixed(1)} s / ` +
          `${Number(rl.forward_m_s || 0).toFixed(3)} m/s / then center + hold`
        : `${Number(rl.control_hz || 60).toFixed(0)} Hz policy / ` +
          `0-0.100 m/s / 1-60 s / completion centers + holds`;
  startRl.disabled =
    !rl.available ||
    rl.active ||
    crawl.active ||
    state.any_armed ||
    !rlSafetyAck.checked;
  stopRl.disabled = !rl.active;
  rlSpeed.disabled = rl.active;
  rlDuration.disabled = rl.active;
  rlSafetyAck.disabled =
    !rl.available || rl.active || crawl.active || state.any_armed;

  const phaseText = crawl.phase.replaceAll("_", " ").toUpperCase();
  const swingText = crawl.swing_corner
    ? ` / ${crawl.swing_corner.replaceAll("_", " ").toUpperCase()}`
    : "";
  const swingPairText = Array.isArray(crawl.swing_pair) && crawl.swing_pair.length
    ? ` / ${crawl.swing_pair
        .map((corner) => corner.replaceAll("_", " ").toUpperCase())
        .join(" + ")}`
    : "";
  const pushText = crawl.push_partner
    ? ` / PUSH ${crawl.push_partner.replaceAll("_", " ").toUpperCase()}`
    : "";
  gaitPanel.classList.toggle("active", crawl.active);
  const cycleText = crawl.stage === "walking"
    ? ` / CYCLE ${crawl.completed_cycles + 1}`
    : "";
  gaitStage.textContent = crawl.active
    ? `${crawl.stage.toUpperCase()}${cycleText} / ${(crawl.progress * 100).toFixed(0)}%`
    : crawl.stage === "complete"
      ? "COMPLETE / HOLDING"
      : "READY / DISARMED";
  gaitPhase.textContent = `${phaseText}${swingText}${swingPairText}${pushText}`;
  gaitProgress.style.width = `${Math.max(0, Math.min(100, crawl.progress * 100))}%`;
  gaitDetail.textContent = crawl.pattern === "diagonal_pair_flat_support_gait_v1"
    ? `Continuous / 2 diagonal placements per cycle / ` +
      `${crawl.stride_mm.toFixed(0)} mm stride / ` +
      `${crawl.lift_mm.toFixed(0)} mm lift / 2 planted shoes / ` +
      `STOP to end`
    : crawl.pattern === "rectangular_flat_support_crawl_v8"
      ? `Continuous / ` +
        `${(crawl.stride_mm).toFixed(0)} mm stride / ` +
        `${(crawl.lift_mm).toFixed(0)} mm lift / ` +
        `3 planted shoes stay flat / ` +
        `STOP to end`
      : `${crawl.stride_mm.toFixed(0)} mm stride / ` +
      `${crawl.lift_mm.toFixed(0)} mm lift / ` +
      `${crawl.stance_fore_aft_mm.toFixed(0)} mm front/rear splay / ` +
      `${crawl.abduction_deg.toFixed(0)} deg outward / ` +
      "STOP to end";

  walkForward.disabled =
    crawl.active ||
    rl.active ||
    state.any_armed;
  walkDiagonalPair.disabled =
    crawl.active ||
    rl.active ||
    state.any_armed;
  setCrawlStance.disabled =
    crawl.active || rl.active;
  stopWalk.disabled = !state.any_armed;
  zeroAll.disabled = crawl.active || rl.active;
  centerAll.disabled = crawl.active || rl.active;
  captureZeroAll.disabled = crawl.active || rl.active || state.any_armed;
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    latestState = await api("/api/state");
    refreshRecentErrors(latestState);
    latestState.legs.forEach(updateLeg);
    updateSummary(latestState);
    connected = true;
    connection.className =
      latestState.crawl.active || latestState.rl_policy?.active
      ? "connection armed"
      : latestState.any_armed
      ? "connection armed"
      : "connection online";
    connectionText.textContent = latestState.rl_policy?.active
      ? "LIVE / RL POLICY ACTIVE"
      : latestState.crawl.active
      ? "LIVE / CRAWL ACTIVE"
      : latestState.any_armed
      ? "LIVE · TORQUE ARMED"
      : "LIVE · ALL DISARMED";
  } catch (error) {
    connected = false;
    connection.className = "connection offline";
    connectionText.textContent = "CONNECTION LOST";
    [
      disarmAll,
      zeroAll,
      centerAll,
      captureZeroAll,
      setCrawlStance,
      walkForward,
      walkDiagonalPair,
      stopWalk,
      rlSpeed,
      rlSafetyAck,
      startRl,
      stopRl,
      resetPower,
    ].forEach((control) => {
      control.disabled = true;
    });
    motorNodes.forEach((row) => {
      row.querySelectorAll("button, input").forEach((control) => {
        control.disabled = true;
      });
    });
    showNotice(error.message, true);
  } finally {
    refreshing = false;
  }
}

disarmAll.addEventListener("click", () => {
  postAction("/api/disarm-all", {}, "All 12 motors disarmed");
});

settingsButton.addEventListener("click", () => {
  settingsDialog.showModal();
});

settingsClose.addEventListener("click", () => {
  settingsDialog.close();
});

clearErrorLog.addEventListener("click", () => {
  recentErrors = [];
  storeRecentErrors();
  renderRecentErrors();
});

zeroAll.addEventListener("click", () => {
  settingsDialog.close();
  postAction("/api/zero-all", {}, "All armed motors returning to zero");
});

centerAll.addEventListener("click", () => {
  settingsDialog.close();
  postAction(
    "/api/center-all",
    { safety_ack: true, confirmation: "CENTER ALL 12" },
    "All 12 motors returning to calibrated zero; torque remains armed",
  );
});

captureZeroAll.addEventListener("click", () => {
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before capturing zero", true);
    return;
  }
  settingsDialog.close();
  postAction(
    "/api/capture-zero-all",
    { safety_ack: true, confirmation: "CAPTURE ZERO ALL" },
    "Current pose saved as calibrated zero for all 12 motors",
  );
});

setCrawlStance.addEventListener("click", () => {
  postAction(
    "/api/crawl-stance",
    { safety_ack: true, confirmation: "SET GAIT START STANCE" },
    "Moving all four legs to the distributed-push start stance",
  );
});

walkForward.addEventListener("click", () => {
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before starting the gait sequence", true);
    return;
  }
  postAction(
    "/api/crawl-forward",
    { safety_ack: true, confirmation: "TEST DISTRIBUTED CRAWL" },
    "Moving to the start stance, then crawling continuously until STOP",
  );
});

walkDiagonalPair.addEventListener("click", () => {
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before starting the gait sequence", true);
    return;
  }
  postAction(
    "/api/diagonal-pair-forward",
    { safety_ack: true, confirmation: "TEST DIAGONAL PAIR GAIT" },
    "Moving to the diagonal stance, then walking continuously until STOP",
  );
});

stopWalk.addEventListener("click", () => {
  postAction("/api/crawl-stop", {}, "Gait sequence stopped; all 12 motors disarmed");
});

rlSafetyAck.addEventListener("change", () => {
  if (latestState) updateSummary(latestState);
});

startRl.addEventListener("click", async () => {
  const forwardSpeed = Number(rlSpeed.value);
  const duration = Number(rlDuration.value);
  if (!Number.isFinite(forwardSpeed) || forwardSpeed < 0 || forwardSpeed > 0.1) {
    showNotice("RL speed must be between 0 and 0.100 m/s", true);
    return;
  }
  if (!Number.isFinite(duration) || duration < 1 || duration > 60) {
    showNotice("RL walk time must be between 1 and 60 seconds", true);
    return;
  }
  if (!rlSafetyAck.checked) {
    showNotice("Confirm support, foot clearance, and the physical cutoff", true);
    return;
  }
  if (latestState?.any_armed) {
    showNotice("Disarm all 12 motors before starting the RL test", true);
    return;
  }
  await postAction(
    "/api/rl-start",
    {
      forward_m_s: forwardSpeed,
      duration_s: duration,
      safety_ack: true,
      confirmation: "START SUPPORTED RL TEST",
    },
    `${duration}-second supported RL walk started at ${forwardSpeed.toFixed(3)} m/s`,
  );
  rlSafetyAck.checked = false;
  if (latestState) updateSummary(latestState);
});

stopRl.addEventListener("click", () => {
  postAction("/api/rl-stop", {}, "RL walking stopped; all 12 motors disarmed");
});

resetPower.addEventListener("click", () => {
  postAction(
    "/api/power-reset",
    {},
    "Power analytics reset; collecting a fresh idle reference",
  );
});

async function sendHeartbeat() {
  if (heartbeatInFlight) return;
  heartbeatInFlight = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 500);
  try {
    await api("/api/heartbeat", "POST", {}, false, controller.signal);
  } catch (_error) {
    // Heartbeat diagnostics are warning-only and independent of telemetry.
  } finally {
    clearTimeout(timeout);
    heartbeatInFlight = false;
  }
}

setInterval(sendHeartbeat, 700);

setInterval(refresh, 900);

renderRecentErrors();
sendHeartbeat();
refresh();
