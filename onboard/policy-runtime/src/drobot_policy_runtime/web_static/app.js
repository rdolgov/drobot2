const queryToken = new URLSearchParams(location.search).get("token") || "";
let token = queryToken || sessionStorage.getItem("drobotToken") || "";
if (queryToken) {
  sessionStorage.setItem("drobotToken", queryToken);
  history.replaceState(null, "", location.pathname);
}

const byId = (id) => document.getElementById(id);
const speed = byId("speed");
const speedValue = byId("speedValue");
const errorBox = byId("error");
const authBox = byId("auth");
const authMessage = byId("authMessage");
const tokenInput = byId("tokenInput");
byId("manual").href = `${location.protocol}//${location.hostname}:8080/`;

function showAuth(message) {
  authMessage.textContent = message;
  tokenInput.value = token;
  authBox.hidden = false;
}

function showError(message = "") {
  errorBox.hidden = !message;
  errorBox.textContent = message;
}

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: {
      ...(token ? {"X-Drobot-Token": token} : {}),
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : null,
  });
  const value = await response.json();
  if (!response.ok) {
    const error = new Error(value.error || `HTTP ${response.status}`);
    if (response.status === 403) {
      error.authRequired = true;
      showAuth("That token was not accepted. Copy the current token from the Pi and try again.");
    }
    throw error;
  }
  return value;
}

function vectorCard(label, values, unit = "") {
  if (!values) return `<div class="vector"><span>${label}</span><strong>Waiting...</strong></div>`;
  return `<div class="vector"><span>${label}</span><strong>${values.map((v) => Number(v).toFixed(3)).join(" &middot; ")}</strong><small>${unit}</small></div>`;
}

function render(state) {
  authBox.hidden = true;
  showError(state.error || "");
  const status = byId("status");
  status.textContent = state.status === "running" ? "Running" : state.status === "error" ? "Error" : "Stopped";
  status.className = `status ${state.status}`;
  byId("imuBackend").textContent = state.imu_backend?.toUpperCase() || "-";
  byId("policyAge").textContent = state.running
    ? "Live - 60 Hz"
    : state.last_policy_time_s ? "Last sample" : "Waiting";
  if (document.activeElement !== speed) speed.value = state.forward_m_s ?? 0.15;
  speedValue.textContent = `${Number(speed.value).toFixed(2)} m/s`;

  const imu = state.imu;
  const heading = state.heading_hold;
  byId("imu").innerHTML = [
    vectorCard("Angular velocity", imu?.angular_velocity_rad_s, "rad/s - x y z"),
    vectorCard("Projected gravity", imu?.projected_gravity, "normalized - x y z"),
    vectorCard("Linear acceleration", imu?.linear_acceleration_m_s2, "m/s&sup2; - x y z"),
    vectorCard(
      "Relative heading",
      heading ? [heading.current_relative_rad, heading.desired_relative_rad, heading.error_rad] : null,
      "rad - current / desired / error",
    ),
    vectorCard(
      heading?.enabled ? "Heading correction" : "Heading correction (disabled)",
      heading ? [heading.correction_rad_s, heading.effective_yaw_rad_s] : null,
      "rad/s - feedback / effective command",
    ),
  ].join("");

  byId("motors").innerHTML = (state.motors || []).map((motor) => `
    <article class="motor">
      <div><span class="servo">ID ${motor.servo_id}</span><span class="action">${Number(motor.normalized_action).toFixed(3)}</span></div>
      <strong>${Number(motor.target_deg).toFixed(1)}&deg;</strong>
      <small>${motor.joint.replaceAll("_", " ")}</small>
    </article>`).join("") || '<p class="waiting">Start inference to see targets.</p>';

}

async function refresh() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    if (!error.authRequired) showError(error.message);
  }
}

byId("tokenForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const candidate = tokenInput.value.trim();
  if (candidate.length < 16) {
    showAuth("The dashboard token must contain at least 16 characters.");
    return;
  }
  token = candidate;
  sessionStorage.setItem("drobotToken", token);
  await refresh();
});
byId("forgetToken").addEventListener("click", () => {
  token = "";
  sessionStorage.removeItem("drobotToken");
  tokenInput.value = "";
  authBox.hidden = true;
  refresh();
});

speed.addEventListener("input", () => speedValue.textContent = `${Number(speed.value).toFixed(2)} m/s`);
speed.addEventListener("change", async () => {
  try { render(await api("/api/command", "POST", {forward_m_s: Number(speed.value)})); }
  catch (error) { if (!error.authRequired) showError(error.message); }
});
byId("start").addEventListener("click", async () => {
  try { render(await api("/api/start", "POST", {forward_m_s: Number(speed.value)})); }
  catch (error) { if (!error.authRequired) showError(error.message); }
});
byId("stop").addEventListener("click", async () => {
  try { render(await api("/api/stop", "POST", {})); }
  catch (error) { if (!error.authRequired) showError(error.message); }
});

refresh();
setInterval(() => {
  if (authBox.hidden) refresh();
}, 300);
