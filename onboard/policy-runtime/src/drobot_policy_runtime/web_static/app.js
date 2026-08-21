const token = new URLSearchParams(location.search).get("token") || sessionStorage.getItem("drobotToken") || "";
if (token) sessionStorage.setItem("drobotToken", token);

const byId = (id) => document.getElementById(id);
const speed = byId("speed");
const speedValue = byId("speedValue");
const errorBox = byId("error");
byId("manual").href = `${location.protocol}//${location.hostname}:8080/`;

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: {"X-Drobot-Token": token, "Content-Type": "application/json"},
    body: body ? JSON.stringify(body) : null,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

function vectorCard(label, values, unit = "") {
  if (!values) return `<div class="vector"><span>${label}</span><strong>Waiting...</strong></div>`;
  return `<div class="vector"><span>${label}</span><strong>${values.map((v) => Number(v).toFixed(3)).join(" &middot; ")}</strong><small>${unit}</small></div>`;
}

function render(state) {
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
  byId("imu").innerHTML = [
    vectorCard("Angular velocity", imu?.angular_velocity_rad_s, "rad/s - x y z"),
    vectorCard("Projected gravity", imu?.projected_gravity, "normalized - x y z"),
    vectorCard("Linear acceleration", imu?.linear_acceleration_m_s2, "m/s&sup2; - x y z"),
  ].join("");

  byId("motors").innerHTML = (state.motors || []).map((motor) => `
    <article class="motor">
      <div><span class="servo">ID ${motor.servo_id}</span><span class="action">${Number(motor.normalized_action).toFixed(3)}</span></div>
      <strong>${Number(motor.target_deg).toFixed(1)}&deg;</strong>
      <small>${motor.joint.replaceAll("_", " ")}</small>
    </article>`).join("") || '<p class="waiting">Start inference to see targets.</p>';

  errorBox.hidden = !state.error;
  errorBox.textContent = state.error || "";
}

async function refresh() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
  }
}

speed.addEventListener("input", () => speedValue.textContent = `${Number(speed.value).toFixed(2)} m/s`);
speed.addEventListener("change", async () => {
  try { render(await api("/api/command", "POST", {forward_m_s: Number(speed.value)})); }
  catch (error) { errorBox.hidden = false; errorBox.textContent = error.message; }
});
byId("start").addEventListener("click", async () => {
  try { render(await api("/api/start", "POST", {forward_m_s: Number(speed.value)})); }
  catch (error) { errorBox.hidden = false; errorBox.textContent = error.message; }
});
byId("stop").addEventListener("click", async () => {
  try { render(await api("/api/stop", "POST", {})); }
  catch (error) { errorBox.hidden = false; errorBox.textContent = error.message; }
});

refresh();
setInterval(refresh, 300);
