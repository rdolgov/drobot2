import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const inputDir = process.argv[2];
if (!inputDir) {
  throw new Error("usage: node analyze_trial.mjs <extracted-recording-directory>");
}

const samples = fs
  .readFileSync(path.join(inputDir, "samples.jsonl"), "utf8")
  .trim()
  .split(/\r?\n/)
  .map(JSON.parse);
const events = fs
  .readFileSync(path.join(inputDir, "events.jsonl"), "utf8")
  .trim()
  .split(/\r?\n/)
  .filter(Boolean)
  .map(JSON.parse);
const metadata = JSON.parse(
  fs.readFileSync(path.join(inputDir, "metadata.json"), "utf8"),
);

const names = metadata.robot.joint_order;
const radToDeg = 180 / Math.PI;

function quantile(values, q) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  const fraction = index - lo;
  return sorted[lo] * (1 - fraction) + sorted[hi] * fraction;
}

function stats(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const sum = finite.reduce((a, b) => a + b, 0);
  return {
    min: Math.min(...finite),
    p50: quantile(finite, 0.5),
    p95: quantile(finite, 0.95),
    max: Math.max(...finite),
    mean: sum / finite.length,
    rms: Math.sqrt(finite.reduce((a, b) => a + b * b, 0) / finite.length),
  };
}

function diffs(values) {
  return values.slice(1).map((value, i) => value - values[i]);
}

function perAxis(rows, transform = (x) => x) {
  return names.map((name, axis) => ({
    joint: name,
    ...stats(rows.map((row) => transform(row[axis]))),
  }));
}

function uniqueRate(timestamps) {
  const unique = [...new Set(timestamps)].sort((a, b) => a - b);
  const intervals = diffs(unique);
  const duration = unique.length > 1 ? unique.at(-1) - unique[0] : 0;
  return {
    distinct_samples: unique.length,
    achieved_hz: duration > 0 ? (unique.length - 1) / duration : null,
    interval_ms: stats(intervals.map((x) => x * 1000)),
  };
}

function correlation(a, b) {
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  return covariance / Math.sqrt(varianceA * varianceB);
}

function trackingLag(axis) {
  const targets = steadyIndices.map((index) => limitedRows[index][axis]);
  const measured = steadyIndices.map((index) => measuredRows[index][axis]);
  let best = { lag_samples: 0, correlation: correlation(targets, measured) };
  for (let lag = 1; lag <= 24; lag += 1) {
    const value = correlation(targets.slice(0, -lag), measured.slice(lag));
    if (Number.isFinite(value) && value > best.correlation) {
      best = { lag_samples: lag, correlation: value };
    }
  }
  return {
    joint: names[axis],
    ...best,
    lag_ms: best.lag_samples * (1000 / metadata.trial.control_hz),
  };
}

const sampleTimes = samples.map((s) => s.monotonic_time_s);
const sampleIntervals = diffs(sampleTimes);
const rollDeg = samples.map((s) => {
  const [, gy, gz] = s.imu.projected_gravity_body;
  return Math.atan2(gy, -gz) * radToDeg;
});
const pitchDeg = samples.map((s) => {
  const [gx, gy, gz] = s.imu.projected_gravity_body;
  return Math.atan2(-gx, Math.sqrt(gy * gy + gz * gz)) * radToDeg;
});
const integratedAngularDisplacementDeg = [0, 1, 2].map((axis) =>
  sampleIntervals.reduce(
    (sum, dt, index) =>
      sum +
      0.5 *
        (samples[index].imu.angular_velocity_body_rad_s[axis] +
          samples[index + 1].imu.angular_velocity_body_rad_s[axis]) *
        dt *
        radToDeg,
    0,
  ),
);
const integratedAbsoluteAngularTravelDeg = [0, 1, 2].map((axis) =>
  sampleIntervals.reduce(
    (sum, dt, index) =>
      sum +
      0.5 *
        (Math.abs(samples[index].imu.angular_velocity_body_rad_s[axis]) +
          Math.abs(samples[index + 1].imu.angular_velocity_body_rad_s[axis])) *
        dt *
        radToDeg,
    0,
  ),
);

const trackingRows = samples.map((s) =>
  s.rate_limited_target_rad.map((target, i) => (target - s.joints.position_rad[i]) * radToDeg),
);
const actionRows = samples.map((s) => s.action);
const requestedRows = samples.map((s) => s.requested_target_rad);
const limitedRows = samples.map((s) => s.rate_limited_target_rad);
const measuredRows = samples.map((s) => s.joints.position_rad);
const measuredVelocityRows = samples.map((s) => s.joints.velocity_rad_s);

const actionDeltaRows = actionRows.slice(1).map((row, n) =>
  row.map((value, i) => value - actionRows[n][i]),
);
const actionSecondDeltaRows = actionDeltaRows.slice(1).map((row, n) =>
  row.map((value, i) => value - actionDeltaRows[n][i]),
);
const limitedVelocityRows = limitedRows.slice(1).map((row, n) =>
  row.map((value, i) => ((value - limitedRows[n][i]) / sampleIntervals[n]) * radToDeg),
);
const limitedAccelerationRows = limitedVelocityRows.slice(1).map((row, n) =>
  row.map((value, i) => (value - limitedVelocityRows[n][i]) / sampleIntervals[n + 1]),
);

const requestLimiterGapRows = requestedRows.map((row, n) =>
  row.map((value, i) => (value - limitedRows[n][i]) * radToDeg),
);
const steadyIndices = samples
  .map((sample, index) => ({ sample, index }))
  .filter(({ sample }) => sample.elapsed_s >= 1.2)
  .map(({ index }) => index);
const steadyTrackingRows = steadyIndices.map((index) => trackingRows[index]);
const steadyLimiterGapRows = steadyIndices.map((index) => requestLimiterGapRows[index]);

function absPerAxis(rows) {
  return names.map((name, axis) => ({
    joint: name,
    ...stats(rows.map((row) => Math.abs(row[axis]))),
  }));
}

const motorDiagnostics = events.filter((e) => e.type === "motor_diagnostic");
const clampEvents = events.filter((e) => e.type === "rl_target_step_clamped");

const actionSummary = names.map((name, axis) => {
  const values = actionRows.map((row) => row[axis]);
  return {
    joint: name,
    ...stats(values),
    saturated_fraction_abs_ge_0_95:
      values.filter((v) => Math.abs(v) >= 0.95).length / values.length,
  };
});

const metrics = {
  recording_id: metadata.recording_id,
  model_sha256: metadata.policy.model_sha256,
  trial: metadata.trial,
  samples: {
    count: samples.length,
    observed_span_s: sampleTimes.at(-1) - sampleTimes[0],
    achieved_hz: (samples.length - 1) / (sampleTimes.at(-1) - sampleTimes[0]),
    interval_ms: stats(sampleIntervals.map((x) => x * 1000)),
    intervals_below_10_ms: sampleIntervals.filter((x) => x < 0.010).length,
    intervals_above_25_ms: sampleIntervals.filter((x) => x > 0.025).length,
    missed_deadlines_final: samples.at(-1).control_timing.missed_deadlines_total,
  },
  sensors: {
    joint_feedback: uniqueRate(samples.map((s) => s.sensor_time_s.joints)),
    imu: uniqueRate(samples.map((s) => s.sensor_time_s.imu)),
    joint_age_ms: stats(samples.map((s) => (s.monotonic_time_s - s.sensor_time_s.joints) * 1000)),
    imu_age_ms: stats(samples.map((s) => (s.monotonic_time_s - s.sensor_time_s.imu) * 1000)),
  },
  body: {
    roll_deg: stats(rollDeg),
    pitch_deg: stats(pitchDeg),
    angular_velocity_abs_rad_s: [0, 1, 2].map((axis) =>
      stats(samples.map((s) => Math.abs(s.imu.angular_velocity_body_rad_s[axis]))),
    ),
    integrated_angular_displacement_deg: integratedAngularDisplacementDeg,
    integrated_absolute_angular_travel_deg: integratedAbsoluteAngularTravelDeg,
    accelerometer_norm_m_s2: stats(
      samples.map((s) => Math.hypot(...s.imu.linear_acceleration_body_m_s2)),
    ),
    accelerometer_delta_norm_m_s2: stats(
      samples.slice(1).map((s, n) => {
        const previous = samples[n].imu.linear_acceleration_body_m_s2;
        const current = s.imu.linear_acceleration_body_m_s2;
        return Math.hypot(...current.map((v, i) => v - previous[i]));
      }),
    ),
  },
  joints: {
    tracking_error_abs_deg: absPerAxis(trackingRows),
    tracking_error_abs_deg_after_1_2_s: absPerAxis(steadyTrackingRows),
    measured_position_deg: perAxis(measuredRows, (x) => x * radToDeg),
    measured_velocity_abs_deg_s: absPerAxis(
      measuredVelocityRows.map((row) => row.map((x) => x * radToDeg)),
    ),
    requested_target_deg: perAxis(requestedRows, (x) => x * radToDeg),
    rate_limited_target_deg: perAxis(limitedRows, (x) => x * radToDeg),
    rate_limited_velocity_abs_deg_s: absPerAxis(limitedVelocityRows),
    rate_limited_acceleration_abs_deg_s2: absPerAxis(limitedAccelerationRows),
    request_to_limiter_gap_abs_deg: absPerAxis(requestLimiterGapRows),
    estimated_tracking_lag_after_1_2_s: names.map((_name, axis) =>
      trackingLag(axis),
    ),
  },
  policy: {
    action: actionSummary,
    action_delta_abs: absPerAxis(actionDeltaRows),
    action_second_delta_abs: absPerAxis(actionSecondDeltaRows),
    samples_with_any_request_limiter_gap_over_2_deg: requestLimiterGapRows.filter(
      (row) => row.some((v) => Math.abs(v) > 2),
    ).length,
    limiter_gap_over_2_deg_fraction_by_joint_after_1_2_s: names.map(
      (joint, axis) => ({
        joint,
        fraction:
          steadyLimiterGapRows.filter((row) => Math.abs(row[axis]) > 2).length /
          steadyLimiterGapRows.length,
      }),
    ),
    clamp_event_count: clampEvents.length,
  },
  electrical_diagnostics: {
    samples: motorDiagnostics.length,
    voltage_v: stats(motorDiagnostics.map((e) => e.voltage_v)),
    current_ma: stats(motorDiagnostics.map((e) => e.current_ma)),
    temperature_c: stats(motorDiagnostics.map((e) => e.temperature_c)),
  },
};

const outputPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "analysis-metrics.json",
);
fs.writeFileSync(outputPath, `${JSON.stringify(metrics, null, 2)}\n`);
console.log(JSON.stringify(metrics, null, 2));
