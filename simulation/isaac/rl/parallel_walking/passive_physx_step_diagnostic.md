# Passive 120 Hz PhysX state diagnostic

Source prepared only. No simulation, test, hardware connection, deployment, or
change to the evaluator/environment/defaults was performed while adding it.

## Opt-in integration boundary

Use only inside an already authorized, single-environment V44 **PhysX** evaluation.
The caller retains ownership of episode timing, phase labels, output/provenance,
and final-hold selection. Refuse an existing output path before launching an
evaluation; `write_jsonl` additionally uses exclusive creation and never overwrites.
The output parent directory must already exist.

```python
from passive_physx_step_diagnostic import PassivePhysxStepDiagnostic

diagnostic = PassivePhysxStepDiagnostic(
    env.unwrapped,
    task=args.task,
    provenance={
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_report_path": str(output_path),
        "evaluation_source_sha256": evaluation_source_sha256,
        "environment_source_sha256": environment_source_sha256,
        "environment_config_sha256": environment_config_sha256,
        "environment_ablation": environment_ablation,
    },
    max_physics_steps=1000,
)
try:
    with diagnostic:
        # Existing evaluation loop; resets remain OUTSIDE control_step contexts.
        # Select final one second using existing evaluator schedule/hold boundary.
        with diagnostic.control_step(
            episode_index=episode,
            control_step_index=step,
            stage="ramp" if phase == "deceleration" else phase,
            capture=is_in_final_hold_second,
        ):
            observation, reward, done, extras = env.step(action)  # existing call
finally:
    diagnostic.write_jsonl(new_trace_path)  # after wrapper restoration, including failure
```

The example is an integration sketch, not a standalone simulator script. Do not
add another env.step, reset, render, force write, or target update for this helper.
Final one second in two episodes normally means 240 physics-step records, each
containing pre/post snapshots. Set capture=True for every ordinary control step
to inspect the whole trajectory, subject to the explicit maximum (1–12000).
Reaching the maximum stops recording, not simulation; metadata reports omitted
eligible steps and truncation. It does not silently retain a rolling window.

## Exact timing and interpretation

The inspected installed `DirectRLEnv.step` ordinary PhysX loop performs:

1. `_apply_action`, then `scene.write_data_to_sim`;
2. `sim.step(render=False)`;
3. optional rendering, then `scene.update(dt=physics_dt)`.

The wrapper reads immediately around step 2, calls the original bound method
once with unchanged arguments, and preserves its return value or exception.
It rejects backend-managed decimation: one backend call must represent one
existing 1/120-second step. The context manager restores the previous method;
if another owner replaced it, it records an error rather than overwrite them.

At the post-step boundary Lab caches have **not** received scene.update. That is
intentional evidence, not automatically a cache bug. Cached fields are copied
first. Direct backend getters follow and are also immediately copied: PhysX
getters may update buffers aliased by Lab, so keeping live tensors would erase
the distinction. Post-step `applied_torque` may still be the estimate computed
before stepping. Compare the two timestamps/boundaries, not just one scalar.

`get_dof_projected_joint_forces()` projects incoming joint forces into the DOF
motion direction according to the installed tensor API. It is independent of
Lab's algebraic implicit-PD estimate; it is not automatically isolated drive
torque or physical motor-current telemetry. The metadata documents units/order.
No capacity or torque source used by training/evaluation gates is replaced.

No contact arrays are captured by this helper. Existing `net_forces_w` is **net
normal force**, not normal plus friction. Locating closed-chain tangential
self-stress additionally needs separately enabled friction/contact-point data
and correctly framed joint reaction wrenches. This helper does not enable those
sensors or change scene configuration.

## Buffered offline comparisons

The JSONL header links caller-supplied provenance and the helper source hash,
and contains an offline summary per episode/stage/joint. It reports raw q/qdot,
pre/post and consecutive-post finite differences, cached-versus-backend gaps,
PD-estimate differences, and projected-force differences. Substep-specific
position/velocity statistics can expose a 120 Hz pattern hidden at 60 Hz.
Unavailable/nonfinite channels carry errors and have zero summary samples,
never invented zero error. Getter failure does not prevent the original physics
step; an original physics exception is recorded and re-raised.

Finite differences are interval averages. Neither endpoint qdot nor their
trapezoidal average is guaranteed to equal them exactly under constrained,
discrete solver integration. Large persistent differences motivate inspection;
they do not alone identify a cache defect or physical actuator bandwidth.
The host copies add synchronization overhead, so this is not a performance or
real-time-deadline benchmark. JSON writing occurs only after capture ends.
