# Diagonal walking revision 4: preserve swing and reward both feet

Prepared September 19 local time / September 20 UTC while revision 3 (R4)
finished its existing 3000-iteration run. The user authorized continuing local
diagnosis, adjustments and training without renewed permission. No Pi operation
or hardware deployment is included.

R4 subsequently completed all 3000 iterations with zero full paired landings.
Its final checkpoint SHA-256 is
`c0c23e1e2994543e07bf6de495ce03dd4dd935b69dc20703a9d1ffa14e4947d1`.
The last 100 rounded iteration summaries averaged 91.51% gate timeouts,
0.07336 rad/s body angular RMS and 0.42 mm/s stance slip. These are unweighted
training summaries; episodes ending at gates and body drift do not establish
walking. `20260919-seed2419-r4/result-summary.json` retains the complete result.

## Evidence for this correction

R4's completed moving episodes repeatedly timed out at lift, with no rear-foot
qualifying touchdowns or full paired steps. A deterministic checkpoint-750
diagnostic showed selected-leg requests cancelling approximately 19–20 mm of
the intended 22 mm lift. Removing random exploration did not resolve it.
Checkpoint-1250 comparisons showed that limiting swing corrections could pass
the lift-position gate, but still did not produce real paired clearance or
landings. These results are retained under
`simulation/reviews/ankle-v24-diagonal/`.

Exact URDF FK matched measured simulator ankle pivots within 0.00045 mm in the
checkpoint-1750 audit. Supporting-leg compression and body motion explain why
nominal planar height is not actual sole clearance. Restricted static-load
calculations suggested little torque margin, but subsequent narrower/upright
stance, attitude-compensation and support-ankle feedback prototypes failed
their dynamic diagnostics. None of those geometry/reference changes is adopted.

## Changes

1. During lift, advance and lower, the selected legs' learned hip/knee request
   is converted to a planar ankle-pivot position. It is constrained to within
   8 mm fore/aft of the nominal path and between 10 mm extra lift and 2 mm
   cancelled lift. The same IK converts the result back to joint targets.
   Supporting legs retain their original residual authority. Abduction and
   ankle residuals are unchanged. Physical joint bounds and output rate,
   packet and acceleration limits still apply afterward; joint clamping can
   affect Cartesian feasibility, so measured gates and clipping diagnostics
   remain necessary. This constraint does not measure or certify floor contact.
2. Real sole-clearance shaping increases monotonically from zero toward each
   foot's requested height, instead of using a Gaussian that gave little credit
   near the floor. Both opposite support feet must remain supporting. Excess
   clearance above the reference plus 10 mm has a separate bounded penalty.
3. Each selected foot can earn a bounded, once-per-placement lift-progress
   bonus (maximum 0.4 per foot). The existing pair-minimum lift progress bonus
   remains. Repeating the same bobbing height cannot earn this bonus again.
   A small support-qualified force-unloading reward bridges the moment before
   real clearance appears. Unloading alone is not a successful step.
4. During lift/advance, quiet stance/support rewards scale from 25% to 100%
   with achieved clearance, reducing the reward for remaining planted. Quiet
   standing episodes keep their reward. Forward velocity reward applies to the
   all-foot push stage, preventing failed lift attempts from earning it merely
   by leaning or shifting the body.
5. Per-leg supported lift-clear fractions, the unconstrained requested target
   and which joint requests were projected are logged. Full paired landings,
   per-leg completed strides, timeout stages, contact evidence and blockers
   remain separate.

The broad 80/280 mm stance, startup half-step, 80–100 mm final stride targets,
22 mm lift, six-stage schedule, original calibration, USD/masses, conservative
torque-speed envelope, motor randomization and stopping behavior are unchanged.
The full-pair success criteria and timeout failure handling are unchanged.
Startup half-steps never count as full paired landings.

## Bounded candidate check

The revision was first prepared in an isolated source copy, leaving R4's active
source untouched. `revision4-checkpoint1250/report.json` records four first
episodes with the saved checkpoint, its matching 0.8333 curriculum fraction,
both starting pairs, and nominal/85% motors. All four passed lift; two reached
settle, two stopped at advance. All ultimately timed out, with no full paired
landings. This establishes execution and improved scheduler progression only;
the checkpoint was not trained for the revised reward/controller and does not
qualify the gait. The next training run must establish actual rear clearance.

## Training runs and infrastructure recovery

R5 (`20260919-seed2419-r5`) started after R4 finished, but its supervisor
stopped around iteration 120 because Windows denied an atomic replacement of
`status.json`. The worker then failed writing to its closed stdout pipe.
Only `model_0.pt` was saved. This was a logging failure, not a completed training
result; its original status, logs and traceback are preserved in the run folder.
Early front-foot clearance improved, but rear clearance and full pairs remained
zero, so the partial run does not establish walking.

The supervisor now retries transient Windows sharing/access conflicts, treats
progress-write errors as nonfatal, and sends worker output directly to a file.
The gait-summary snapshot uses the same retry/nonfatal writer; its in-memory
totals continue accumulating if a snapshot cannot be saved.
Its monitor tails that file independently. Unexpected monitor failures retain
the run lock while the worker finishes and are recorded as failures, rather
than silently reporting completion. Terminal completion also requires no worker
error file. A separate final-status file is a fallback if the usual status file
remains locked.

Use `20260919-seed2419-r6`, a fresh output directory, for the restart.
Initialize from the original V23 walking actor (not a stair actor or the
failed R4 final policy), with a fresh critic/optimizer, seed 2419, 256
environments, and 3000 iterations. Preserve source/calibration provenance,
the old run and all diagnostic evidence. Read `status.json` for actual launch
and completion; preparation of this document is not evidence of a started job.

This remains an experimental stateful controller. Its weights cannot be
copied into the deployed V23 runtime. Continue periodic monitoring and perform
a bounded independent evaluation of the matching final controller before
calling it ready for a supervised robot handoff.
