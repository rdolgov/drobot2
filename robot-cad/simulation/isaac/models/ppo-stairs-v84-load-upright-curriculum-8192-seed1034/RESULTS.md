# V84 observed-load transfer curriculum

This is an intermediate transfer milestone, not a stair-climbing policy. The
policy has 82 inputs: the prior 81 camera-blind IMU/proprioceptive/contact and
analytic-stair fields plus the active swing foot's measured total normal load.
It controls six compact outputs: all three front-left joints and the three
support hip-abduction joints.

The exact simulator stair is `180 mm` rise by `250 mm` tread and the applied
joint-effort cap is `0.8825985 N m`. RGB is used only for the review video.

V84 continued V83 for 8,192 PPO steps from a deterministic retained
front-right foothold. Its curriculum used `(8 N, 0.975)`, `(4 N, 0.977)`, and
the deployment gate `(1 N, 0.9781476)`, requiring two accepted transfers per
stage. It completed two transfers at the 8 N stage and advanced to 4 N. The
minimum sampled swing load was `5.145 N`; the completed foothold remained
loaded and support slip stayed within the `35 mm` training cap. It did not pass
4 N or the strict 1 N gate.

Fresh deterministic evaluation on seed 1038 failed `0/3`: all three runs
exceeded the 35 mm slip cap, and only one reached the retained first foothold.
The recorded strict-gate episode is therefore diagnostic evidence, not a
success video. V80 remains the isolated 190 mm lift proof; V84 does not yet
compose that lift after a real first foothold.

The next bounded experiment should add support-leg hip-flexion authority and
explicit pitch/upright shaping while retaining the measured-load input and the
strict final gates.
