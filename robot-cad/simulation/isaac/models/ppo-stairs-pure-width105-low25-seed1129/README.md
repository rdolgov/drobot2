# Pure-PPO width105 lower-reset checkpoint

This package contains iteration 734 from the first gradual lower-posture stage.
It starts at 0.42 m instead of 0.46 m and interpolates every hip-flexion and
knee reset 25% from the normal stance toward the verified fully folded limits.
The task still requires a four-foot-supported landing inside the centered
210 mm band of the exact 180 mm rise by 250 mm tread.

## Measured result

- Low25 PPO: 11/1,900 stochastic training episodes (0.5789%) over 307,200
  transitions.
- Low25 low-entropy continuation: 9/3,094 (0.2909%) over 491,520 transitions;
  rejected because it did not improve the source run.
- Low50 transfer at 0.38 m and 50% fold: 0/2,166 over 307,200 transitions;
  rejected, so Low75 and Low100 were not started.
- Deterministic model 640 comparison: 0/298 episodes.
- Deterministic model 734 review seed 1133: 0/9 episodes. The associated video
  is a behavior review, not a certified success recording.
- SHA-256 of `model_734.pt`:
  `fbbabb961232413ef9031d6665cd01e91f5adc1784d707c7a00358fff6c63e53`.

This is evidence that a modestly lower starting posture improves stochastic
first-tread exploration, not evidence of a deterministic landing, body
transfer, completed step, or stair climb.

PPO controls all 12 joints without a gait phase, prescribed leg order, inverse
kinematics, or action trajectory. Actor inputs remain hardware-representable
IMU, VL53L5CX depth, joint position/velocity, previous action, and four foot
load/contact channels. RGB is review-only and effort remains capped at
0.8825985 N*m.
