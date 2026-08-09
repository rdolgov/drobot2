# V80 isolated 190 mm front-foot lift

V80 is the deliberately simplified stair prerequisite: raise the front-left
foot at least `190 mm` for `0.50 s` while all three support feet remain loaded
and the robot does not tip. The world retains the exact `180 mm` stair rise and
`250 mm` tread depth, although this task does not attempt tread placement.

The policy was fine-tuned for 2,048 additional PPO steps from the verified V17
checkpoint. It uses the real-test encoder directions, joint limits, and the
`0.8825985 N m` applied effort cap. All six recent training episodes passed;
lift was `203.626-208.060 mm`, maximum tilt was `2.758 deg`, and maximum
support-foot slip was `3.382 mm`.

Independent deterministic seed-1031 evaluation passed `5/5` episodes:

| Metric | Result |
| --- | ---: |
| measured front-left lift | `204.900-208.077 mm` |
| required lift hold | `0.50 s` in every episode |
| maximum body tilt | `2.324 deg` |
| maximum support-foot slip | `3.337 mm` |
| minimum support-foot normal load | `10.386 N` |

The 166-frame external-camera replay is
`reviews/ppo-stairs-v80-single-foot-190mm-eval-seed1031.mp4`. The recorded
trajectory reached `207.761 mm` lift with `2.130 deg` tilt and `3.174 mm`
support slip. Its SHA-256 is
`1f7800f75b7c38f93e545692d7da45d45f84c01c9d06874d91a8447b71e658ef`.

The policy is camera-blind. It consumes IMU, joint/proprioceptive state,
previous action, support contact/load, and the known analytic stair profile.
The RGB camera is used only to record review evidence. This result proves a
stable isolated foot lift; it does not prove a second-foot transfer or a full
stair climb.

## Reproduce

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v80_single_foot_190mm_ppo.py
```

The tracked policy SHA-256 is
`d583a32276eab4b00f897b551951d106635c7bfd10099151f9a6a635a571706`.
The training, evaluation, and recording report SHA-256 values are respectively
`cabfa7dfb683161814e0f7a33ccec112e7a6f78e23e4bddff14986750727bb29`,
`55585bcedc1ab36c90d96a3e3d3ef5771ce0a9504da90c6aff875fc1c74742bb`,
and `66050acd1997b600c972a4c0179c3460faa0f237c20850fb6f396f7fa65f1698`.
