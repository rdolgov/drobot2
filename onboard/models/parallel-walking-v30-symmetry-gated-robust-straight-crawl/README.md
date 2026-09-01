# V30 symmetry-gated robust straight crawl

`model_5000.onnx` is an experimental, user-requested Raspberry Pi deployment
of the V30 diagnostic checkpoint. It is not an accepted release policy.

Nominal simulation was fall-free and substantially straighter than prior
branches, but the policy missed the exact-contact, three/four-foot support,
all-four-release, and sustained-effort acceptance gates. Begin physical trials
at the metadata-recommended `0.005 m/s`, with the robot supported and the stop
control immediately available.

Rollback policy:

```text
/home/rd/drobot2/onboard/models/parallel-walking-v24-padded-feet-forward-bias/model_3248.onnx
```

Training and evaluation evidence is recorded in
`simulation/docs/rl-symmetry-gated-robust-straight-v30.md`.
