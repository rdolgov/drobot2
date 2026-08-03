# V82 front-left transfer unload reward

This package is a bounded negative/intermediate result, not a stair-climbing
policy. It trains six compact outputs during the transfer from a verified
front-right foothold into front-left swing: front-left hip abduction, hip
flexion, and knee plus the three support-leg hip-abduction joints.

The exact simulator stair is `180 mm` rise by `250 mm` tread. Applied joint
effort is capped at `0.8825985 N m`. The policy is camera-blind; RGB is used
only for review recording.

Training ran for 4,096 PPO steps on seed 1034 after adding a direct reward for
reducing the next swing foot's measured normal load. Training reduced the
minimum swing-foot load to `5.145 N`, versus `7.232 N` in the preceding V81
run, while retaining the completed tread contact. It did not reach the `1 N`
unload gate and completed zero transfers.

Fresh evaluation on seeds 1036 onward produced one transfer-phase episode in
three. It retained `20.010 N` on the front-right tread with `31.576 mm`
maximum support slip, but the front-left foot remained loaded at `7.979 N`
and lifted only `6.707 mm`. The other two episodes exceeded the strict
`35 mm` slip cap before completing the first foothold. Success rate is `0/3`.

Use the packaged model only for continued transfer-phase research. The proven
isolated 190 mm lift remains V80; V82 does not compose that lift after a real
first stair foothold.
