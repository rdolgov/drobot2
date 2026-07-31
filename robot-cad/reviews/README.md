# CAD review artifacts

Store editable CAD Viewer orthographic markup JSON and exported PNG review
sheets here. Prefer names that include the target and intent, for example:

`upper_arm-cable-route-three-view-markup.json`

The `legacy/` folder retains design intent brought from the source project.
Generated verification snapshots belong under `exports/renders/` instead.

Simulator review media may also be committed here when it is an intentional
handoff rather than bulk run output. `ppo-stairs-v5-10mm-four-step-success.mp4`
and its PNG/JSON companions record the verified stochastic shallow-stair
episode; the JSON states the deterministic and sim-to-real limitations.
The same reviewed clip is available on the private Sites handoff at
https://drobot-stairs-v3-smoke-20260731.romka.chatgpt.site.
