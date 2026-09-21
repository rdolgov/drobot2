# Diagonal start availability message

Installed on the Pi at 2026-09-21 00:05 UTC (September 20 local time).
The static HTML and JavaScript now display the reason diagonal start is
disabled, explain the disarmed-start workflow, and show armed idle state
correctly. Diagonal start is also disabled while RL initialization is pending.

Source inspection found that both the UI and controller require disarmed
motors for diagonal start. SET GAIT START STANCE arms the motors and prepares
the crawl stance. This update preserves that controller requirement; it does
not claim to have observed the user's current motor or browser state.

After supporting the robot and selecting DISARM ALL, the diagonal start
button initializes ankle leveling and moves to its own stance. The UI now
states this beside the button. Refresh the page to load the new static files.

Only the two static files were replaced. The controller service was not
restarted, and no motor command was sent. No tests, compilation, simulations,
browser validation, hardware checks, or walking were run.

Pi backup:
`/home/rd/drobot2/debug/deployment-backups/diagonal-ui-start-help-20260921T000528150267Z`

Archive SHA-256:
`e6a248c68f15a4c5c9a056172348e219d93d1146825266c5ff91ac2ca84734f9`

The installer checks the previous static-file hashes before replacing them,
backs up their exact bytes, and restores them on a write failure. The original
V2 deployment package remains historical; its expected static-file hashes
now differ from this update, so reapplying it requires deliberate review.
