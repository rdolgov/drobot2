# Working mass estimate from the recorded robot and four new motors

The user supplied the method on September 20: use the recorded 12-servo robot
weight plus the four new servos' specified weight. This is sufficient to
continue diagnosis; a new weigh-in is not a prerequisite.

`simulation/docs/rl-robust-straight-low-stance-v25.md:22` records the assembled
12-servo robot at **7 lb**, including its installed battery assembly. Converting
the units gives 3.17514659 kg. This many digits expresses the unit conversion,
not the precision of the original measurement. Do not add the battery again.

The repository's servo ledger identifies the 12 V **ST-3215-C018**, also marketed
as STS3215/ST3215. The [Feetech product specification](https://www.feetechrc.com/525603.html)
was checked September 20 and specifies **55 ± 1 g** per motor. Therefore:

| Item | Mass |
| --- | ---: |
| Recorded 12-servo robot | 3.175 kg |
| Four new motors | 0.220 kg |
| Working 16-servo estimate | **3.395 kg, approximately 7.49 lb** |

The four motors' specified tolerance contributes ±4 g worst case. This is not
a total-robot confidence interval: baseline scale precision and net changes in
printed feet, mounts, fasteners or wiring are not quantified by that tolerance.

The current V2.3 asset already includes all 16 servos and estimates 3.444020 kg.
It is 48.873 g (1.419%) heavier than this simpler estimate. Do not add the four
motors again to that asset. Its extra detail includes CAD-derived foot/shank
masses and mounting/tread allowances, not a new physical weigh-in.

The bounded `r7-model249-mass3395` diagnostic applies the 3.39514659 kg total
through a uniform density perturbation, scaling both mass and inertia while
preserving centers of mass and relative distribution. This is a sensitivity
comparison to the saved R7 default-controller diagnostic, not a new authoritative
per-part mass ledger or a modified asset. Torque limits, controller, success
criteria and checkpoint remain unchanged. The completed four-trial result
still has zero full pairs and zero qualifying rear touchdowns: both starting
pairs, at nominal and 85% motor strength, timed out. The 1.419% mass reduction
does not resolve this controller's failed steps. See
[`r7-model249-mass3395/report.json`](r7-model249-mass3395/report.json).

Exact arithmetic and source references are saved in `mass-estimate-20260920.json`.
