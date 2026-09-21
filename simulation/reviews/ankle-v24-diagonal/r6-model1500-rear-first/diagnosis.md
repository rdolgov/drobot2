# Rear-first lift within each diagonal: rejected diagnostic

The same checkpoint-1500 actor and full curriculum used in
`r6-model1500-mean` were evaluated with a single isolated reference change:
the rear partner takes the first half of the lift stage and remains raised
while the front partner takes the second half. Total lift-stage time is
unchanged. During the delay the front foot's scheduled stance bit remains set.
The original residual projection remains active, so commanded ordering alone
does not prove actual physical ordering.

Four first-episode trials covered both starting diagonals with nominal and
85% motors. All timed out: two at settle and two at advance, at 6.00–10.35 s.
There were no full paired landings or qualifying rear touchdowns. The largest
sampled selected rear clearance was only 0.076 mm; no rear foot reached 8 mm.
Front feet still lifted, and one weaker trial produced a front-left touchdown.
No physical falls occurred before the early timeouts. The copied helper,
checkpoint, metadata, trace and `diagnostic-summary.json` preserve the result.

Reject this staggered timing change as a training default. It did not solve
rear unloading, and the actor was not trained for it. The result strengthens
the case for investigating loaded support behavior rather than assuming that
lift timing alone is sufficient. Actuator/joint limits, other reference stages,
and full-step qualification were unchanged. R6 continued with its original
contract. No Pi operation occurred.
