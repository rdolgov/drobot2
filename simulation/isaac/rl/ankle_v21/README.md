# Printable ankle V2.1 training

The owning [training document](../../../docs/rl-ankle-v21-planted.md) records
the task, authorization, model assumptions, rewards, exact commands and
completion/recording workflow. Live status is
[`workflow.json`](../../../reviews/ankle-v21-planted/workflow.json).

`env.py` extends only the shared mechanics of the isolated V2 pilot. The
original V2 tasks and V43/V44 remain unchanged. These 16-action policies are
not compatible with the existing 12-motor hardware runtime.
