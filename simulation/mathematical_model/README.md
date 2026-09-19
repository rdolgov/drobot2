# Drobot2 mathematical model and executable manual

Start with [Drobot2_Model_Manual.ipynb](Drobot2_Model_Manual.ipynb).
It is an engineering manual: derive the equation, implement it, inspect its
values, then use it in the next stage. The final result is an importable
NumPy model, independent of Jupyter and Isaac Sim.

**Status:** implemented and source-inspected, but not executed or verified.
The notebook has empty outputs and null execution counts. No tests, linters,
type checking, builds, notebook execution, simulations or hardware checks
were run for this change. No packages were installed or servers started.

## Start Jupyter

From the repository root, run the following when you want to use the model:

    py -3 -m venv .venv-robot-model
    .\.venv-robot-model\Scripts\python.exe -m pip install -e ".\simulation\mathematical_model[notebook]"
    .\.venv-robot-model\Scripts\python.exe -m jupyterlab simulation/mathematical_model/Drobot2_Model_Manual.ipynb

Python 3.10 or newer is required. Select this environment if opening the
notebook in VS Code. Execute cells in order; their values and plots are
computed from the selected URDF. These commands have not been run by Codex.

The runtime package needs only NumPy. The notebook extra adds JupyterLab,
IPython's kernel, Matplotlib and SymPy. Installation uses the package manager;
it does not install or import Isaac Sim.

## Model profiles

| Name | Movable joints | Endpoint and mass source |
| --- | ---: | --- |
| original | 12 | Original URDF, fork-proxy center, approximately 4.526 kg legacy estimates |
| rigid_shoe | 12 | Calibrated baseline URDF, explicit fixed foot bodies, 3.18 kg nominal total |
| active_ankle | 16 | Active concept URDF, explicit moving foot bodies, 3.50 kg nominal total |

The notebook defaults to **rigid_shoe**, preserving the original three-joint
leg geometry while including consistent shoe mass. This is the isolated
baseline concept's estimated mass distribution, not a new measurement of the
physical robot. The original bare-fork proxy center is not its sphere-bottom
contact; sole_contacts deliberately requires explicit foot bodies.

Read source provenance from model.urdf_path and model.source_sha256.
The files under cad/exports/urdf remain authoritative and are never modified.
The model rejects unsupported analytical leg geometry instead of silently
using stale dimensions. URDFRobot remains available for general fixed and
revolute trees; it does not assume Drobot's closed-form leg geometry.

## Reuse from a Python program

    from drobot_model import DrobotModel

    robot = DrobotModel.from_repository(
        r"C:\Users\roman\Documents\dev\drobot2", variant="rigid_shoe"
    )
    q, body_pose = robot.stance()
    feet = robot.foot_positions(q, base_pose=body_pose)
    M = robot.mass_matrix(q, base_pose=body_pose, floating=True)

There is also a complete standalone
[foot-target example](examples/foot_target.py). After installing:

    .\.venv-robot-model\Scripts\python.exe simulation/mathematical_model/examples/foot_target.py --repo .

That example computes joint targets and model quantities. It sends no
commands to hardware.

## Implemented operators

- Rigid transforms, SO(3) exponential, screw exponential and adjoint.
- URDF tree FK and arbitrary link-point geometric Jacobians.
- Closed-form 3/4-joint leg FK; position, angular, space and body Jacobians.
- Analytical IK branches and limit handling; bounded damped numerical IK
  with explicit success/residual reporting.
- Named world-foot target IK with body-pose conversion.
- COM positions, inertia rotation, kinetic-energy mass matrix and gravity.
- Recursive Newton–Euler inverse dynamics and bias terms, fixed or floating.
- Forward dynamics by a linear solve.
- Minimum-norm static point-force balance.
- Prescribed independent sticking contact dynamics and force diagnostics.

See [the manual](Drobot2_Model_Manual.ipynb) for equations, implementation
steps and complete examples. See the source docstrings for frame and shape
contracts. All dynamics are instantaneous operators; no time integration
or automatic contact-mode selection is provided.

## Important modeling boundaries

- SI units and URDF joint signs. Joint order is explicit by name.
- Floating speeds are [world base linear, world base angular, joint rates].
  Geometric/twist six-vectors are angular first.
- External wrenches act ON the robot; their moments are about the specified
  point. Inverse dynamics subtracts these generalized external forces.
- Floating inverse dynamics returns six base wrench entries before joint
  torques. These are feasibility requirements, not six extra actuators.
- Source effort/speed limits are exposed. Dynamics does not clamp effort or
  implement a motor torque-speed curve; check feasibility separately.
- Source damping/Coulomb friction is optional. Zero-speed stiction, gearbox
  rotor inertia, compliance, latency and thermal behavior are not identified.
- Contact solves assume the caller supplied the active, stationary,
  independent point contacts and a compatible velocity. There is no terrain,
  collision, impact or contact switching model. A point sole cannot transmit
  an independent contact moment.
- ContactSolution.force_feasible checks normal/friction forces only. None
  means a friction coefficient was not specified; it does not imply known
  physical feasibility. It does not check motor limits or tipping stability.
- The static force solve is minimum-norm equality balance, not a constrained
  optimizer. Negative normal or friction-infeasible forces are reported.
- Mass distribution and inertias are source assumptions. A 3.18 kg total
  does not establish the accuracy of each link's COM or inertia tensor.

## Files and editing

- [Drobot2_Model_Manual.ipynb](Drobot2_Model_Manual.ipynb): primary notebook.
- [Drobot2_Model_Manual.py](Drobot2_Model_Manual.py): matching readable
  percent-cell source, usable in editors with Python notebook cells.
- [drobot_model](drobot_model): reusable package.
- [figures/leg-chain.svg](figures/leg-chain.svg): authored coordinate diagram.
- [tools/write_notebook.py](tools/write_notebook.py): serialize the cell source
  to notebook JSON without executing its code.

Edit either notebook cells directly or the percent-cell source. If editing
the source, regenerate the notebook with:

    py simulation/mathematical_model/tools/write_notebook.py

Regeneration overwrites the notebook and clears its outputs; save any manual
notebook changes first. This is a document serialization step, not model
execution. No ongoing synchronization runs in the background.
