# %% [markdown]
# # Drobot2: an executable modeling manual
#
# Build a usable robot model from its geometry, one calculation at a time.
# Each section introduces an equation, implements it, and carries the result
# into the next section. The final object can be imported into another Python
# program without Jupyter or Isaac Sim.
#
# **Main model:** original three joints per leg with rigid shoes, using the
# 12-joint calibrated baseline URDF. The original bare-fork URDF is also
# supported. An active-ankle extension appears at the end.
#
# **Execution status:** all code cells were authored but not executed.
# No model checks, notebook execution, simulation, builds or hardware checks
# were run when creating this manual. Outputs appear when you run the cells.
#
# ## 0. Set up the Python environment
#
# Jupyter is the notebook application; Python is the kernel running the cells.
# From the repository root, these commands create a separate environment:
#
#     py -3 -m venv .venv-robot-model
#     .\.venv-robot-model\Scripts\python.exe -m pip install -e ".\simulation\mathematical_model[notebook]"
#     .\.venv-robot-model\Scripts\python.exe -m jupyterlab simulation/mathematical_model/Drobot2_Model_Manual.ipynb
#
# Alternatively, open the notebook in VS Code and select that environment.
# Run cells in order. Installation commands are instructions, not notebook
# cells, and were not executed during authoring.
#
# | Contract | Choice |
# |---|---|
# | Body axes | +X forward, +Y left, +Z up |
# | Length / mass / time | meters / kilograms / seconds |
# | Joint coordinates | radians, URDF signs |
# | Joint order | FL, FR, RL, RR; abduction, flexion, knee, optional ankle |
# | Transform | T_AB maps B components into A |
# | Geometric twist | angular first: [omega, velocity at point] |
# | Floating speed | [base linear velocity in W, angular velocity in W, joint rates] |
# | Forces | external force ON robot; world components |
#
# ![The leg coordinate chain](figures/leg-chain.svg)

# %%
from pathlib import Path
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Set this explicitly only if you launch Jupyter outside the repository.
REPOSITORY_OVERRIDE = None  # Example: Path(r"C:\Users\roman\Documents\dev\drobot2")
if REPOSITORY_OVERRIDE is None:
    locations = [Path.cwd(), *Path.cwd().parents]
    REPO = next((p for p in locations if (p / "cad/exports/urdf/quadruped_robot.urdf").is_file()), None)
    if REPO is None:
        raise FileNotFoundError("Set REPOSITORY_OVERRIDE to your drobot2 repository.")
else:
    REPO = Path(REPOSITORY_OVERRIDE).resolve()
sys.path.insert(0, str(REPO / "simulation/mathematical_model"))

from drobot_model import (
    DrobotModel, URDFRobot, ExternalWrench, LEGS,
    transform, inverse_transform, rx, ry, rz, skew, so3_exp, screw_exp,
)
np.set_printoptions(precision=6, suppress=True, linewidth=110)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True})

# %% [markdown]
# ## 1. Load one coherent geometry and mass ledger
#
# A kinematic endpoint is not automatically a physical mass. Read each link's
# mass, center of mass and inertia from the same selected URDF.
#
# | Variant | Joints | Endpoint / inertials |
# |---|---:|---|
# | original | 12 | fork-proxy center; legacy solid-PLA mass estimates, about 4.526 kg |
# | rigid_shoe | 12 | physical fixed shoes; baseline concept scaled to 3.18 kg |
# | active_ankle | 16 | moving foot bodies and added actuators; concept nominal 3.50 kg |
#
# The 3.18 kg total matches the recorded whole-robot measurement. Its component
# distribution is still estimated. Matching total weight is not proof of COM
# or inertia accuracy. None of these profiles is a hardware calibration.
#
# We use **rigid_shoe** throughout the main manual so the endpoint and its
# mass come from a consistent source. This preserves the original hip/knee
# topology. The package default is the same variant.

# %%
robot = DrobotModel.from_repository(REPO, variant="rigid_shoe")
leg_name = "front_left"
leg = robot.leg(leg_name)
indices = robot.leg_indices(leg_name)

print("URDF:", robot.urdf_path)
print("Source SHA-256:", robot.source_sha256)
print("Number of actuated joints:", robot.n)
print("URDF mass (kg):", robot.total_mass)
print("Ordered joint names:", robot.joint_names)
print("\nLeg root (m):", leg.root)
print("Side s:", leg.side)
print("Hip offset a, b, c (m):", leg.a, leg.b, leg.c)
print("L1, L, sole extension e (m):",
      leg.proximal_length, leg.distal_length, leg.sole_extension)
print("Leg limits (degrees):", np.rad2deg(leg.lower), np.rad2deg(leg.upper))

# %% [markdown]
# ## 2. Implement the coordinate operations
#
# A rotation's columns are its child-frame axes expressed in the parent.
# A proper rotation satisfies \(R^TR=I\) and \(\det R=1\). A rigid transform is
#
# $$
# T=\begin{bmatrix}R&p\\0&1\end{bmatrix},
# \qquad T^{-1}=\begin{bmatrix}R^T&-R^Tp\\0&1\end{bmatrix}.
# $$
#
# The following small functions show the arithmetic used by the package.
# Translation is applied after rotation when T acts on a column point.

# %%
def R_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def R_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def T_from_R_p(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T

s = leg.side
R0 = R_z(s * np.pi / 2) @ R_x(np.pi / 2)
R1 = R_z(-np.pi / 2) @ R_x(-np.pi / 2)
print("Root orientation R0:\n", R0)
print("Fixed hip-to-flexion orientation R1:\n", R1)
print("Downward link direction at zero:\n", R0 @ R1 @ np.array([1., 0., 0.]))

# %% [markdown]
# ## 3. Build this leg's transform chain
#
# Use \(q_i=(\alpha,\beta,\kappa)\), side \(s=+1\) left and \(-1\) right,
# and \(L_2=L+e\) for the rigid shoe.
# The XML joint origin is followed by rotation about its local axis:
#
# $$
# \begin{aligned}
# T^B_H&=A(R_0,r_i)\,\mathcal R_z(\alpha),\\
# T^H_P&=A(R_1,(a,-b,c)^T)\,\mathcal R_z(-s\beta),\\
# T^P_D&=A(I,(L_1,0,0)^T)\,\mathcal R_z(-s\kappa),\\
# T^D_C&=A(I,(L_2,0,0)^T).
# \end{aligned}
# $$
#
# H is the abduction output, P the hip-flexion pivot, D the knee pivot,
# and C the sole center. The minus-side factor belongs to the pitch axes;
# it is not a motor calibration sign.

# %%
q_leg = np.deg2rad([10., 25., -55.])
q = np.zeros(robot.n)
q[indices] = q_leg
alpha, beta, kappa = q_leg
L1, L2 = leg.proximal_length, leg.distal_length + leg.sole_extension

T_B_H = T_from_R_p(R0, leg.root) @ transform(R_z(alpha))
T_H_P = T_from_R_p(R1, [leg.a, -leg.b, leg.c]) @ transform(R_z(-s * beta))
T_P_D = transform(translation=[L1, 0, 0]) @ transform(R_z(-s * kappa))
T_D_C = transform(translation=[L2, 0, 0])
T_B_C = T_B_H @ T_H_P @ T_P_D @ T_D_C

print("Body-to-hip:\n", T_B_H)
print("Body-to-knee:\n", T_B_H @ T_H_P @ T_P_D)
print("Body-to-sole:\n", T_B_C)

# %% [markdown]
# ## 4. Expand the position into a practical formula
#
# Zero pitch points down. Each pitch link contributes sine to forward reach
# and cosine to downward reach:
#
# $$
# F=L_1\sin\beta+L_2\sin(\beta+\kappa),\quad
# D=L_1\cos\beta+L_2\cos(\beta+\kappa),\quad H=b+D.
# $$
#
# Rotate the hip offset and pitch chain together around the abduction axis.
# For the URDF root position \(r_i=(r_x,r_y,r_z)\):
#
# $$
# p^B_C=\begin{bmatrix}
# r_x+sc+F\\
# r_y+s(a\cos\alpha+H\sin\alpha)\\
# r_z+a\sin\alpha-H\cos\alpha
# \end{bmatrix}.
# $$
#
# This is the position formula you can use in a lightweight controller.
# The full URDF tree in the package also supplies every other link and sensor
# frame, plus orientations.

# %%
def foot_position_from_math(q_local, geometry):
    alpha, beta, kappa = q_local
    s = geometry.side
    L1 = geometry.proximal_length
    L2 = geometry.distal_length + geometry.sole_extension
    F = L1 * np.sin(beta) + L2 * np.sin(beta + kappa)
    D = L1 * np.cos(beta) + L2 * np.cos(beta + kappa)
    H = geometry.b + D
    return geometry.root + np.array([
        s * geometry.c + F,
        s * (geometry.a * np.cos(alpha) + H * np.sin(alpha)),
        geometry.a * np.sin(alpha) - H * np.cos(alpha),
    ])

p = foot_position_from_math(q_leg, leg)
print("Sole center in B (m):", p)
print("Same point in millimeters:", p * 1000)
print("Reusable model's full foot pose:\n", robot.foot_pose(q, leg_name))

# %% [markdown]
# ### Inspect the chain geometrically
#
# These plots are generated from joint-axis points when you execute the cell.
# They are geometric projections, not collision or contact simulations.
# The line between the hip roots and pitch pivots includes the real offset.

# %%
def plot_leg(q_local, geometry, jacobian=None):
    points = geometry.points(q_local)
    labels = ["abduction", "hip flexion", "knee", "fork", "sole"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, component, xlabel in zip(axes, [0, 1], ["X forward (m)", "Y left (m)"]):
        ax.plot(points[:, component], points[:, 2], "o-", color="#146b55")
        for point, label in zip(points, labels):
            ax.annotate(label, (point[component], point[2]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        if jacobian is not None:
            for j, color in enumerate(["#ac6f24", "#347bb2", "#9553a1"]):
                arrow = .15 * jacobian[:, j]
                ax.quiver(points[-1, component], points[-1, 2], arrow[component], arrow[2],
                          angles="xy", scale_units="xy", scale=1, color=color,
                          label=["alpha", "beta", "kappa"][j])
            ax.legend()
        ax.set(xlabel=xlabel, ylabel="Z up (m)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig

plot_leg(q_leg, leg)
plt.show()

# %% [markdown]
# ## 5. Differentiate the position to obtain the Jacobian
#
# \(J_p=\partial p/\partial q_i\) has rows X/Y/Z and columns alpha/beta/knee.
# Differentiate in radians. The helper derivatives are
#
# $$
# F_\beta=D,\quad D_\beta=-F,\quad
# F_\kappa=L_2\cos(\beta+\kappa),\quad
# D_\kappa=-L_2\sin(\beta+\kappa).
# $$
#
# Substitution gives
#
# $$
# J_p=\begin{bmatrix}
# 0&D&L_2\cos\theta\\
# s(-a\sin\alpha+H\cos\alpha)&-sF\sin\alpha&-sL_2\sin\theta\sin\alpha\\
# a\cos\alpha+H\sin\alpha&F\cos\alpha&L_2\sin\theta\cos\alpha
# \end{bmatrix},\quad\theta=\beta+\kappa.
# $$
#
# Here SymPy performs that derivation symbolically and produces a NumPy
# function. This cell is part of building the model, not precomputed output.

# %%
al, be, ka = sp.symbols("alpha beta kappa", real=True)
aa, bb, cc, ell1, ell2 = sp.symbols("a b c L_1 L_2", real=True)
ss = sp.symbols("s", real=True)
r_x, r_y, r_z = sp.symbols("r_x r_y r_z", real=True)
F_symbolic = ell1 * sp.sin(be) + ell2 * sp.sin(be + ka)
D_symbolic = ell1 * sp.cos(be) + ell2 * sp.cos(be + ka)
H_symbolic = bb + D_symbolic
p_symbolic = sp.Matrix([
    r_x + ss * cc + F_symbolic,
    r_y + ss * (aa * sp.cos(al) + H_symbolic * sp.sin(al)),
    r_z + aa * sp.sin(al) - H_symbolic * sp.cos(al),
])
J_symbolic = p_symbolic.jacobian([al, be, ka])
display(J_symbolic)
J_function = sp.lambdify((al, be, ka, ss, aa, bb, ell1, ell2), J_symbolic, "numpy")
J = np.asarray(J_function(*q_leg, leg.side, leg.a, leg.b, L1, L2), dtype=float)
print("Numerical Jp (m/rad):\n", J)

# %% [markdown]
# ## 6. Use the Jacobian for velocity and conditioning
#
# The instantaneous relation is exact:
# \(\dot p=J_p\dot q_i\). For a finite step it is a local approximation:
# \(\Delta p\approx J_p\Delta q_i\).
#
# Each column is also \(z_j\times(p-o_j)\). This construction lets the
# general URDF model find Jacobians for any link point, including each COM.
# The angular column is \(z_j\), so the geometric Jacobian is
# \([J_\omega;J_p]\).
#
# A spatial twist uses a different lower component:
# \(v_s=\dot p-\omega\times p\). Accordingly,
# \(J_s=[J_\omega;J_p+[p]_\times J_\omega]\).
#
# The position determinant is
# \(\det J_p=-sHL_1L_2\sin\kappa\). A straight knee loses a position-velocity
# direction. Singular values show directional gains more clearly than
# the determinant alone.

# %%
qdot_leg = np.array([.1, .2, -.1])  # rad/s
print("Foot velocity (m/s):", J @ qdot_leg)
print("Geometric Jacobian [angular; point-linear]:\n", leg.geometric_jacobian(q_leg))
print("Space-twist Jacobian:\n", leg.space_jacobian(q_leg))
U, singular_values, Vt = np.linalg.svd(J)
print("Singular values (m/rad):", singular_values)
print("Weakest foot-velocity direction in B:", U[:, -1])
plot_leg(q_leg, leg, jacobian=J)
plt.show()

# %% [markdown]
# ## 7. Invert the geometry to find joint angles
#
# Subtract the root and convert Y to an outward coordinate:
#
# $$
# X=p_x-r_x-sc,\quad Y=s(p_y-r_y),\quad Z=p_z-r_z.
# $$
#
# The roll-plane equations imply \(Y^2+Z^2=a^2+H^2\), giving
#
# $$
# H=\pm\sqrt{Y^2+Z^2-a^2},\quad
# \alpha=\operatorname{atan2}(Z,Y)+\operatorname{atan2}(H,a).
# $$
#
# Set \(F=X,D=H-b\). The cosine law and triangle bearing give
#
# $$
# c_\kappa=\frac{F^2+D^2-L_1^2-L_2^2}{2L_1L_2},\quad
# \kappa=\operatorname{atan2}(\pm\sqrt{1-c_\kappa^2},c_\kappa),
# $$
# $$
# \beta=\operatorname{atan2}(F,D)-
# \operatorname{atan2}(L_2\sin\kappa,L_1+L_2\cos\kappa).
# $$
#
# The code below spells out the usual downward roll branch. The reusable
# solver enumerates both roll branches and both knee branches, rejects
# out-of-range targets, applies stored joint limits, and uses a seed to
# choose a nearby configuration.

# %%
def downward_ik_branch(target, geometry, knee_sign=-1):
    s = geometry.side
    X = target[0] - geometry.root[0] - s * geometry.c
    Y = s * (target[1] - geometry.root[1])
    Z = target[2] - geometry.root[2]
    radius_squared = Y**2 + Z**2 - geometry.a**2
    if radius_squared < 0:
        raise ValueError("target is inside the roll-offset exclusion cylinder")
    H = np.sqrt(radius_squared)
    alpha = np.arctan2(Z, Y) + np.arctan2(H, geometry.a)
    F, D = X, H - geometry.b
    l1, l2 = geometry.proximal_length, geometry.distal_length + geometry.sole_extension
    cosine_knee = (F**2 + D**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if not -1 <= cosine_knee <= 1:
        raise ValueError("target is outside the two-link workspace")
    kappa = np.arctan2(knee_sign * np.sqrt(1 - cosine_knee**2), cosine_knee)
    beta = np.arctan2(F, D) - np.arctan2(l2 * np.sin(kappa), l1 + l2 * np.cos(kappa))
    return np.array([alpha, beta, kappa])

target_B = p + np.array([.012, -.008, .006])  # move the endpoint by a few mm
print("One geometric branch (degrees):", np.rad2deg(downward_ik_branch(target_B, leg)))
candidates = leg.ik_candidates(target_B)
print("Branches within stored limits (degrees):", [np.rad2deg(c) for c in candidates])
q_target = leg.ik(target_B, seed=q_leg)
print("Selected branch (degrees):", np.rad2deg(q_target))

# %% [markdown]
# ### Numerical inverse kinematics
#
# Solve \(\min_{\Delta q}\|J\Delta q-e\|^2+\lambda^2\|\Delta q\|^2\), where
# \(e=p^*-p(q)\). The stationary condition gives
#
# $$
# \Delta q=J^T(JJ^T+\lambda^2I)^{-1}e.
# $$
#
# Use a linear solve, a step bound and backtracking. Damping does not make
# an unreachable target reachable. The result carries a success flag and
# residual; callers must not silently treat a failed iteration as solved.

# %%
numerical_result = leg.numerical_ik(target_B, seed=q_leg, damping=.01)
print(numerical_result)
if not numerical_result.success:
    print("No target command should be accepted from this numerical result.")

# %% [markdown]
# ## 8. Include body pose and all four legs
#
# A world target becomes a body target through
# \(p^B=R_{WB}^T(p^W-p_B^W)\). The tree model applies the same transformation
# to every link. It does not confuse a body point with a world point.
#
# The body frame is at the bottom center of the shell, not at its COM.
# Here we place the body 0.4 m up with a small pitch and yaw.

# %%
T_W_B = transform(rz(np.deg2rad(15)) @ ry(np.deg2rad(5)), [0., 0., .4])
feet_W = robot.foot_positions(q, base_pose=T_W_B)
for name, position in feet_W.items():
    print(name, position)

target_W = T_W_B[:3, :3] @ target_B + T_W_B[:3, 3]
q_world_target = robot.feet_ik({leg_name: target_W}, seed=q, base_pose=T_W_B)
print("Named updated joint configuration:")
display(dict(zip(robot.joint_names, q_world_target)))

# %% [markdown]
# ## 9. Create a joint trajectory from a Cartesian path
#
# A path provides targets; IK makes them joint configurations. Preserve the
# previous solution as the next seed to avoid jumping between branches.
# A quintic progress function has zero first and second derivatives at its
# ends:
#
# $$
# u=t/T,\qquad s(u)=10u^3-15u^4+6u^5.
# $$
#
# For this small swing example, use a smooth vertical clearance term
# \(64h\,u^3(1-u)^3\). This section computes geometric samples only.
# It does not establish collision clearance or dynamic gait feasibility.

# %%
times = np.linspace(0., 1., 51)
start = p.copy()
end = p + np.array([.025, 0., .005])
path, joint_path = [], []
previous = q_leg.copy()
for u in times:
    progress = 10*u**3 - 15*u**4 + 6*u**5
    target = (1-progress)*start + progress*end
    target = target + np.array([0., 0., 64*.015*u**3*(1-u)**3])
    previous = leg.ik(target, seed=previous)
    path.append(target)
    joint_path.append(previous.copy())
path, joint_path = np.array(path), np.array(joint_path)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(path[:, 0], path[:, 2])
axes[0].set(xlabel="Body X (m)", ylabel="Body Z (m)", title="Requested sole-center path")
axes[1].plot(times, np.rad2deg(joint_path))
axes[1].set(xlabel="Time (s)", ylabel="Joint angle (degrees)", title="IK joint trajectory")
axes[1].legend(["abduction", "hip flexion", "knee"])
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Convert external force to generalized effort
#
# Power equality gives \(F^T\dot p=F^TJ\dot q=\tau_{\rm ext}^T\dot q\).
# Therefore \(\tau_{\rm ext}=J^TF\).
#
# For a force ON the robot, static balance is \(g=\tau_{\rm act}+J^TF\).
# Consequently the balancing actuator torque is \(g-J^TF\), not simply
# \(+J^TF\). Add \(J_\omega^Tm\) if a moment acts about the selected point.
#
# The following 8 N is an illustrative upward point force. It is not a
# solved load distribution over the four feet.

# %%
endpoint_link, endpoint_local = robot.endpoints[leg_name]
external = ExternalWrench(endpoint_link, endpoint_local, force_world=(0., 0., 8.))
tau_external = robot.external_generalized_force(q, [external])
print("Joint generalized force from the external load (N m):", tau_external[indices])
print("Full selected profile effort limits (N m):", robot.effort_limits)

# %% [markdown]
# ## 11. Construct mass, inertia and gravity from the same transforms
#
# Each URDF inertial element gives \(m_\ell,c_\ell,I_\ell\). Its COM is
# \(p_\ell=o_\ell+R_\ell c_\ell\), and its world tensor about COM is
# \(I_\ell^W=R_\ell I_\ell R_\ell^T\).
#
# Use the axis Jacobian construction at each COM, with zeros for joints
# that are not ancestors of that link. Then
#
# $$
# \mathcal K=\frac12\sum_\ell
# \left(m_\ell\dot p_\ell^T\dot p_\ell+
# \omega_\ell^T I_\ell^W\omega_\ell\right)=\frac12\dot q^TM\dot q,
# $$
# $$
# M=\sum_\ell(m_\ell J_{v,\ell}^TJ_{v,\ell}+
# J_{\omega,\ell}^TI_\ell^WJ_{\omega,\ell}).
# $$
#
# With gravity vector \(g_W=(0,0,-9.81)\),
# \(\mathcal V=-\sum m_\ell g_W^Tp_\ell\) and
# \(g(q)=-\sum J_{v,\ell}^Tm_\ell g_W\).
# The next cell assembles these terms explicitly. Fixed bodies contribute
# no joint-coordinate kinetic energy, but contribute to floating dynamics.

# %%
terms = robot.com_terms(q)
selected_com = terms["front_left_proximal_link"]
display(selected_com)

M_from_energy = np.zeros((robot.n, robot.n))
g_from_energy = np.zeros(robot.n)
gravity_W = np.array([0., 0., -9.81])
for term in terms.values():
    m, Jv, Jw, Iw = term["mass"], term["Jv"], term["Jw"], term["inertia_world"]
    M_from_energy += m * Jv.T @ Jv + Jw.T @ Iw @ Jw
    g_from_energy -= Jv.T @ (m * gravity_W)

print("Selected leg mass-matrix block (kg m²):\n", M_from_energy[np.ix_(indices, indices)])
print("Selected leg gravity effort (N m):", g_from_energy[indices])
print("Whole robot COM in B (m):", robot.center_of_mass(q))
print("Potential energy, relative to world z=0 (J):", robot.potential_energy(q))

# %% [markdown]
# ## 12. Add velocity terms and solve rigid-body dynamics
#
# Euler–Lagrange gives
#
# $$
# M(q)\ddot q+c_v(q,\dot q)+g(q)+d(\dot q)=\tau+J^TF.
# $$
#
# In coordinate rates, one construction is
# \((c_v)_i=\sum_{j,k}\Gamma_{ijk}\dot q_j\dot q_k\), with
#
# $$
# \Gamma_{ijk}=\frac12\left(
# \frac{\partial M_{ij}}{\partial q_k}+
# \frac{\partial M_{ik}}{\partial q_j}-
# \frac{\partial M_{jk}}{\partial q_i}\right).
# $$
#
# The reusable model evaluates the same dynamics through recursive
# Newton–Euler, avoiding numerical differentiation of M:
#
# 1. Carry angular velocity and angular acceleration down the tree.
# 2. Carry origin acceleration, including tangential and centripetal terms.
# 3. Move acceleration to each COM and compute \(F=m(a_C-g_W)\).
# 4. Compute \(N_C=I^W\dot\omega+\omega\times I^W\omega\).
# 5. Shift each wrench to its link origin, add child wrenches backward,
#    and project moment onto the joint axis.
#
# For a child pivot at displacement r from its parent, the key equations are
#
# $$
# \omega_c=\omega_p+z\dot q,\quad
# \dot\omega_c=\dot\omega_p+z\ddot q+\omega_p\times(z\dot q),
# $$
# $$
# a_c=a_p+\dot\omega_p\times r+\omega_p\times(\omega_p\times r).
# $$
#
# The implementation is readable in
# [rigid_body.py](drobot_model/rigid_body.py), methods _motion and
# inverse_dynamics. Here is the per-link COM/wrench arithmetic in isolation:

# %%
def link_inertial_wrench(mass, com_local, inertia_local, R, omega, angular_accel,
                         origin_accel, gravity_world):
    r = R @ com_local
    I_world = R @ inertia_local @ R.T
    com_accel = origin_accel + np.cross(angular_accel, r) + np.cross(omega, np.cross(omega, r))
    force = mass * (com_accel - gravity_world)
    moment_at_com = I_world @ angular_accel + np.cross(omega, I_world @ omega)
    moment_at_origin = moment_at_com + np.cross(r, force)
    return force, moment_at_origin

qdot = np.zeros(robot.n)
qdot[indices] = [.1, .2, -.1]
qdd_desired = np.zeros(robot.n)
qdd_desired[indices] = [.2, -.3, .15]
tau_required = robot.inverse_dynamics(q, qdot, qdd_desired)
h = robot.bias_forces(q, qdot)  # Coriolis/centrifugal + gravity; friction off by default.
print("Velocity-dependent effort (N m):", (h - robot.gravity_forces(q))[indices])
print("Torque for chosen acceleration (N m):", tau_required[indices])

# A model-use question: what acceleration results at 80% of that effort?
qdd_at_reduced_effort = robot.forward_dynamics(q, qdot, .8 * tau_required)
print("Acceleration at reduced effort (rad/s²):", qdd_at_reduced_effort[indices])

# %% [markdown]
# Joint friction can be included with friction=True. It uses the source
# damping and Coulomb coefficients; sign(0)=0 is not a stiction model.
# No torque-speed, thermal, latency, backlash or geared rotor model is
# silently added. Effort limits are exposed, not automatically enforced.
#
# ## 13. Let the base translate and rotate
#
# Store the configuration as \((p_B,R_B,q)\), and the generalized speed as
# \(\nu=[v_B^W,\omega_B^W,\dot q]\). The original topology has 18 speeds.
# Rotation obeys \(\dot R_B=[\omega_B^W]_\times R_B\); angular velocity is
# not an Euler-angle derivative.
#
# For an endpoint \(p_i^W=p_B^W+R_B^Wp_i^B\):
#
# $$
# \dot p_i^W=\underbrace{
# [I,\ -[R_B^Wp_i^B]_\times,\ R_B^WJ_i^BE_i]}_{\mathcal J_i}\nu.
# $$
#
# E_i selects that leg's joints. The full mass matrix uses COM Jacobians
# with these extra six columns. Newton–Euler propagates world angular
# velocity directly, so it does not need an Euler-angle chart.

# %%
q_stance, base_stance = robot.stance(fore_aft_m=.080)
nu = np.zeros(robot.n + 6)
M_floating = robot.mass_matrix(q_stance, base_pose=base_stance, floating=True)
J_foot_full = robot.foot_jacobian(q_stance, leg_name, base_pose=base_stance, floating=True)
print("Level base height for sole-center contact (m):", base_stance[2, 3])
print("Floating mass matrix shape:", M_floating.shape)
print("Base translational mass block:\n", M_floating[:3, :3])
print("Foot Jacobian shape:", J_foot_full.shape)
print("Foot Jacobian base columns [translation, rotation]:\n", J_foot_full[:, :6])
print("World COM (m):", robot.center_of_mass(q_stance, base_pose=base_stance))

# %% [markdown]
# ## 14. Solve an explicit ground-contact mode
#
# A floating robot has no direct base actuators:
#
# $$
# \mathcal M\dot\nu+h=S^T\tau+\mathcal J_c^T\lambda,
# \qquad S=[0_{n\times6},I_n].
# $$
#
# For stationary sticking point contacts,
# \(\mathcal J_c\nu=0\) and
# \(\mathcal J_c\dot\nu+\dot{\mathcal J}_c\nu=0\). Solve:
#
# $$
# \begin{bmatrix}\mathcal M&-\mathcal J_c^T\\
# \mathcal J_c&0\end{bmatrix}
# \begin{bmatrix}\dot\nu\\\lambda\end{bmatrix}
# =
# \begin{bmatrix}S^T\tau-h\\-\dot{\mathcal J}_c\nu\end{bmatrix}.
# $$
#
# The package computes \(\dot{\mathcal J}_c\nu\) from its acceleration
# recursion with generalized acceleration set to zero.
#
# First find a static balance: solve the unactuated six equations for
# minimum-norm contact forces, then compute joint torque. This is an
# equality solution, not a friction-constrained optimizer. Its force
# diagnostics expose negative normal force or exceeded friction.
#
# **Assumption for this example:** one point at each rigid sole center on a
# level plane, and illustrative friction coefficient 0.6. The geometry
# places these centers on the plane. No terrain discovery, collision
# detection, finite sole tipping moments, contact switching or impacts
# are solved. The bare-fork variant deliberately rejects sole_contacts.

# %%
contacts = robot.sole_contacts(friction_coefficient=.6)
balance = robot.static_equilibrium(q_stance, contacts, base_pose=base_stance)
print("Static point forces ON robot (N), FL/FR/RL/RR:\n", balance.forces_world)
print("Static joint torque (N m):", balance.torque)
print("Normal/friction force feasibility:", balance.force_feasible)
print("Effort headroom against selected URDF limits (N m):",
      robot.effort_limits - np.abs(balance.torque))

# Perturb one actuator torque, keeping the same instantaneous sticking mode.
torque_command = balance.torque.copy()
torque_command[robot.index["front_left_hip_flexion"]] += .025
contact_response = robot.constrained_forward_dynamics(
    q_stance, nu, torque_command, contacts, base_pose=base_stance
)
print("Predicted base linear acceleration (m/s²):", contact_response.acceleration[:3])
print("Predicted base angular acceleration (rad/s²):", contact_response.acceleration[3:6])
print("Predicted contact forces (N):\n", contact_response.forces_world)
print("Force feasibility in the assumed mode:", contact_response.force_feasible)

# %% [markdown]
# If a required normal force is negative, that contact cannot remain active.
# If friction is exceeded, sticking is not admissible. Change modes rather
# than accepting the equality solution as physical. Redundant contact
# rows and incompatible sticking velocities raise explicit errors.
#
# The notebook does not time-integrate a walking simulation. It provides
# instantaneous kinematic and dynamic operators you can use in planning,
# analysis or a separate simulator. A contact-aware integrator is additional
# work; a naive Euler loop is not a replacement for impact and mode handling.
#
# ## 15. Use the finished model outside the notebook
#
# The preceding calculations correspond to the reusable NumPy package,
# whose only runtime dependency is NumPy. SymPy, plotting and Jupyter are
# notebook tools, not required to call FK, IK or dynamics.
#
# This function consumes a body pose, current configuration and a world
# target; it returns model data. It sends no servo commands.

# %%
def plan_one_foot(model, q_now, body_pose, leg_name, target_world):
    q_next = model.feet_ik({leg_name: target_world}, seed=q_now, base_pose=body_pose)
    return {
        "joint_names": model.joint_names,
        "q_rad": q_next,
        "foot_pose_world": model.foot_pose(q_next, leg_name, base_pose=body_pose),
        "foot_jacobian_world": model.foot_jacobian(q_next, leg_name, base_pose=body_pose),
        "mass_matrix_fixed_base": model.mass_matrix(q_next, base_pose=body_pose),
        "gravity_effort_fixed_base": model.gravity_forces(q_next, base_pose=body_pose),
    }

current_feet = robot.foot_positions(q_stance, base_pose=base_stance)
swing_target = current_feet["front_left"] + np.array([.010, 0., .020])
plan = plan_one_foot(robot, q_stance, base_stance, "front_left", swing_target)
print("Planned FL joint angles (degrees):", np.rad2deg(plan["q_rad"][robot.leg_indices("front_left")]))
print("Planned world foot pose:\n", plan["foot_pose_world"])

# %% [markdown]
# To use the package in another project, install this directory as shown in
# section 0 and import DrobotModel. Pass a repository root, or construct the
# class directly from a supported URDF path.
#
# | Operation | Public method |
# |---|---|
# | All link transforms | kinematics |
# | Named foot pose / position | foot_pose / foot_positions |
# | Any local point Jacobian | point_jacobian / geometric_jacobian |
# | Analytical leg inverse | leg(name).ik / ik_candidates |
# | Numerical position inverse | leg(name).numerical_ik |
# | Named world foot targets | feet_ik |
# | COM and energy inputs | center_of_mass / com_terms |
# | Mass and gravity | mass_matrix / gravity_forces |
# | Inverse dynamics | inverse_dynamics |
# | Forward dynamics | forward_dynamics |
# | Static point balance | static_equilibrium |
# | Prescribed sticking mode | constrained_forward_dynamics |
#
# For hardware, add the existing logical-angle and calibrated-encoder
# mapping. URDF angles and generalized effort are model coordinates;
# the ST3215's internal controller does not directly expose every modeled
# torque input. Refer to the hardware calibration layer before deployment.
#
# ## 16. Extend the model with an active ankle
#
# Add \(\delta\), with \(\gamma=\beta+\kappa+\delta\).
#
# $$
# F_4=L_1\sin\beta+L\sin(\beta+\kappa)+e\sin\gamma,\quad
# D_4=L_1\cos\beta+L\cos(\beta+\kappa)+e\cos\gamma.
# $$
#
# The abduction formula is unchanged after replacing F,D. Each pitch
# Jacobian column differentiates only its downstream terms. The fourth
# angular column is the same pitch axis; the scalar pitch row is [0,1,1,1].
#
# For a chosen sole pitch, subtract the rotated foot offset:
# \(F_A=F-e\sin\gamma,\ D_A=D-e\cos\gamma\), solve the two-link triangle,
# then set \(\delta=\gamma-\beta-\kappa\).
#
# Pitch here is within the rolled leg plane, not arbitrary world RPY.
# Four joints still do not provide a six-dimensional independently
# controllable foot pose. The module requires a pitch for analytical ankle
# IK; numerical_ik can solve a position-only redundant task from a seed.
#
# The active URDF includes the added motor and moving foot inertia, so the
# dynamics automatically becomes 22-dimensional for the floating robot.

# %%
ankle_robot = DrobotModel.from_repository(REPO, variant="active_ankle")
ankle_leg = ankle_robot.leg("front_left")
ankle_angles = np.deg2rad([0., 25., -50., 25.])
ankle_target = ankle_leg.fk(ankle_angles)[:3, 3]
ankle_solution = ankle_leg.ik(ankle_target, seed=ankle_angles, sole_pitch=0.)
J_position_ankle = ankle_leg.jacobian(ankle_solution)
J_position_pitch = np.vstack([J_position_ankle, [0., 1., 1., 1.]])
print("Ankle configuration (degrees):", np.rad2deg(ankle_solution))
print("Position + pitch Jacobian:\n", J_position_pitch)
print("Active model mass (kg):", ankle_robot.total_mass)
print("Floating velocity dimension:", ankle_robot.n + 6)

# %% [markdown]
# ## Model boundaries and source files
#
# This is an implemented mathematical model, not an identified digital twin.
# Mass distribution, inertias, friction and actuator behavior still need
# physical characterization. Explicit contact modes are assumed, not
# discovered. IK limits do not include collisions. Static force feasibility
# does not prove stability or motor thermal feasibility.
#
# Original geometric authority:
# [original URDF](../../cad/exports/urdf/quadruped_robot.urdf) and
# [CAD joint ledger](../../cad/specs/quadruped-urdf-ledger.md).
# The notebook default uses the
# [12-joint rigid-shoe baseline](../../cad/exports/urdf/quadruped_robot_v2_ankle_baseline.urdf).
# The ankle extension uses the
# [active concept](../../cad/exports/urdf/quadruped_robot_v2_ankle_active.urdf).
#
# The repository
# [planar helper](../isaac/_quadruped_runtime.py) supplies the existing stance
# context; it is not imported by this implementation. No Isaac or hardware
# module is loaded.
#
# Mathematical conventions:
# [Modern Robotics: space Jacobian](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-1-1-space-jacobian/),
# [Lagrangian dynamics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/chapter-8-1-lagrangian-formulation-of-dynamics-part-1-of-2/),
# [constrained dynamics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/8-7-constrained-dynamics/).
#
# Source code:
# [transforms](drobot_model/transforms.py),
# [closed-form leg model](drobot_model/leg.py),
# [URDF and dynamics](drobot_model/rigid_body.py),
# [Drobot adapter](drobot_model/robot.py).
#
# The .ipynb and this readable cell source describe the same manual.
# Neither notebook cells nor the model package were executed during creation.
