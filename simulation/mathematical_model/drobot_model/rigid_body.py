"""URDF tree kinematics and readable world-frame rigid-body dynamics.

No simulator, mesh loading, collision engine, controller or hardware imports.
Only fixed and revolute/continuous joints are supported. Floating generalized
speeds are [v_base_world, omega_base_world, qdot]. All external wrenches act
ON the robot. Base acceleration is the world derivative of base velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

from .transforms import pose_matrix, rpy_rotation, skew, so3_exp, transform, vector


@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray
    inertia: np.ndarray  # About COM, expressed in link axes.


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    speed: float
    damping: float
    friction: float


@dataclass
class KinematicState:
    poses: dict[str, np.ndarray]
    axes: dict[str, np.ndarray]
    pivots: dict[str, np.ndarray]


@dataclass
class Motion:
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray


@dataclass
class PointContact:
    link: str
    point: tuple[float, float, float] | np.ndarray
    normal_world: tuple[float, float, float] = (0., 0., 1.)
    friction_coefficient: float | None = None


@dataclass
class ExternalWrench:
    link: str
    point: tuple[float, float, float] | np.ndarray
    force_world: tuple[float, float, float] | np.ndarray
    moment_world: tuple[float, float, float] | np.ndarray = (0., 0., 0.)


@dataclass
class ContactSolution:
    acceleration: np.ndarray
    forces_world: np.ndarray
    torque: np.ndarray
    normal_forces: np.ndarray
    friction_margins: np.ndarray  # NaN when coefficient is unspecified.
    force_feasible: bool | None  # None means friction is unspecified.
    equilibrium_residual: float


def _xyz(text, default=(0., 0., 0.)):
    return vector(default if text is None else [float(v) for v in text.split()], 3)


def _origin(node):
    if node is None:
        return np.eye(4)
    return transform(rpy_rotation(_xyz(node.get("rpy"))), _xyz(node.get("xyz")))


class URDFRobot:
    def __init__(self, urdf_path, *, joint_order=None):
        self.urdf_path = Path(urdf_path).resolve()
        source = self.urdf_path.read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        root = ET.fromstring(source)
        if root.tag != "robot":
            raise ValueError("expected a URDF robot element")
        self.name = root.get("name", self.urdf_path.stem)
        self.links = {}
        for node in root.findall("link"):
            name = node.attrib["name"]
            if name in self.links:
                raise ValueError(f"duplicate link {name}")
            inertial = node.find("inertial")
            if inertial is None:
                mass, com, inertia = 0., np.zeros(3), np.zeros((3, 3))
            else:
                mass = float(inertial.find("mass").attrib["value"])
                T = _origin(inertial.find("origin"))
                values = inertial.find("inertia").attrib
                inertia = np.array([[float(values["i" + x + y] if "i" + x + y in values
                                                 else values["i" + y + x])
                                     for y in "xyz"] for x in "xyz"])
                com, R = T[:3, 3], T[:3, :3]
                inertia = R @ inertia @ R.T
                if not np.isfinite(mass) or mass < 0 or not np.all(np.isfinite(inertia)):
                    raise ValueError(f"invalid inertial data on {name}")
            self.links[name] = Link(name, mass, com, inertia)
        by_name, child_joints, children = {}, {}, {name: [] for name in self.links}
        for node in root.findall("joint"):
            name, kind = node.attrib["name"], node.attrib["type"]
            if kind not in {"fixed", "revolute", "continuous"} or node.find("mimic") is not None:
                raise ValueError(f"{name}: only independent fixed/revolute joints are supported")
            parent, child = node.find("parent").attrib["link"], node.find("child").attrib["link"]
            if name in by_name or child in child_joints:
                raise ValueError(f"duplicate joint or multiply parented link at {name}")
            if parent not in self.links or child not in self.links:
                raise ValueError(f"{name} refers to a missing link")
            axis_node = node.find("axis")
            axis = _xyz(None if axis_node is None else axis_node.get("xyz"), (1, 0, 0))
            if np.linalg.norm(axis) <= 1e-12:
                raise ValueError(f"{name} has a zero axis")
            axis = axis / np.linalg.norm(axis)
            limit = node.find("limit")
            lim = {} if limit is None else limit.attrib
            dyn_node = node.find("dynamics")
            dyn = {} if dyn_node is None else dyn_node.attrib
            lower = float(lim.get("lower", "-inf")) if kind == "revolute" else -np.inf
            upper = float(lim.get("upper", "inf")) if kind == "revolute" else np.inf
            j = Joint(name, parent, child, kind, _origin(node.find("origin")), axis,
                      lower, upper, float(lim.get("effort", "inf")),
                      float(lim.get("velocity", "inf")), float(dyn.get("damping", 0)),
                      float(dyn.get("friction", 0)))
            by_name[name], child_joints[child] = j, j
            children[parent].append(j)
        roots = set(self.links) - set(child_joints)
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.root_link = next(iter(roots))
        self.joints = []
        visited = {self.root_link}
        def visit(parent):
            for joint in children[parent]:
                if joint.child in visited:
                    raise ValueError("cycle in URDF")
                visited.add(joint.child)
                self.joints.append(joint)
                visit(joint.child)
        visit(self.root_link)
        if visited != set(self.links):
            raise ValueError("disconnected URDF tree")
        active = [j.name for j in self.joints if j.kind != "fixed"]
        names = active if joint_order is None else list(joint_order)
        if len(names) != len(set(names)) or set(names) != set(active):
            raise ValueError("joint_order must contain every movable joint exactly once")
        self.joint_names = tuple(names)
        self.index = {name: i for i, name in enumerate(names)}
        self.joint_by_name = by_name
        self.n = len(names)
        self.lower = np.array([by_name[n].lower for n in names])
        self.upper = np.array([by_name[n].upper for n in names])
        self.effort_limits = np.array([by_name[n].effort for n in names])
        self.speed_limits = np.array([by_name[n].speed for n in names])
        self.damping = np.array([by_name[n].damping for n in names])
        self.coulomb_friction = np.array([by_name[n].friction for n in names])
        self.ancestors = {self.root_link: ()}
        for j in self.joints:
            own = () if j.kind == "fixed" else (j.name,)
            self.ancestors[j.child] = self.ancestors[j.parent] + own

    @property
    def total_mass(self):
        return sum(link.mass for link in self.links.values())

    def configuration(self, values: dict[str, float]) -> np.ndarray:
        if set(values) != set(self.joint_names):
            raise ValueError("configuration dictionary must specify every joint name exactly once")
        return vector([values[name] for name in self.joint_names], self.n, "q")

    def kinematics(self, q, *, base_pose=None) -> KinematicState:
        q = vector(q, self.n, "q")
        poses, axes, pivots = {self.root_link: pose_matrix(base_pose)}, {}, {}
        for j in self.joints:
            before = poses[j.parent] @ j.origin
            if j.kind == "fixed":
                poses[j.child] = before
            else:
                axes[j.name] = before[:3, :3] @ j.axis
                pivots[j.name] = before[:3, 3]
                poses[j.child] = before @ transform(so3_exp(j.axis * q[self.index[j.name]]))
        return KinematicState(poses, axes, pivots)

    def _point_jacobians(self, state, link, point, floating):
        T = state.poses[link]
        p = T[:3, 3] + T[:3, :3] @ vector(point, 3, "local point")
        offset = 6 if floating else 0
        Jv, Jw = np.zeros((3, self.n + offset)), np.zeros((3, self.n + offset))
        if floating:
            r = p - state.poses[self.root_link][:3, 3]
            Jv[:, :3], Jv[:, 3:6], Jw[:, 3:6] = np.eye(3), -skew(r), np.eye(3)
        for name in self.ancestors[link]:
            column = offset + self.index[name]
            Jv[:, column] = np.cross(state.axes[name], p - state.pivots[name])
            Jw[:, column] = state.axes[name]
        return p, Jv, Jw

    def point_pose(self, q, link, point=(0., 0., 0.), *, base_pose=None):
        T = self.kinematics(q, base_pose=base_pose).poses[link]
        return T @ transform(translation=point)

    def point_jacobian(self, q, link, point=(0., 0., 0.), *, base_pose=None, floating=False):
        state = self.kinematics(q, base_pose=base_pose)
        return self._point_jacobians(state, link, point, floating)[1]

    def geometric_jacobian(self, q, link, point=(0., 0., 0.), *, base_pose=None, floating=False):
        state = self.kinematics(q, base_pose=base_pose)
        _, Jv, Jw = self._point_jacobians(state, link, point, floating)
        return np.vstack([Jw, Jv])

    def center_of_mass(self, q, *, base_pose=None):
        if self.total_mass <= 0:
            raise ValueError("robot has no positive mass")
        state = self.kinematics(q, base_pose=base_pose)
        return sum(link.mass * (state.poses[name][:3, 3] + state.poses[name][:3, :3] @ link.com)
                   for name, link in self.links.items()) / self.total_mass

    def com_terms(self, q, *, base_pose=None, floating=False):
        """Mass, COM, world inertia and Jacobians: inspectable energy inputs."""
        state = self.kinematics(q, base_pose=base_pose)
        result = {}
        for name, link in self.links.items():
            if link.mass == 0:
                continue
            p, Jv, Jw = self._point_jacobians(state, name, link.com, floating)
            R = state.poses[name][:3, :3]
            result[name] = dict(mass=link.mass, position=p, Jv=Jv, Jw=Jw,
                                inertia_world=R @ link.inertia @ R.T)
        return result

    def mass_matrix(self, q, *, base_pose=None, floating=False):
        size = self.n + (6 if floating else 0)
        M = np.zeros((size, size))
        for term in self.com_terms(q, base_pose=base_pose, floating=floating).values():
            Jv, Jw = term["Jv"], term["Jw"]
            M += term["mass"] * (Jv.T @ Jv) + Jw.T @ term["inertia_world"] @ Jw
        return M

    def potential_energy(self, q, *, base_pose=None, gravity=(0., 0., -9.81)):
        gravity = vector(gravity, 3, "world gravity")
        return -sum(t["mass"] * (gravity @ t["position"])
                    for t in self.com_terms(q, base_pose=base_pose).values())

    def gravity_forces(self, q, *, base_pose=None, floating=False, gravity=(0., 0., -9.81)):
        gravity = vector(gravity, 3, "world gravity")
        g = np.zeros(self.n + (6 if floating else 0))
        for t in self.com_terms(q, base_pose=base_pose, floating=floating).values():
            g -= t["Jv"].T @ (t["mass"] * gravity)
        return g

    def _motion(self, q, velocity, acceleration, base_pose, floating):
        state = self.kinematics(q, base_pose=base_pose)
        offset = 6 if floating else 0
        velocity = vector(velocity, self.n + offset, "generalized velocity")
        acceleration = vector(acceleration, self.n + offset, "generalized acceleration")
        z = np.zeros(3)
        motions = {self.root_link: Motion(velocity[:3] if floating else z.copy(),
            velocity[3:6] if floating else z.copy(), acceleration[:3] if floating else z.copy(),
            acceleration[3:6] if floating else z.copy())}
        for j in self.joints:
            parent = motions[j.parent]
            r = state.poses[j.child][:3, 3] - state.poses[j.parent][:3, 3]
            relative_w, relative_a = z.copy(), z.copy()
            if j.kind != "fixed":
                i = offset + self.index[j.name]
                relative_w = state.axes[j.name] * velocity[i]
                relative_a = state.axes[j.name] * acceleration[i]
            w = parent.angular_velocity
            alpha = parent.angular_acceleration
            motions[j.child] = Motion(
                parent.linear_velocity + np.cross(w, r),
                w + relative_w,
                parent.linear_acceleration + np.cross(alpha, r) + np.cross(w, np.cross(w, r)),
                alpha + relative_a + np.cross(w, relative_w))
        return state, motions

    def external_generalized_force(self, q, wrenches, *, base_pose=None, floating=False):
        state = self.kinematics(q, base_pose=base_pose)
        result = np.zeros(self.n + (6 if floating else 0))
        for wrench in wrenches:
            _, Jv, Jw = self._point_jacobians(state, wrench.link, wrench.point, floating)
            result += Jv.T @ vector(wrench.force_world, 3, "external force")
            result += Jw.T @ vector(wrench.moment_world, 3, "external moment about point")
        return result

    def inverse_dynamics(self, q, velocity, acceleration, *, base_pose=None,
                         floating=False, gravity=(0., 0., -9.81), friction=False,
                         external_wrenches=()):
        """Required generalized effort by world-frame recursive Newton–Euler.

        Fixed base: n motor efforts. Floating: [required base force, required
        base moment, n motor efforts]. A physical floating robot can realize
        a requested motion only when contact/external forces account for the
        first six entries. They are NOT six extra motor commands.
        """
        offset = 6 if floating else 0
        velocity = vector(velocity, self.n + offset, "velocity")
        gravity = vector(gravity, 3, "world gravity")
        state, motions = self._motion(q, velocity, acceleration, base_pose, floating)
        forces, moments = {}, {}
        for name, link in self.links.items():
            R, motion = state.poses[name][:3, :3], motions[name]
            r = R @ link.com
            ac = (motion.linear_acceleration + np.cross(motion.angular_acceleration, r)
                  + np.cross(motion.angular_velocity, np.cross(motion.angular_velocity, r)))
            I = R @ link.inertia @ R.T
            forces[name] = link.mass * (ac - gravity)
            moments[name] = (I @ motion.angular_acceleration
                            + np.cross(motion.angular_velocity, I @ motion.angular_velocity)
                            + np.cross(r, forces[name]))
        effort = np.zeros(self.n + offset)
        for j in reversed(self.joints):
            if j.kind != "fixed":
                effort[offset + self.index[j.name]] = state.axes[j.name] @ moments[j.child]
            r = state.poses[j.child][:3, 3] - state.poses[j.parent][:3, 3]
            forces[j.parent] += forces[j.child]
            moments[j.parent] += moments[j.child] + np.cross(r, forces[j.child])
        if floating:
            effort[:3], effort[3:6] = forces[self.root_link], moments[self.root_link]
        if friction:
            qdot = velocity[offset:]
            effort[offset:] += self.damping * qdot + self.coulomb_friction * np.sign(qdot)
        effort -= self.external_generalized_force(q, external_wrenches,
                                                  base_pose=base_pose, floating=floating)
        return effort

    def bias_forces(self, q, velocity, *, base_pose=None, floating=False,
                    gravity=(0., 0., -9.81), friction=False):
        return self.inverse_dynamics(q, velocity, np.zeros(self.n + (6 if floating else 0)),
                                    base_pose=base_pose, floating=floating, gravity=gravity,
                                    friction=friction)

    def forward_dynamics(self, q, velocity, torque, *, base_pose=None, floating=False,
                         gravity=(0., 0., -9.81), friction=False, external_wrenches=()):
        tau = vector(torque, self.n, "actuator torque")
        actuation = np.r_[np.zeros(6), tau] if floating else tau
        M = self.mass_matrix(q, base_pose=base_pose, floating=floating)
        h = self.bias_forces(q, velocity, base_pose=base_pose, floating=floating,
                             gravity=gravity, friction=friction)
        ext = self.external_generalized_force(q, external_wrenches,
                                              base_pose=base_pose, floating=floating)
        return np.linalg.solve(M, actuation + ext - h)

    def contact_jacobian(self, q, contacts, *, base_pose=None):
        state = self.kinematics(q, base_pose=base_pose)
        if not contacts:
            return np.zeros((0, self.n + 6))
        return np.vstack([self._point_jacobians(state, c.link, c.point, True)[1] for c in contacts])

    def contact_acceleration_bias(self, q, velocity, contacts, *, base_pose=None):
        """Compute Jdot*nu analytically via zero-generalized-acceleration recursion."""
        state, motions = self._motion(q, velocity, np.zeros(self.n + 6), base_pose, True)
        values = []
        for c in contacts:
            motion = motions[c.link]
            r = state.poses[c.link][:3, :3] @ vector(c.point, 3)
            values.append(motion.linear_acceleration + np.cross(motion.angular_acceleration, r)
                          + np.cross(motion.angular_velocity, np.cross(motion.angular_velocity, r)))
        return np.concatenate(values) if values else np.zeros(0)

    @staticmethod
    def _contact_result(acceleration, forces, torque, contacts, residual):
        normal_forces, margins = [], []
        for c, force in zip(contacts, forces):
            normal = vector(c.normal_world, 3, "contact normal")
            length = np.linalg.norm(normal)
            if length < 1e-12:
                raise ValueError("contact normal must be nonzero")
            normal = normal / length
            fn = float(force @ normal)
            normal_forces.append(fn)
            mu = c.friction_coefficient
            if mu is not None and (not np.isfinite(mu) or mu < 0):
                raise ValueError("friction coefficient must be finite and nonnegative")
            margins.append(np.nan if mu is None else mu * fn - np.linalg.norm(force - fn * normal))
        normals, margins = np.array(normal_forces), np.array(margins)
        feasible = bool(np.all(normals >= -1e-8) and np.all(margins[np.isfinite(margins)] >= -1e-8))
        if feasible and np.any(np.isnan(margins)):
            feasible = None
        return ContactSolution(acceleration, forces, torque, normals, margins, feasible, float(residual))

    def static_equilibrium(self, q, contacts, *, base_pose=None, gravity=(0., 0., -9.81)):
        """Minimum-norm contact-force balance, NOT a friction-constrained optimizer."""
        contacts = tuple(contacts)
        if not contacts:
            raise ValueError("static floating equilibrium requires contacts")
        J = self.contact_jacobian(q, contacts, base_pose=base_pose)
        g = self.gravity_forces(q, base_pose=base_pose, floating=True, gravity=gravity)
        force, *_ = np.linalg.lstsq(J[:, :6].T, g[:6], rcond=None)
        residual = np.linalg.norm(J[:, :6].T @ force - g[:6])
        if residual > 1e-7:
            raise ValueError("these contacts cannot balance the floating base wrench")
        torque = g[6:] - J[:, 6:].T @ force
        return self._contact_result(np.zeros(self.n + 6), force.reshape(-1, 3), torque,
                                    contacts, residual)

    def constrained_forward_dynamics(self, q, velocity, torque, contacts, *,
                                     base_pose=None, gravity=(0., 0., -9.81), friction=False):
        """Solve a supplied independent stationary sticking point-contact mode.

        Reports normal/friction feasibility; does not choose contact modes,
        enforce terrain gaps, model impulses, or clamp torque to motor limits.
        """
        contacts = tuple(contacts)
        nu, tau = vector(velocity, self.n + 6, "velocity"), vector(torque, self.n, "torque")
        if not contacts:
            acceleration = self.forward_dynamics(q, nu, tau, base_pose=base_pose,
                floating=True, gravity=gravity, friction=friction)
            return self._contact_result(acceleration, np.zeros((0, 3)), tau, (), 0.)
        M = self.mass_matrix(q, base_pose=base_pose, floating=True)
        h = self.bias_forces(q, nu, base_pose=base_pose, floating=True, gravity=gravity, friction=friction)
        J = self.contact_jacobian(q, contacts, base_pose=base_pose)
        if np.linalg.matrix_rank(J) < J.shape[0]:
            raise ValueError("redundant contact constraints; choose an independent contact mode")
        if np.linalg.norm(J @ nu, ord=np.inf) > 1e-7:
            raise ValueError("velocity is incompatible with stationary sticking contacts; use an impact/mode model")
        bias = self.contact_acceleration_bias(q, nu, contacts, base_pose=base_pose)
        k = J.shape[0]
        K = np.block([[M, -J.T], [J, np.zeros((k, k))]])
        rhs = np.r_[np.r_[np.zeros(6), tau] - h, -bias]
        try:
            answer = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError("singular contact system; remove redundant constraints or choose an independent mode") from error
        acceleration, force = answer[:self.n + 6], answer[self.n + 6:]
        return self._contact_result(acceleration, force.reshape(-1, 3), tau, contacts,
                                    np.linalg.norm(M @ acceleration + h - np.r_[np.zeros(6), tau] - J.T @ force))
