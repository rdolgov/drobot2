"""Closed-form original and ankle leg equations, with explicit URDF signs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .transforms import adjoint, inverse_transform, rx, rz, skew, transform, vector


class IKError(ValueError):
    """The requested task has no admissible analytical solution."""


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    residual_m: float
    iterations: int
    message: str


@dataclass
class LegModel:
    """Angles are [abduction, hip flexion, knee, (ankle)] in radians.

    Geometry is supplied by DrobotModel.leg() from the selected URDF.
    Inertias are deliberately not part of this kinematic object.
    """
    side: int
    root: np.ndarray
    a: float
    b: float
    c: float
    proximal_length: float
    distal_length: float
    sole_extension: float
    lower: np.ndarray
    upper: np.ndarray
    active_ankle: bool = False

    def __post_init__(self):
        self.root = vector(self.root, 3, "leg root").copy()
        self.lower = vector(self.lower, self.dof, "lower limits").copy()
        self.upper = vector(self.upper, self.dof, "upper limits").copy()
        scalars = [self.a, self.b, self.c, self.proximal_length,
                   self.distal_length, self.sole_extension]
        if (self.side not in (-1, 1) or not np.all(np.isfinite(scalars))
                or min(self.proximal_length, self.distal_length) <= 0
                or self.sole_extension < 0 or np.any(self.lower >= self.upper)):
            raise ValueError("invalid signed leg geometry or joint limits")

    @property
    def dof(self) -> int:
        return 4 if self.active_ankle else 3

    def _terms(self, q):
        q = vector(q, self.dof, "leg q")
        if self.active_ankle:
            lengths = np.array([self.proximal_length, self.distal_length, self.sole_extension])
            angles = np.cumsum(q[1:])
        else:
            lengths = np.array([self.proximal_length, self.distal_length + self.sole_extension])
            angles = np.cumsum(q[1:])
        F = float(lengths @ np.sin(angles))
        D = float(lengths @ np.cos(angles))
        return q, lengths, angles, F, D

    def fk(self, q) -> np.ndarray:
        q, _, angles, F, D = self._terms(q)
        alpha, theta, s = q[0], angles[-1], self.side
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        H = self.b + D
        p = self.root + [s * self.c + F,
                        s * (self.a * ca + H * sa), self.a * sa - H * ca]
        R = np.array([[st, -s * ct, 0],
                      [s * sa * ct, sa * st, s * ca],
                      [-ca * ct, -s * ca * st, sa]])
        return transform(R, p)

    def transform_chain(self, q) -> tuple[np.ndarray, ...]:
        """Individual transforms: root→hip→proximal→distal→(ankle)→endpoint."""
        q = vector(q, self.dof, "leg q")
        s = self.side
        R0 = rz(s * np.pi / 2) @ rx(np.pi / 2)
        R1 = rz(-np.pi / 2) @ rx(-np.pi / 2)
        parts = [
            transform(R0, self.root) @ transform(rz(q[0])),
            transform(R1, [self.a, -self.b, self.c]) @ transform(rz(-s * q[1])),
            transform(translation=[self.proximal_length, 0, 0]) @ transform(rz(-s * q[2])),
        ]
        if self.active_ankle:
            parts.append(transform(translation=[self.distal_length, 0, 0])
                         @ transform(rz(-s * q[3])))
            parts.append(transform(translation=[self.sole_extension, 0, 0]))
        else:
            parts.append(transform(translation=[self.distal_length + self.sole_extension, 0, 0]))
        return tuple(parts)

    def points(self, q) -> np.ndarray:
        """Root, hip-flexion pivot, knee, fork/ankle pivot, endpoint in B."""
        q = vector(q, self.dof, "leg q")
        alpha, beta, knee = q[:3]
        ca, sa, s = np.cos(alpha), np.sin(alpha), self.side
        def point(F, D):
            return self.root + [s * self.c + F,
                s * (self.a * ca + (self.b + D) * sa),
                self.a * sa - (self.b + D) * ca]
        F1, D1 = self.proximal_length * np.sin(beta), self.proximal_length * np.cos(beta)
        fork = point(F1 + self.distal_length * np.sin(beta + knee),
                     D1 + self.distal_length * np.cos(beta + knee))
        return np.array([self.root, point(0, 0), point(F1, D1), fork, self.fk(q)[:3, 3]])

    def jacobian(self, q) -> np.ndarray:
        q, lengths, angles, F, D = self._terms(q)
        sa, ca, s, H = np.sin(q[0]), np.cos(q[0]), self.side, self.b + D
        J = np.zeros((3, self.dof))
        J[:, 0] = [0, s * (-self.a * sa + H * ca), self.a * ca + H * sa]
        for j in range(1, self.dof):
            dF = lengths[j-1:] @ np.cos(angles[j-1:])
            dD = -lengths[j-1:] @ np.sin(angles[j-1:])
            J[:, j] = [dF, s * sa * dD, -ca * dD]
        return J

    def angular_jacobian(self, q) -> np.ndarray:
        alpha = vector(q, self.dof, "leg q")[0]
        J = np.zeros((3, self.dof))
        J[:, 0] = [self.side, 0, 0]
        J[:, 1:] = np.array([0, -np.cos(alpha), -self.side * np.sin(alpha)])[:, None]
        return J

    def geometric_jacobian(self, q) -> np.ndarray:
        return np.vstack([self.angular_jacobian(q), self.jacobian(q)])

    def space_jacobian(self, q) -> np.ndarray:
        p, Jw = self.fk(q)[:3, 3], self.angular_jacobian(q)
        return np.vstack([Jw, self.jacobian(q) + skew(p) @ Jw])

    def body_jacobian(self, q) -> np.ndarray:
        return adjoint(inverse_transform(self.fk(q))) @ self.space_jacobian(q)

    def ik_candidates(self, position, *, sole_pitch=None, tolerance=1e-9) -> list[np.ndarray]:
        """Enumerate roll/knee branches inside stored limits.

        Position is the chosen endpoint in B. For an active ankle, sole_pitch
        is required and is measured within the rolled leg plane, not world RPY.
        Tolerance is a dimensionless relative roundoff allowance.
        """
        p = vector(position, 3, "target")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.active_ankle and sole_pitch is None:
            raise ValueError("active ankle analytical IK requires sole_pitch; use numerical_ik for position only")
        if sole_pitch is not None and not np.isfinite(sole_pitch):
            raise ValueError("sole_pitch must be finite")
        if not self.active_ankle and sole_pitch is not None:
            raise ValueError("rigid foot orientation follows position; no independent sole_pitch input")
        X = p[0] - self.root[0] - self.side * self.c
        Y = self.side * (p[1] - self.root[1])
        Z = p[2] - self.root[2]
        radius2 = Y * Y + Z * Z - self.a * self.a
        scale2 = max(Y * Y + Z * Z, self.a * self.a, 1e-12)
        if radius2 < -tolerance * scale2:
            return []
        root_H = np.sqrt(max(0., radius2))
        l1 = self.proximal_length
        l2 = self.distal_length + (0 if self.active_ankle else self.sole_extension)
        candidates = []
        for H in (root_H, -root_H):
            alpha = np.arctan2(Z, Y) + np.arctan2(H, self.a)
            F, D = X, H - self.b
            if self.active_ankle:
                F -= self.sole_extension * np.sin(sole_pitch)
                D -= self.sole_extension * np.cos(sole_pitch)
            ck = (F * F + D * D - l1 * l1 - l2 * l2) / (2 * l1 * l2)
            if abs(ck) > 1 + tolerance:
                continue
            ck = np.clip(ck, -1., 1.)
            for sign in (1, -1):
                knee = np.arctan2(sign * np.sqrt(max(0., 1 - ck * ck)), ck)
                if np.hypot(F, D) < 1e-12 and abs(l1 - l2) < 1e-12:
                    continue  # Folded continuum; no unique hip angle in this solver.
                hip = np.arctan2(F, D) - np.arctan2(l2 * np.sin(knee), l1 + l2 * np.cos(knee))
                values = [alpha, hip, knee]
                if self.active_ankle:
                    values.append(sole_pitch - hip - knee)
                values = np.arctan2(np.sin(values), np.cos(values))
                if np.all(values >= self.lower - 1e-10) and np.all(values <= self.upper + 1e-10):
                    if not any(np.linalg.norm(values - old) < 1e-9 for old in candidates):
                        candidates.append(np.clip(values, self.lower, self.upper))
        return candidates

    def ik(self, position, *, seed=None, sole_pitch=None) -> np.ndarray:
        candidates = self.ik_candidates(position, sole_pitch=sole_pitch)
        if not candidates:
            raise IKError("target has no analytical branch within the selected geometry and joint limits")
        reference = np.zeros(self.dof) if seed is None else vector(seed, self.dof, "seed")
        return min(candidates, key=lambda q: np.linalg.norm(q - reference)).copy()

    def numerical_ik(self, position, seed, *, damping=.01, tolerance_m=1e-6,
                     max_iterations=100, max_step_rad=.15) -> IKResult:
        """Bounded, backtracked damped least squares for endpoint position."""
        target = vector(position, 3, "target")
        q = np.clip(vector(seed, self.dof, "seed"), self.lower, self.upper)
        if (not np.all(np.isfinite([damping, tolerance_m, max_step_rad]))
                or min(damping, tolerance_m, max_step_rad) <= 0
                or not isinstance(max_iterations, int) or max_iterations < 1):
            raise ValueError("solver parameters must be finite and positive")
        for iteration in range(max_iterations + 1):
            error = target - self.fk(q)[:3, 3]
            residual = float(np.linalg.norm(error))
            if residual <= tolerance_m:
                return IKResult(q, True, residual, iteration, "position tolerance reached")
            if iteration == max_iterations:
                break
            J = self.jacobian(q)
            step = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), error)
            step *= min(1., max_step_rad / max(np.linalg.norm(step), 1e-15))
            for backtrack in range(14):
                trial = np.clip(q + step * .5**backtrack, self.lower, self.upper)
                if np.linalg.norm(target - self.fk(trial)[:3, 3]) < residual - 1e-14:
                    q = trial
                    break
            else:
                return IKResult(q, False, residual, iteration, "stalled at a limit, singularity, or local minimum")
        return IKResult(q, False, residual, max_iterations, "iteration limit reached")
