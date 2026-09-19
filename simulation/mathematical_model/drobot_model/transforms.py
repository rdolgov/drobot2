"""SI units, column vectors, T_AB maps B coordinates into A.

Twists are angular-first [omega, v]. Floating generalized speeds elsewhere
are [base linear velocity, base angular velocity, joint rates], all in world.
"""
from __future__ import annotations

import numpy as np


def vector(value, size: int, name: str = "vector") -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result


def skew(value) -> np.ndarray:
    x, y, z = vector(value, 3)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rpy_rotation(rpy) -> np.ndarray:
    roll, pitch, yaw = vector(rpy, 3, "rpy")
    return rz(yaw) @ ry(pitch) @ rx(roll)


def so3_exp(rotation_vector) -> np.ndarray:
    phi = vector(rotation_vector, 3, "rotation vector")
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-7:
        t2 = theta * theta
        a = 1 - t2 / 6 + t2 * t2 / 120
        b = .5 - t2 / 24 + t2 * t2 / 720
    else:
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta**2
    return np.eye(3) + a * K + b * (K @ K)


def transform(rotation=None, translation=None) -> np.ndarray:
    result = np.eye(4)
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        result[:3, :3] = R
    if translation is not None:
        result[:3, 3] = vector(translation, 3, "translation")
    return result


def pose_matrix(value=None) -> np.ndarray:
    """Reject malformed poses at the public model boundary."""
    if value is None:
        return np.eye(4)
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("base_pose must be a finite 4x4 rigid transform")
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-7, rtol=0)):
        raise ValueError("base_pose must contain a proper rotation and [0,0,0,1] bottom row")
    return T.copy()


def inverse_transform(T) -> np.ndarray:
    T = pose_matrix(T)
    R, p = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ p)


def adjoint(T) -> np.ndarray:
    T = pose_matrix(T)
    R, p = T[:3, :3], T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])


def screw_exp(screw, displacement: float) -> np.ndarray:
    """Exponential of a constant [omega, v] screw times displacement."""
    screw = vector(screw, 6, "screw")
    if not np.isfinite(displacement):
        raise ValueError("displacement must be finite")
    phi, rho = screw[:3] * displacement, screw[3:] * displacement
    theta, K = np.linalg.norm(phi), skew(phi)
    if theta < 1e-7:
        t2 = theta * theta
        b = .5 - t2 / 24 + t2 * t2 / 720
        c = 1 / 6 - t2 / 120 + t2 * t2 / 5040
    else:
        b = (1 - np.cos(theta)) / theta**2
        c = (theta - np.sin(theta)) / theta**3
    V = np.eye(3) + b * K + c * (K @ K)
    return transform(so3_exp(phi), V @ rho)
