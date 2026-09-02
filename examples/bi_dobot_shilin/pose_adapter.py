"""Pose conversions between the Dobot/OpenPI and LingBot VLA conventions.

The Dobot side uses a 20D vector::

    left_xyz(3), left_rotation_columns_6d(6),
    right_xyz(3), right_rotation_columns_6d(6),
    left_gripper(1), right_gripper(1)

The LingBot VLA ``own_robot`` config uses a 16D vector::

    left_xyz(3), left_quaternion_xyzw(4),
    right_xyz(3), right_quaternion_xyzw(4),
    left_gripper(1), right_gripper(1)

The 6D rotation is stored as the first rotation-matrix column followed by the
second column. Incoming columns are orthonormalized before quaternion
conversion to suppress small numerical/controller errors.
"""

from __future__ import annotations

import numpy as np


ROBOT_POSE_DIM = 20
LINGBOT_POSE_DIM = 16
_EPS = 1e-8


def _as_finite_array(value: np.ndarray, *, name: str, last_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != last_dim:
        raise ValueError(f"{name} must have shape (..., {last_dim}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _normalize(vectors: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < _EPS):
        raise ValueError(f"{name} contains a near-zero vector")
    return vectors / norms


def rotation_6d_columns_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert two stored rotation-matrix columns to a proper rotation matrix."""

    rotation_6d = _as_finite_array(rotation_6d, name="rotation_6d", last_dim=6)
    first = _normalize(rotation_6d[..., 0:3], name="rotation first column")
    second_raw = rotation_6d[..., 3:6]
    second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second = _normalize(second, name="rotation second column")
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to canonical normalized xyzw quaternions."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains NaN or infinite values")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    trace = m00 + m11 + m22

    quaternion = np.zeros((*matrix.shape[:-2], 4), dtype=np.float64)

    positive = trace > 0
    scale = 2.0 * np.sqrt(np.maximum(trace + 1.0, _EPS))
    quaternion[..., 0] = np.where(positive, (m21 - m12) / scale, quaternion[..., 0])
    quaternion[..., 1] = np.where(positive, (m02 - m20) / scale, quaternion[..., 1])
    quaternion[..., 2] = np.where(positive, (m10 - m01) / scale, quaternion[..., 2])
    quaternion[..., 3] = np.where(positive, 0.25 * scale, quaternion[..., 3])

    x_largest = (~positive) & (m00 > m11) & (m00 > m22)
    scale_x = 2.0 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, _EPS))
    quaternion[..., 0] = np.where(x_largest, 0.25 * scale_x, quaternion[..., 0])
    quaternion[..., 1] = np.where(x_largest, (m01 + m10) / scale_x, quaternion[..., 1])
    quaternion[..., 2] = np.where(x_largest, (m02 + m20) / scale_x, quaternion[..., 2])
    quaternion[..., 3] = np.where(x_largest, (m21 - m12) / scale_x, quaternion[..., 3])

    y_largest = (~positive) & (~x_largest) & (m11 > m22)
    scale_y = 2.0 * np.sqrt(np.maximum(1.0 - m00 + m11 - m22, _EPS))
    quaternion[..., 0] = np.where(y_largest, (m01 + m10) / scale_y, quaternion[..., 0])
    quaternion[..., 1] = np.where(y_largest, 0.25 * scale_y, quaternion[..., 1])
    quaternion[..., 2] = np.where(y_largest, (m12 + m21) / scale_y, quaternion[..., 2])
    quaternion[..., 3] = np.where(y_largest, (m02 - m20) / scale_y, quaternion[..., 3])

    z_largest = (~positive) & (~x_largest) & (~y_largest)
    scale_z = 2.0 * np.sqrt(np.maximum(1.0 - m00 - m11 + m22, _EPS))
    quaternion[..., 0] = np.where(z_largest, (m02 + m20) / scale_z, quaternion[..., 0])
    quaternion[..., 1] = np.where(z_largest, (m12 + m21) / scale_z, quaternion[..., 1])
    quaternion[..., 2] = np.where(z_largest, 0.25 * scale_z, quaternion[..., 2])
    quaternion[..., 3] = np.where(z_largest, (m10 - m01) / scale_z, quaternion[..., 3])

    quaternion = _normalize(quaternion, name="quaternion")
    # q and -q represent the same rotation. A non-negative w makes the state
    # deterministic and matches the dataset conversion used for training.
    return np.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to proper rotation matrices."""

    quaternion = _as_finite_array(
        quaternion,
        name="quaternion_xyzw",
        last_dim=4,
    )
    quaternion = _normalize(quaternion, name="quaternion_xyzw")
    x, y, z, w = np.moveaxis(quaternion, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def matrix_to_rotation_6d_columns(matrix: np.ndarray) -> np.ndarray:
    """Store the first matrix column followed by the second matrix column."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def robot_state_to_lingbot_state(state: np.ndarray) -> np.ndarray:
    """Convert Dobot/OpenPI 20D state vectors to LingBot VLA 16D vectors."""

    state = _as_finite_array(state, name="robot state", last_dim=ROBOT_POSE_DIM)
    left_quaternion = matrix_to_quaternion_xyzw(
        rotation_6d_columns_to_matrix(state[..., 3:9])
    )
    right_quaternion = matrix_to_quaternion_xyzw(
        rotation_6d_columns_to_matrix(state[..., 12:18])
    )
    converted = np.concatenate(
        (
            state[..., 0:3],
            left_quaternion,
            state[..., 9:12],
            right_quaternion,
            state[..., 18:20],
        ),
        axis=-1,
    )
    return converted.astype(np.float32, copy=False)


def lingbot_actions_to_robot_actions(actions: np.ndarray) -> np.ndarray:
    """Convert LingBot VLA 16D action vectors to Dobot/OpenPI 20D vectors."""

    actions = _as_finite_array(
        actions,
        name="LingBot actions",
        last_dim=LINGBOT_POSE_DIM,
    )
    left_rotation = matrix_to_rotation_6d_columns(
        quaternion_xyzw_to_matrix(actions[..., 3:7])
    )
    right_rotation = matrix_to_rotation_6d_columns(
        quaternion_xyzw_to_matrix(actions[..., 10:14])
    )
    converted = np.concatenate(
        (
            actions[..., 0:3],
            left_rotation,
            actions[..., 7:10],
            right_rotation,
            actions[..., 14:16],
        ),
        axis=-1,
    )
    return converted.astype(np.float32, copy=False)
