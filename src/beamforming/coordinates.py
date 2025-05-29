from enum import StrEnum

import numpy as np


class CoordinateSystem(StrEnum):
    CARTESIAN = "cartesian"
    POLAR = "polar"  # 2D only: (r, θ)
    CYLINDRICAL = "cylindrical"  # (r, θ, z)
    SPHERICAL = "spherical"  # (r, θ, φ) where θ is azimuth and φ is elevation


def rotation_matrix_axis_angle(axis, angle):
    """
    Create rotation matrix from axis and angle using Rodrigues' rotation formula
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize

    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c

    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def rotation_matrix_x(angle):
    """Create rotation matrix for rotation around x-axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle):
    """Create rotation matrix for rotation around y-axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle):
    """Create rotation matrix for rotation around z-axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def to_cartesian_3d(coordinates, system):
    """Convert coordinates to 3D Cartesian"""
    coords = np.asarray(coordinates)
    result = np.zeros((coords.shape[0], 3))

    if system == CoordinateSystem.CARTESIAN:
        if coords.shape[1] == 2:
            # 2D Cartesian to 3D Cartesian
            result[:, 0] = coords[:, 0]  # x
            result[:, 1] = coords[:, 1]  # y
            # z = 0 by default
        else:
            # Already 3D Cartesian
            result = coords

    elif system == CoordinateSystem.POLAR:
        # Polar (r, θ) to 3D Cartesian
        r = coords[:, 0]
        theta = coords[:, 1]
        result[:, 0] = r * np.cos(theta)  # x
        result[:, 1] = r * np.sin(theta)  # y
        # z = 0 by default

    elif system == CoordinateSystem.CYLINDRICAL:
        # Cylindrical (r, θ, z) to 3D Cartesian
        r = coords[:, 0]
        theta = coords[:, 1]
        result[:, 0] = r * np.cos(theta)  # x
        result[:, 1] = r * np.sin(theta)  # y
        if coords.shape[1] > 2:  # Check if z is provided
            result[:, 2] = coords[:, 2]  # z

    elif system == CoordinateSystem.SPHERICAL:
        # Spherical (r, θ, φ) to 3D Cartesian
        r = coords[:, 0]
        theta = coords[:, 1]  # azimuth
        if coords.shape[1] > 2:
            phi = coords[:, 2]  # elevation
        else:
            phi = np.zeros_like(r)

        result[:, 0] = r * np.cos(theta) * np.cos(phi)  # x
        result[:, 1] = r * np.sin(theta) * np.cos(phi)  # y
        result[:, 2] = r * np.sin(phi)  # z

    return result
