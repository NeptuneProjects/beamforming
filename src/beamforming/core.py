from enum import StrEnum


class CoordinateSystem(StrEnum):
    CARTESIAN = "cartesian"
    POLAR = "polar"  # 2D only: (r, θ)
    CYLINDRICAL = "cylindrical"  # (r, θ, z)
    SPHERICAL = "spherical"  # (r, θ, φ) where θ is azimuth and φ is elevation
