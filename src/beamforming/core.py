from abc import ABC, abstractmethod
from enum import StrEnum


class BeamformerBase(ABC):
    def __init__(self, array_geometry):
        """
        Parameters:
        -----------
        array_geometry : ArrayGeometry
            Geometry of the sensor array
        """
        self.array_geometry = array_geometry

    @abstractmethod
    def process(self, signal, direction):
        """
        Process signal to enhance a specific direction

        Parameters:
        -----------
        signal : Signal
            Multichannel input signal
        direction : ndarray
            Direction of interest

        Returns:
        --------
        Signal
            Beamformed output signal
        """
        pass

    @abstractmethod
    def response(self, frequencies, directions):
        """
        Calculate beampattern/directivity

        Parameters:
        -----------
        frequencies : ndarray
            Frequencies to calculate response for
        directions : ndarray
            Directions to calculate response for

        Returns:
        --------
        ndarray
            Beampattern response
        """
        pass


class CoordinateSystem(StrEnum):
    CARTESIAN = "cartesian"
    POLAR = "polar"  # 2D only: (r, θ)
    CYLINDRICAL = "cylindrical"  # (r, θ, z)
    SPHERICAL = "spherical"  # (r, θ, φ) where θ is azimuth and φ is elevation
