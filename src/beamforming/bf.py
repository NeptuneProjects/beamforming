from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal import ShortTimeFFT, get_window

from beamforming.array import ArrayGeometry
from beamforming.coordinates import CoordinateSystem


class BeamformerBase(ABC):

    def __init__(
        self,
        array_geometry: ArrayGeometry,
    ) -> None:
        self.array_geometry = array_geometry

    @abstractmethod
    def get_weights(
        self,
        frequency: float,
        direction: Sequence[float],
        coordinate_system: CoordinateSystem,
        **kwargs
    ):
        pass

    def _initialize_stft(
        self,
        sampling_rate: float,
        window_type: str = "hann",
        window_size: int = 1024,
        hop: int = 512,
        nfft: int | None = None,
    ) -> None:
        window = get_window(window_type, window_size)
        self.STFT = ShortTimeFFT(window, hop, sampling_rate, mfft=nfft)

    def process(
        self,
        data: npt.NDArray,
        direction: Sequence[float],
        sampling_rate: float,
        window_type: str = "hann",
        window_size: int = 1024,
        hop_size: int = 512,
        nfft: int = None,
        **kwargs
    ) -> npt.NDArray[np.float64]:
        self._initialize_stft(
            sampling_rate,
            window_type=window_type,
            window_size=window_size,
            hop=hop_size,
            nfft=nfft,
        )

        stft_data = self.STFT.stft(data)
        beamformed_stft_data = self.process_frequency_domain(
            stft_data, direction, **kwargs
        )
        return self.STFT.istft(beamformed_stft_data, k1=data.shape[-1])

    # @abstractmethod
    # def process_frequency_domain(
    #     self,
    #     stft_data: npt.NDArray[np.complex128],
    #     direction: Sequence[float],
    #     coordinate_system: CoordinateSystem,
    #     **kwargs
    # ):
    #     pass

    def process_frequency_domain(
        self,
        stft_data: npt.NDArray[np.complex128],
        direction: Sequence[float],
        coordinate_system: CoordinateSystem,
    ) -> npt.NDArray[np.complex128]:
        # stft_data is expected to be in shape (n_sensors, n_frequencies, n_time_bins)
        _, n_freq_bins, n_frames = stft_data.shape
        output_stft = np.zeros((n_freq_bins, n_frames), dtype=np.complex128)
        for f_idx, freq in enumerate(self.STFT.f):
            weights = self.get_weights(freq, direction, coordinate_system)
            for frame in range(n_frames):
                output_stft[f_idx, frame] = np.sum(weights * stft_data[:, f_idx, frame])

        return output_stft

    def response(
        self,
        frequencies: Sequence[float],
        directions: Sequence[float] | Sequence[Sequence[float]],
        coordinate_system: CoordinateSystem,
        normalize: bool = True,
        steering_direction: Sequence[float] | None = None,
        **kwargs
    ) -> npt.NDArray[np.float64]:
        frequencies = np.atleast_1d(frequencies)
        directions = np.atleast_1d(directions)
        response = np.zeros((len(frequencies), len(directions)))

        if steering_direction is None:
            steering_direction = directions[0] if len(directions) > 0 else [0.0, 0.0]

        for f_idx, freq in enumerate(frequencies):
            weights = self.get_weights(
                freq, steering_direction, coordinate_system, **kwargs
            )

            for d_idx, direction in enumerate(directions):
                steering_vector = self.array_geometry.steering_vector(
                    freq, direction, coordinate_system
                )

                response[f_idx, d_idx] = np.abs(np.sum(weights * steering_vector))

        if normalize:
            for f_idx in range(len(frequencies)):
                if np.max(response[f_idx]) > 0:
                    response[f_idx] /= np.max(response[f_idx])

        return response


class DelayAndSumBeamformer(BeamformerBase):
    def __init__(
        self, array_geometry: ArrayGeometry, weights: Sequence[float] | None = None
    ) -> None:
        super().__init__(array_geometry)
        if weights is None:
            self.weights = np.ones(array_geometry.n_sensors) / array_geometry.n_sensors
        else:
            self.weights = np.asarray(weights) / np.sum(np.abs(weights))

        if len(self.weights) != array_geometry.n_sensors:
            raise ValueError(
                f"Weights length {len(self.weights)} does not match number "
                f"of sensors {array_geometry.n_sensors}."
            )

    def get_weights(
        self,
        frequency: float,
        direction: Sequence[float],
        coordinate_system: CoordinateSystem,
    ) -> npt.NDArray[np.complex128]:
        steering_vector = self.array_geometry.steering_vector(
            frequency, direction, coordinate_system
        )
        return self.weights * np.conjugate(steering_vector)


class MVDRBeamformer(BeamformerBase):
    """
    Minimum Variance Distortionless Response (MVDR) Beamformer

    Also known as Capon beamformer, it minimizes output power while maintaining
    unity gain in the look direction.
    """

    def __init__(self, array_geometry, diagonal_loading=0.01):
        super().__init__(array_geometry)
        self.diagonal_loading = diagonal_loading
        self.covariance_matrices = {}  # Frequency-dependent covariance matrices

    def train(
        self,
        data: npt.NDArray[np.float64],
        sampling_rate: float,
        window_type: str = "hann",
        window_size: int = 1024,
        hop_size: int = 512,
        nfft: int = None,
    ):
        """
        Estimate spatial covariance matrices from signals
        """
        # TODO: Refactor into a separate covariance module
        self._initialize_stft(
            sampling_rate,
            window_type=window_type,
            window_size=window_size,
            hop=hop_size,
            nfft=nfft,
        )
        stft_data = self.STFT.stft(data)
        frequencies = self.STFT.f

        n_channels, _, n_frames = stft_data.shape

        # Estimate covariance matrices for each frequency
        for f_idx, freq in enumerate(frequencies):
            if freq == 0:  # Skip DC
                continue

            # Get all frames for this frequency
            X = stft_data[:, f_idx, ...]  # shape: (n_frames, n_channels)
            # Estimate covariance matrix
            R = np.zeros((n_channels, n_channels), dtype=complex)
            for frame in range(n_frames):
                x = X[:, frame].reshape(-1, 1)  # Make column vector
                R += x @ np.conjugate(x).T

            R /= n_frames

            # Apply diagonal loading for regularization
            R += self.diagonal_loading * np.eye(n_channels) * np.trace(R) / n_channels

            # Store covariance matrix
            self.covariance_matrices[freq] = R

        return self

    def get_weights(self, frequency, direction, coordinate_system, **kwargs):
        """
        Calculate MVDR beamforming weights

        For MVDR, the weights are:
        w = R^-1 * d / (d^H * R^-1 * d)
        where R is the covariance matrix and d is the steering vector
        """
        # Get steering vector
        d = self.array_geometry.steering_vector(frequency, direction, coordinate_system)

        # Get covariance matrix (or identity if not trained)
        if frequency in self.covariance_matrices:
            R = self.covariance_matrices[frequency]
        else:
            # Fall back to delay-and-sum if not trained
            return super().get_weights(
                frequency, direction, coordinate_system, **kwargs
            )

        # Calculate R^-1 * d
        try:
            R_inv_d = np.linalg.solve(R, d)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            R_inv = np.linalg.pinv(R)
            R_inv_d = R_inv @ d

        # Calculate d^H * R^-1 * d (scalar)
        d_H_R_inv_d = np.conjugate(d) @ R_inv_d

        # MVDR weights
        w = R_inv_d / d_H_R_inv_d

        return w

    # def process_frequency_domain(
    #     self, stft_data, direction, coordinate_system, **kwargs
    # ):
    #     """
    #     Apply MVDR beamforming in frequency domain
    #     """
    #     # Extract STFT data
    #     stft = stft_data["stft"]
    #     frequencies = stft_data["frequencies"]
    #     times = stft_data["times"]

    #     n_bins, n_frames, n_channels = stft.shape

    #     # Initialize output
    #     output_stft = np.zeros((n_bins, n_frames), dtype=complex)

    #     # Process each frequency bin
    #     for f_idx, freq in enumerate(frequencies):
    #         # Get weights for this frequency
    #         weights = self.get_weights(freq, direction, coordinate_system, **kwargs)

    #         # Apply weights and sum
    #         for frame in range(n_frames):
    #             output_stft[f_idx, frame] = np.sum(weights * stft[f_idx, frame])

    #     # Return in same format
    #     return output_stft
