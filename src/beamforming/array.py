import numpy as np

from beamforming.core import CoordinateSystem


class ArrayGeometry:
    def __init__(
        self,
        coordinates,
        coordinate_system=CoordinateSystem.CARTESIAN,
        fs=None,
        sound_speed=1500.0,
    ):
        """
        Parameters:
        -----------
        coordinates : ndarray
            Shape (n_sensors, 2) or (n_sensors, 3) depending on dimensions
            Coordinates in the specified system
        coordinate_system : CoordinateSystem
            Specifies the format of input coordinates
        fs : float, optional
            Sampling frequency in Hz
        sound_speed : float, optional
            Speed of sound in meters/second
        """
        self.fs = fs
        self.sound_speed = sound_speed

        # Convert input coordinates to Cartesian 3D (our internal standard)
        self.coordinates_original = np.asarray(coordinates)
        self.original_coordinate_system = coordinate_system
        self.sensor_positions = self._to_cartesian_3d(
            self.coordinates_original, coordinate_system
        )

        # Track transformations
        self.rotation_matrix = np.eye(3)  # Identity matrix (no rotation)
        self.translation_vector = np.zeros(3)  # No translation

    @property
    def center(self):
        """Centroid of the array in Cartesian coordinates"""
        return np.mean(self.sensor_positions, axis=0)

    @property
    def dimensions(self):
        """Returns 2 if all z-coordinates are 0, otherwise 3"""
        if np.allclose(self.sensor_positions[:, 2], 0):
            return 2
        return 3

    @property
    def n_sensors(self):
        """Number of sensors in the array"""
        return self.sensor_positions.shape[0]

    def apply_transformation(self, matrix):
        """
        Apply a 4x4 homogeneous transformation matrix

        Parameters:
        -----------
        matrix : ndarray, shape (4, 4)
            Homogeneous transformation matrix

        Returns:
        --------
        self : ArrayGeometry
            Returns self for method chaining
        """
        # Extract rotation and translation components
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        # Apply rotation
        center = self.center
        centered_positions = self.sensor_positions - center
        rotated_positions = np.dot(centered_positions, rotation.T)
        self.sensor_positions = rotated_positions + center

        # Apply translation
        self.sensor_positions += translation

        # Update transformation tracking
        self.rotation_matrix = np.dot(rotation, self.rotation_matrix)
        self.translation_vector += translation

        return self

    @classmethod
    def create_circular_array(
        cls, n_sensors, radius, plane="xy", center=None, fs=None, sound_speed=1500.0
    ):
        """
        Create a uniform circular array

        Parameters:
        -----------
        n_sensors : int
            Number of sensors
        radius : float
            Radius of the circle in meters
        plane : str
            Plane in which to create the circle ('xy', 'yz', or 'xz')
        center : ndarray, optional
            Center coordinates of the array, default is origin
        fs : float, optional
            Sampling frequency in Hz

        Returns:
        --------
        ArrayGeometry
            Circular array configuration
        """
        if center is None:
            center = np.zeros(3)
        else:
            center = np.asarray(center)

        positions = np.zeros((n_sensors, 3))
        angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)

        if plane == "xy":
            positions[:, 0] = radius * np.cos(angles)  # x
            positions[:, 1] = radius * np.sin(angles)  # y
        elif plane == "yz":
            positions[:, 1] = radius * np.cos(angles)  # y
            positions[:, 2] = radius * np.sin(angles)  # z
        elif plane == "xz":
            positions[:, 0] = radius * np.cos(angles)  # x
            positions[:, 2] = radius * np.sin(angles)  # z

        positions += center

        return cls(
            positions,
            coordinate_system=CoordinateSystem.CARTESIAN,
            fs=fs,
            sound_speed=sound_speed,
        )

    @classmethod
    def create_linear_array(
        cls, n_sensors, spacing, axis="x", center=True, fs=None, sound_speed=1500.0
    ):
        """
        Create a uniform linear array

        Parameters:
        -----------
        n_sensors : int
            Number of sensors
        spacing : float
            Spacing between sensors in meters
        axis : str
            Axis along which to place the array ('x', 'y', or 'z')
        center : bool
            If True, center the array at the origin
        fs : float, optional
            Sampling frequency in Hz

        Returns:
        --------
        ArrayGeometry
            Linear array configuration
        """
        positions = np.zeros((n_sensors, 3))
        idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

        if center:
            start = -spacing * (n_sensors - 1) / 2
        else:
            start = 0

        positions[:, idx] = np.arange(n_sensors) * spacing + start

        return cls(
            positions,
            coordinate_system=CoordinateSystem.CARTESIAN,
            fs=fs,
            sound_speed=sound_speed,
        )

    # @classmethod
    # def create_planar_array(cls, grid_shape, spacing, plane="xy", center=True, fs=None):
    #     """
    #     Create a uniform planar array

    #     Parameters:
    #     -----------
    #     grid_shape : tuple
    #         (rows, cols) for the grid
    #     spacing : float or tuple
    #         Spacing between elements in meters. If tuple, (row_spacing, col_spacing)
    #     plane : str
    #         Orientation plane ('xy', 'yz', or 'xz')
    #     center : bool
    #         If True, center the array at the origin
    #     fs : float, optional
    #         Sampling frequency in Hz

    #     Returns:
    #     --------
    #     ArrayGeometry
    #         Planar array configuration
    #     """
    #     # Implementation here

    # @classmethod
    # def create_spherical_array(
    #     cls, n_sensors, radius, distribution="fibonacci", fs=None
    # ):
    #     """
    #     Create a spherical array

    #     Parameters:
    #     -----------
    #     n_sensors : int
    #         Number of sensors
    #     radius : float
    #         Radius of the sphere in meters
    #     distribution : str
    #         Distribution method ('fibonacci', 'uniform', 'gaussian')
    #     fs : float, optional
    #         Sampling frequency in Hz

    #     Returns:
    #     --------
    #     ArrayGeometry
    #         Spherical array configuration
    #     """
    #     # Implementation here

    def delays(self, direction, reference_point="center"):
        """
        Calculate time delays for given direction

        Parameters:
        -----------
        direction : ndarray or tuple
            Direction vector or [azimuth, elevation] in radians
        reference_point : str or ndarray
            Reference point for delay calculation ('center', 'first', or coordinates)

        Returns:
        --------
        delays : ndarray
            Time delays for each sensor in seconds
        """
        # Convert direction format if needed
        if isinstance(direction, (list, tuple)) and len(direction) <= 3:
            # Convert [azimuth, elevation] to unit vector
            azimuth = direction[0]
            elevation = direction[1] if len(direction) > 1 else 0.0
            direction_vector = np.array(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
        else:
            direction_vector = np.asarray(direction)
            # Normalize
            direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Determine reference point
        if reference_point == "center":
            ref_pos = self.center
        elif reference_point == "first":
            ref_pos = self.sensor_positions[0]
        else:
            ref_pos = np.asarray(reference_point)

        # Calculate projection of sensor positions onto direction vector
        relative_positions = self.sensor_positions - ref_pos
        projections = np.dot(relative_positions, direction_vector)

        # Convert to time delays
        delays = (
            -projections / self.sound_speed
        )  # Negative because delay is in the opposite direction of arrival

        return delays

    def get_coordinates(self, coordinate_system=None):
        """
        Get sensor coordinates in the specified coordinate system

        Parameters:
        -----------
        coordinate_system : CoordinateSystem, optional
            Desired coordinate system. If None, returns the original coordinates.

        Returns:
        --------
        ndarray
            Sensor coordinates in the requested format
        """
        if coordinate_system is None:
            return self.coordinates_original

        # Convert from internal 3D Cartesian to requested system
        cart = self.sensor_positions
        result = None

        if coordinate_system == CoordinateSystem.CARTESIAN:
            if self.dimensions == 2:
                result = cart[:, :2]  # Return just x, y for 2D
            else:
                result = cart  # Return x, y, z for 3D

        elif coordinate_system == CoordinateSystem.POLAR:
            if self.dimensions == 3:
                print(
                    "Warning: Converting 3D array to 2D polar coordinates (ignoring z)"
                )

            result = np.zeros((self.n_sensors, 2))
            x, y = cart[:, 0], cart[:, 1]
            result[:, 0] = np.sqrt(x**2 + y**2)  # r
            result[:, 1] = np.arctan2(y, x)  # θ

        elif coordinate_system == CoordinateSystem.CYLINDRICAL:
            result = np.zeros((self.n_sensors, 3))
            x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
            result[:, 0] = np.sqrt(x**2 + y**2)  # r
            result[:, 1] = np.arctan2(y, x)  # θ
            result[:, 2] = z  # z

        elif coordinate_system == CoordinateSystem.SPHERICAL:
            result = np.zeros((self.n_sensors, 3))
            x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]

            r_xy = np.sqrt(x**2 + y**2)
            result[:, 0] = np.sqrt(r_xy**2 + z**2)  # r
            result[:, 1] = np.arctan2(y, x)  # θ (azimuth)
            result[:, 2] = np.arctan2(z, r_xy)  # φ (elevation)

        return result

    def get_transformation_matrix(self):
        """
        Get the 4x4 homogeneous transformation matrix representing all transformations

        Returns:
        --------
        ndarray, shape (4, 4)
            Homogeneous transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.translation_vector
        return matrix

    def plot(self, coordinate_system=None, ax=None, show=None, **kwargs):
        """
        Plot the array geometry

        Parameters:
        -----------
        coordinate_system : CoordinateSystem, optional
            System to plot in (defaults to Cartesian)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        show : bool, optional
            Whether to show the plot immediately
        **kwargs : dict
            Additional arguments passed to plotting functions

        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        from beamforming.plotting import plot_array_geometry

        if coordinate_system is None:
            coordinate_system = CoordinateSystem.CARTESIAN

        # Get coordinates in the requested system
        coords = self.get_coordinates(coordinate_system)

        # Pass to the plotting function
        return plot_array_geometry(
            coords,
            coordinate_system=coordinate_system,
            array_center=self.center,
            rotation_matrix=self.rotation_matrix,
            ax=ax,
            show=show,
            **kwargs
        )

    def reset_transform(self):
        """
        Reset all transformations (rotation and translation)

        Returns:
        --------
        self : ArrayGeometry
            Returns self for method chaining
        """
        # Recompute sensor positions from original coordinates
        self.sensor_positions = self._to_cartesian_3d(
            self.coordinates_original, self.original_system
        )

        # Reset transformation tracking
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.zeros(3)

        return self

    def rotate(self, rotation_matrix):
        """
        Rotate array using a 3x3 rotation matrix
        """
        # Apply rotation to sensor positions
        center = self.center
        centered_positions = self.sensor_positions - center

        rotated_positions = np.dot(centered_positions, rotation_matrix.T)
        self.sensor_positions = rotated_positions + center

        # Update transformation tracking
        self.rotation_matrix = np.dot(rotation_matrix, self.rotation_matrix)

        return self

    def rotate_axis_angle(self, axis, angle, degrees=False):
        """
        Rotate array around an axis by an angle
        """
        if degrees:
            angle = np.radians(angle)

        rot_matrix = self._rotation_matrix_axis_angle(axis, angle)
        return self.rotate(rot_matrix)

    def rotate_azimuth(self, azimuth, degrees=False):
        """
        Rotate array in azimuth (around z-axis)
        """
        if degrees:
            azimuth = np.radians(azimuth)
        return self.rotate(self._rotation_matrix_z(azimuth))

    def rotate_azimuth_elevation(self, azimuth, elevation, degrees=False):
        """
        Rotate array in both azimuth and elevation
        """
        # First rotate in elevation, then in azimuth
        return self.rotate_elevation(elevation, degrees).rotate_azimuth(
            azimuth, degrees
        )

    def rotate_elevation(self, elevation, degrees=False):
        """
        Rotate array in elevation (around y-axis)
        """
        if degrees:
            elevation = np.radians(elevation)
        return self.rotate(self._rotation_matrix_y(elevation))

    def rotate_euler(self, angles, sequence="zyx", degrees=False):
        """
        Rotate array using Euler angles

        Parameters:
        -----------
        angles : list or ndarray
            Euler angles for rotation
        sequence : str
            Rotation sequence (e.g., 'zyx', 'xyz')
        degrees : bool
            If True, angles are in degrees, otherwise radians
        """
        if degrees:
            angles = np.radians(angles)

        rot_matrix = np.eye(3)
        for i, axis in enumerate(sequence):
            angle = angles[i]
            if axis.lower() == "x":
                rot_i = self._rotation_matrix_x(angle)
            elif axis.lower() == "y":
                rot_i = self._rotation_matrix_y(angle)
            elif axis.lower() == "z":
                rot_i = self._rotation_matrix_z(angle)
            else:
                raise ValueError(f"Invalid axis: {axis}")

            rot_matrix = np.dot(rot_i, rot_matrix)

        return self.rotate(rot_matrix)

    def _rotation_matrix_axis_angle(self, axis, angle):
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

    def _rotation_matrix_x(self, angle):
        """Create rotation matrix for rotation around x-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rotation_matrix_y(self, angle):
        """Create rotation matrix for rotation around y-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rotation_matrix_z(self, angle):
        """Create rotation matrix for rotation around z-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _to_cartesian_3d(self, coordinates, system):
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

    def translate(self, vector):
        """
        Translate array by a vector

        Parameters:
        -----------
        vector : ndarray, shape (3,)
            Translation vector [x, y, z]

        Returns:
        --------
        self : ArrayGeometry
            Returns self for method chaining
        """
        vector = np.asarray(vector)
        if vector.shape == (2,):
            # If 2D vector, assume z=0
            vector = np.append(vector, 0)

        # Apply translation
        self.sensor_positions += vector

        # Update transformation tracking
        self.translation_vector += vector

        return self
