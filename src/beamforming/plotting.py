import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum

from beamforming.coordinates import CoordinateSystem


def plot_array_geometry(
    coords,
    coordinate_system=CoordinateSystem.CARTESIAN,
    array_center=None,
    rotation_matrix=None,
    ax=None,
    show=None,
    marker="o",
    color="blue",
    size=50,
    show_axes=True,
    show_center=True,
    show_labels=True,
    title=None,
    equal_aspect=True,
    **kwargs
):
    """
    Plot array geometry in the specified coordinate system

    Parameters:
    -----------
    coords : ndarray
        Coordinates in the specified system
    coordinate_system : CoordinateSystem
        The coordinate system of the provided coordinates
    array_center : ndarray, optional
        Center of the array (for reference)
    rotation_matrix : ndarray, optional
        Current rotation matrix (for visualization)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None a new figure is created
    show : bool, optional
        Whether to show the plot immediately
    marker : str
        Marker style for sensors
    color : str
        Color for sensors
    size : float
        Marker size
    show_axes : bool
        Whether to show coordinate axes
    show_center : bool
        Whether to show array center
    show_labels : bool
        Whether to show sensor labels
    title : str, optional
        Plot title
    equal_aspect : bool
        Whether to use equal aspect ratio
    **kwargs : dict
        Additional arguments passed to plotting functions

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    coords = np.asarray(coords)
    is_3d = (coords.shape[1] > 2) or (
        coordinate_system in [CoordinateSystem.CYLINDRICAL, CoordinateSystem.SPHERICAL]
    )

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    # Call the appropriate plotting function based on coordinate system
    if coordinate_system == CoordinateSystem.CARTESIAN:
        ax = _plot_cartesian(coords, ax, is_3d, marker, color, size, show_labels)
    elif coordinate_system == CoordinateSystem.POLAR:
        ax = _plot_polar(coords, ax, marker, color, size, show_labels)
    elif coordinate_system == CoordinateSystem.CYLINDRICAL:
        ax = _plot_cylindrical(coords, ax, marker, color, size, show_labels)
    elif coordinate_system == CoordinateSystem.SPHERICAL:
        ax = _plot_spherical(coords, ax, marker, color, size, show_labels)

    # Show array center
    if show_center and array_center is not None:
        if is_3d:
            ax.scatter(
                [array_center[0]],
                [array_center[1]],
                [array_center[2]],
                color="red",
                marker="x",
                s=size * 1.5,
                label="Center",
            )
        else:
            ax.scatter(
                [array_center[0]],
                [array_center[1]],
                color="red",
                marker="x",
                s=size * 1.5,
                label="Center",
            )

    # Show coordinate axes and orientation
    if show_axes and rotation_matrix is not None:
        _plot_orientation(
            ax,
            array_center,
            rotation_matrix,
            is_3d,
            scale=kwargs.get("axis_scale", 0.2),
        )

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Array Geometry - {coordinate_system.value}")

    # Set equal aspect ratio if requested
    if equal_aspect:
        if is_3d:
            # Set equal aspect ratio for 3D
            _set_axes_equal(ax)
        else:
            ax.set_aspect("equal")

    # Add legend if labels are shown
    if show_labels or show_center:
        ax.legend()

    # Show plot if requested
    if show or (show is None and ax is not None):
        plt.tight_layout()
        plt.show()

    return ax


def _plot_cartesian(coords, ax, is_3d, marker, color, size, show_labels):
    """Plot array in Cartesian coordinates"""
    if is_3d:
        for i, (x, y, z) in enumerate(coords):
            ax.scatter([x], [y], [z], marker=marker, color=color, s=size)
            if show_labels:
                ax.text(x, y, z, f" {i}", fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, marker=marker, color=color, s=size)
            if show_labels:
                ax.text(x, y, f" {i}", fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    return ax



def _plot_cylindrical(coords, ax, marker, color, size, show_labels):
    """Plot array in cylindrical coordinates"""
    for i, (r, theta, z) in enumerate(coords):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter([x], [y], [z], marker=marker, color=color, s=size)
        if show_labels:
            ax.text(x, y, z, f" {i}", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax



def _plot_orientation(ax, center, rotation_matrix, is_3d, scale=0.2):
    """Plot coordinate axes to show orientation"""
    if center is None:
        center = np.zeros(3)

    # Create basis vectors (after rotation)
    x_axis = rotation_matrix.dot([scale, 0, 0])
    y_axis = rotation_matrix.dot([0, scale, 0])
    z_axis = rotation_matrix.dot([0, 0, scale])

    if is_3d:
        # X-axis (red)
        ax.quiver(
            center[0],
            center[1],
            center[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            arrow_length_ratio=0.1,
        )

        # Y-axis (green)
        ax.quiver(
            center[0],
            center[1],
            center[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            arrow_length_ratio=0.1,
        )

        # Z-axis (blue)
        ax.quiver(
            center[0],
            center[1],
            center[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            arrow_length_ratio=0.1,
        )
    else:
        # Only plot X and Y for 2D
        ax.arrow(
            center[0],
            center[1],
            x_axis[0],
            x_axis[1],
            head_width=0.05 * scale,
            head_length=0.1 * scale,
            fc="red",
            ec="red",
        )
        ax.arrow(
            center[0],
            center[1],
            y_axis[0],
            y_axis[1],
            head_width=0.05 * scale,
            head_length=0.1 * scale,
            fc="green",
            ec="green",
        )


def _plot_polar(coords, ax, marker, color, size, show_labels):
    """Plot array in polar coordinates"""
    # Convert polar to cartesian for plotting
    for i, (r, theta) in enumerate(coords):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter(x, y, marker=marker, color=color, s=size)
        if show_labels:
            ax.text(x, y, f" {i}", fontsize=8)

    # Set up polar grid
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add polar grid lines
    r_ticks = np.linspace(0, np.max(coords[:, 0]) * 1.1, 5)[1:]
    for r in r_ticks:
        circle = plt.Circle(
            (0, 0), r, fill=False, color="gray", linestyle="--", alpha=0.3
        )
        ax.add_artist(circle)

    theta_ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for theta in theta_ticks:
        x = [0, np.cos(theta) * np.max(coords[:, 0]) * 1.1]
        y = [0, np.sin(theta) * np.max(coords[:, 0]) * 1.1]
        ax.plot(x, y, "gray", linestyle="--", alpha=0.3)
        ax.text(
            x[1] * 1.05,
            y[1] * 1.05,
            f"{int(np.degrees(theta))}°",
            fontsize=8,
            ha="center",
            va="center",
            color="gray",
        )

    return ax


def _plot_spherical(coords, ax, marker, color, size, show_labels):
    """Plot array in spherical coordinates"""
    for i, (r, theta, phi) in enumerate(coords):
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)
        ax.scatter([x], [y], [z], marker=marker, color=color, s=size)
        if show_labels:
            ax.text(x, y, z, f" {i}", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


def _set_axes_equal(ax):
    """Set 3D axes to equal scale"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call it a sphere.
    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


# Additional plotting functions for specific array types
def plot_beam_pattern(
    beamformer,
    frequency,
    grid_points=360,
    elevation=0,
    ax=None,
    show=None,
    colormap="viridis",
    **kwargs
):
    """
    Plot beam pattern for a given beamformer

    Parameters:
    -----------
    beamformer : BeamformerBase
        Beamformer to plot pattern for
    frequency : float
        Frequency in Hz
    grid_points : int
        Number of points in the grid
    elevation : float
        Fixed elevation angle for 2D plot (in degrees)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show : bool, optional
        Whether to show the plot immediately
    colormap : str
        Colormap to use
    **kwargs : dict
        Additional arguments for plotting

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Implementation for beam pattern plotting
    # This would call the beamformer's response method and visualize it
    pass


# Function for interactive 3D visualization
def interactive_array_plot(
    array_geometry, coordinate_system=CoordinateSystem.CARTESIAN
):
    """
    Create an interactive 3D plot of the array geometry

    Parameters:
    -----------
    array_geometry : ArrayGeometry
        Array geometry to visualize
    coordinate_system : CoordinateSystem
        Coordinate system for visualization

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the interactive plot
    """
    from matplotlib.widgets import Slider, Button

    # Implementation for interactive visualization
    # This could include sliders for rotation, etc.
    pass


def plot_array_response(
    array_geometry, frequency, direction=None, ax=None, show=None, **kwargs
):
    """
    Plot array response for a given frequency and optional steering direction

    Parameters:
    -----------
    array_geometry : ArrayGeometry
        Array geometry to use
    frequency : float
        Frequency in Hz
    direction : ndarray, optional
        Steering direction
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show : bool, optional
        Whether to show the plot immediately
    **kwargs : dict
        Additional arguments for plotting

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Calculate and plot array response
    pass


def plot_array_comparison(
    arrays,
    labels=None,
    coordinate_system=CoordinateSystem.CARTESIAN,
    ax=None,
    show=None,
    colors=None,
    **kwargs
):
    """
    Plot multiple arrays for comparison

    Parameters:
    -----------
    arrays : list of ArrayGeometry
        Arrays to compare
    labels : list of str, optional
        Labels for each array
    coordinate_system : CoordinateSystem
        Coordinate system for visualization
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show : bool, optional
        Whether to show the plot immediately
    colors : list, optional
        Colors for each array
    **kwargs : dict
        Additional arguments for plotting

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Plot multiple arrays with different colors/markers
    pass
