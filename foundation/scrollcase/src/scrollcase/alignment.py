import numpy as np
from scipy.spatial import ConvexHull
import logging
from . import smallest_circle
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _project_points_onto_plane(
    points: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normal must be a unit vector."""
    # reference = np.array([0, 1, 0]) if abs(normal[0]) > 0.9 else np.array([1, 0, 0])
    reference = np.array([0, 1, 0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    d = np.dot(points, normal)
    points_global = points - d[:, None] * normal
    points_local = np.hstack(
        [np.dot(points_global, u)[:, None], np.dot(points_global, v)[:, None]]
    )
    return points_global, points_local, d, np.vstack([u, v])


def fit_cylinder(
    points: np.ndarray, tol: float, _debug: bool = False
) -> tuple[np.ndarray | np.ndarray, float, float, float]:
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Optimize cylinder axis
    def objective(normal):
        normal_unit = normal / np.linalg.norm(normal)
        _, projection_local, _, _ = _project_points_onto_plane(hull_points, normal_unit)
        _, radius = smallest_circle.smallest_enclosing_circle(projection_local)
        # logger.debug(f"Trying direction {normal_unit} --> radius {radius}")
        return radius

    # Normal must be a unit vector
    constraints = [{"type": "eq", "fun": lambda n: np.dot(n, n) - 1}]

    n0 = np.array([0, 0, 1])
    result = minimize(
        objective,
        n0,
        constraints=constraints,
        method="SLSQP",
        tol=tol,
        # options={"disp": True},
    )

    best_normal = result.x / np.linalg.norm(result.x)
    best_radius = result.fun

    logger.info(f"Best direction: {best_normal} --> radius {best_radius}")

    normal = best_normal
    min_radius = best_radius

    # Transform points to cylinder coordinates
    projection_global, projection_local, d, basis = _project_points_onto_plane(
        hull_points, normal
    )
    center, radius = smallest_circle.smallest_enclosing_circle(projection_local)

    rotation_matrix = np.column_stack([basis.T, normal])
    rotation_matrix = rotation_matrix.T
    global_center = basis.T @ center  # center in global coordinates

    height = max(d) - min(d)

    translation_matrix = rotation_matrix @ -global_center
    translation_matrix[2] -= min(d)

    transformed_coords = rotation_matrix @ (hull_points.T) + translation_matrix[:, None]

    if _debug:
        transformed_coords = transformed_coords.T
        # transformed_coords[:, 2] -= z_translation

        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, max(d) - min(d), 100)
        Theta, Z = np.meshgrid(theta, z)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)

        scatter_points = go.Scatter3d(
            x=transformed_coords[:, 0],
            y=transformed_coords[:, 1],
            z=transformed_coords[:, 2],
            mode="markers",
            marker=dict(opacity=0.5, size=2),
        )

        # Create the surface plot
        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="reds",
            opacity=0.1,
            showscale=False,
            contours=dict(z=dict(show=True, color="black", width=1)),
        )

        # Combine both plots
        fig = go.Figure(data=[scatter_points, surface])

        # Set the aspect ratio and layout
        fig.update_layout(
            scene={"camera_projection_type": "orthographic", "aspectmode": "data"},
            showlegend=False,
            width=1000,
            height=1000,
        )

        fig.show(renderer="browser")

    return rotation_matrix, translation_matrix, min_radius, height
