from enum import Enum, auto
import logging
import tempfile
from typing import Optional

import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import numpy as np
import trimesh

from . import alignment

logger = logging.getLogger(__name__)


def show_meshlib(*meshes: mm.Mesh, show_axis: bool = True):
    trimesh_meshes = []
    for mesh in meshes:
        with tempfile.NamedTemporaryFile("w", suffix=".stl") as f:
            mm.saveMesh(mesh, f.name)
            tri_mesh = trimesh.load(f.name)
            trimesh_meshes.append(tri_mesh)
    axis = trimesh.creation.axis(origin_size=10)
    if show_axis:
        trimesh_meshes.append(axis)
    scene = trimesh.Scene(trimesh_meshes)
    return scene.show(resolution=(1200, 600))


def count_vertices(mesh: mm.Mesh):
    return mn.getNumpyVerts(mesh).shape[0]


def _copy_meshlib(mesh: mm.Mesh):
    copy = mm.Mesh()
    copy.addPart(mesh)
    return copy


def load_mesh(mesh_file: str) -> mm.Mesh:
    return mm.loadMesh(mesh_file)


class RotationStrategy(Enum):
    NONE = auto(), "No rotation"
    SECOND_PRINCIPAL = auto(), "Split along second principal axis (wide)."
    THIRD_PRINCIPAL = auto(), "Split along third principal axis."
    DEG_45 = auto(), "45 degrees to principal axes."


def affine_rotation(angle_rad: float) -> mm.AffineXf3f:
    """Affine rotation about Z axis."""
    rotation_mat = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad), 0],
            [-np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    rotation_mat_mm = mm.AffineXf3f(
        mm.Matrix3f(*[mm.Vector3f(*d) for d in rotation_mat]),
        mm.Vector3f(0, 0, 0),
    )
    return rotation_mat_mm


def build_lining(
    mesh_scroll: mm.Mesh,
    *,
    rotation_strategy: RotationStrategy = RotationStrategy.THIRD_PRINCIPAL,
    cylinder_axis_tol: float = 0.01,
    voxel_size_diagonal_percent: float = 0.4,
    simplify_max_error_diagonal_percent: float = 1,
    target_scale_diagonal_mm: Optional[float] = None,
    _debug=False,
) -> tuple[mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, float, float]:
    """Build lining given a scroll mesh.

    Args:
        mesh_scroll: Scroll mesh. See load_mesh.
        rotation_strategy: How to rotate the mesh with respect to the cylinder axis.
        Defaults to RotationStrategy.DEG_45.
        voxel_size_diagonal_percent: Size of voxels as a percentage of the diagonal
        size. Defaults to 0.4.
        simplify_max_error_diagonal_percent: Maximum simplification error as a
        percentage of the diagonal size. Defaults to 1.
        target_scale_diagonal_mm: Target diagonal size of the mesh. Defaults to None.

    Returns:
        Lining meshes for the positive and negative side, cavity meshes for the
        positive and negative side, aligned scroll mesh, cylinder radius, and height.
    """
    # TODO(akoen): support different upper and lower margin
    # NOTE(akoen): This method seems to work for all meshes. However, some things that
    # may someday cause trouble: tunnels, holes, etc.
    # Look at Meshlib GitHub "heal", "tunnel"

    # Optionally scale mesh
    if target_scale_diagonal_mm:
        scale = target_scale_diagonal_mm / mesh_scroll.computeBoundingBox().diagonal()

        logger.info("Scaling mesh by factor %.2f", scale)
        scale_mat = mm.AffineXf3f.linear(mm.Matrix3f.scale(scale))
        mesh_scroll.transform(scale_mat)
    elif mesh_scroll.computeBoundingBox().diagonal() < 5:
        logger.warning("Mesh is very small. Have you forgotten to scale?")

    voxel_size = (
        mesh_scroll.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    logger.debug(f"Initial mesh: {count_vertices(mesh_scroll)} vertices")

    # Simplify mesh
    decimate_settings = mm.DecimateSettings()
    decimate_settings.maxError = simplify_max_error_diagonal_percent / 100
    decimate_settings.packMesh = True
    mm.decimateMesh(mesh_scroll, decimate_settings)

    logger.debug(f"Simplified mesh: {count_vertices(mesh_scroll)} vertices")

    logger.info("Aligning mesh")
    points = np.array([(point.x, point.y, point.z) for point in mesh_scroll.points])
    rotation_mat, translation_mat, radius, height = alignment.fit_cylinder(
        points, cylinder_axis_tol, _debug=_debug
    )

    # Fit minimum bounding cylinder
    A = mm.Matrix3f(*[mm.Vector3f(*d) for d in rotation_mat])
    mesh_scroll.transform(mm.AffineXf3f(A, mm.Vector3f(*translation_mat)))

    # Optimize Rotation
    vertices = mn.getNumpyVerts(mesh_scroll)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    cov_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices].T

    match rotation_strategy:
        case RotationStrategy.NONE:
            pass
        case RotationStrategy.SECOND_PRINCIPAL:
            # NOTE(akoen): Technically we should project the principal component
            rotation = np.arccos(np.dot(principal_components[1], [1, 0, 0]))
            mesh_scroll.transform(affine_rotation(rotation))

        case RotationStrategy.THIRD_PRINCIPAL:
            # NOTE(akoen): Technically we should project the principal component
            rotation = np.arccos(np.dot(principal_components[2], [1, 0, 0]))
            mesh_scroll.transform(affine_rotation(rotation))

        case RotationStrategy.DEG_45:
            # NOTE(akoen): Technically we should project the principal component
            rotation = np.arccos(np.dot(principal_components[2], [1, 0, 0]))
            rotation += np.pi / 4
            mesh_scroll.transform(affine_rotation(rotation))

    # Offset 2mm
    logger.info("Offsetting mesh")
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    mesh_offset_2mm = mm.offsetMesh(mesh_scroll, offset=2, params=params)

    logger.debug(f"Offset mesh: {count_vertices(mesh_offset_2mm)} vertices")

    # Split along YZ plane
    logger.info("Splitting mesh")
    cut_plane = mm.Plane3f(mm.Vector3f(0, 1, 0), 0)
    hole_edges_pos = mm.UndirectedEdgeBitSet()
    hole_edges_neg = mm.UndirectedEdgeBitSet()
    split_mesh_pos = _copy_meshlib(mesh_offset_2mm)
    mm.trimWithPlane(split_mesh_pos, cut_plane, outCutEdges=hole_edges_pos)
    split_mesh_neg = _copy_meshlib(mesh_offset_2mm)
    mm.trimWithPlane(split_mesh_neg, -cut_plane, outCutEdges=hole_edges_neg)

    logger.debug(f"Split mesh pos: {count_vertices(split_mesh_pos)} vertices")

    # Remove overhangs
    logger.info("Removing overhangs")
    cavity_mesh_pos = _copy_meshlib(split_mesh_pos)
    cavity_mesh_neg = _copy_meshlib(split_mesh_neg)
    extrusion_dir = mm.Vector3f(0, 1, 0)
    extrusion_distance = 0
    mm.fixUndercuts(cavity_mesh_pos, extrusion_dir, voxel_size, extrusion_distance)
    mm.fixUndercuts(cavity_mesh_neg, -extrusion_dir, voxel_size, extrusion_distance)

    logger.debug(f"Cavity_mesh_pos: {count_vertices(cavity_mesh_pos)} vertices")

    logger.info("Building lining")
    cavity_mesh_pos_offset = mm.offsetMesh(cavity_mesh_pos, offset=2, params=params)
    cavity_mesh_neg_offset = mm.offsetMesh(cavity_mesh_neg, offset=2, params=params)
    mm.trimWithPlane(cavity_mesh_pos_offset, cut_plane, outCutEdges=hole_edges_pos)
    mm.trimWithPlane(cavity_mesh_neg_offset, -cut_plane, outCutEdges=hole_edges_neg)

    fill_hole_params = mm.FillHoleParams()
    edges_pos = mm.std_vector_Id_EdgeTag(hole_edges_pos)
    edges_neg = mm.std_vector_Id_EdgeTag(hole_edges_neg)
    mm.fillHoles(cavity_mesh_pos_offset, edges_pos, fill_hole_params)
    mm.fillHoles(cavity_mesh_neg_offset, edges_neg, fill_hole_params)

    lining_mesh_pos = mm.voxelBooleanSubtract(
        cavity_mesh_pos_offset, cavity_mesh_pos, voxel_size
    )
    lining_mesh_neg = mm.voxelBooleanSubtract(
        cavity_mesh_neg_offset, cavity_mesh_neg, voxel_size
    )

    logger.debug(f"Lining mesh pos: {count_vertices(lining_mesh_pos)} vertices")

    return (
        lining_mesh_pos,
        lining_mesh_neg,
        cavity_mesh_pos,
        cavity_mesh_neg,
        mesh_scroll,
        radius,
        height,
    )


def combine_case_lining(
    case_mesh: mm.Mesh,
    cavity_mesh,
    lining_mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.4,
):
    voxel_size = (
        lining_mesh.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    combined_mesh = mm.voxelBooleanSubtract(case_mesh, cavity_mesh, voxel_size)
    combined_mesh = mm.voxelBooleanUnite(combined_mesh, lining_mesh, voxel_size)
    return combined_mesh
