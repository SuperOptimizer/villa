import logging
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, Optional

import build123d as bd
import meshlib.mrmeshnumpy as mn
import meshlib.mrmeshpy as mm
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


def mesh_smooth_denoise(mesh: mm.Mesh, gamma: float = 20) -> mm.Mesh:
    """Smooth a mesh with a denoise operation.

    Denoises a mesh per the paper 'Mesh Denoising via a Novel Mumford-Shah Framework'.

    Args:
        gamma: Smoothing amount.
    """
    logger.info(f"Smoothing mesh with denoise, gamma = {gamma}")
    mesh_smooth = _copy_meshlib(mesh)
    settings = mm.DenoiseViaNormalsSettings()
    settings.gamma = gamma
    mm.meshDenoiseViaNormals(mesh_smooth)
    return mesh_smooth


def mesh_smooth_shrink_expand(
    mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.4,
    amount_mm: float = 2,
    shrink_first: bool = True,
):
    """Smooth a mesh with a shrink/expand operation.

    Shrinking first smoothes convexities, expanding first smoothes concavities.
    """
    voxel_size = (
        mesh.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    mesh_smooth = _copy_meshlib(mesh)
    logger.info(f"Smoothing mesh with a shrink/expand offset of {amount_mm} mm")
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    mesh_smooth = mm.offsetMesh(
        mesh_smooth,
        offset=-amount_mm if shrink_first else amount_mm,
        params=params,  # type: ignore
    )
    mesh_smooth = mm.offsetMesh(
        mesh_smooth,
        offset=amount_mm if shrink_first else -amount_mm,
        params=params,  # type: ignore
    )
    return mesh_smooth


def get_principal_components(mesh: mm.Mesh) -> np.ndarray:
    """Get principal components of mesh as a 3x3 array."""
    vertices = mn.getNumpyVerts(mesh)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    cov_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices].T
    return principal_components


def rotate_about_2nd_principal(mesh: mm.Mesh, rotation_rad: float) -> mm.Mesh:
    """Rotate about second principal.

    Rotate mesh so that 2nd principal component is rotatation_rad radians from the
    second (wide) principal component.
    """
    principal_components = get_principal_components(mesh)
    rotated_mesh = _copy_meshlib(mesh)

    p2 = [*principal_components[1][:2], 0]
    first_rot = np.arccos(np.dot(p2, [1, 0, 0]))
    rotated_mesh.transform(affine_rotation(first_rot + rotation_rad))
    return rotated_mesh


@dataclass
class ScrollMesh:
    """Scroll mesh processing options.

    Attributes:
        mesh_path: Path to scroll mesh.
        rotation_from_2nd_principal_rad. Rotation from the second principal vector
        about the cylinder axis. Defaults to 0.
        voxel_size_diagonal_percent: Size of voxels as a percentage of the diagonal
        size. Defaults to 0.4.
        simplify_max_error_diagonal_percent: Maximum simplification error as a
        percentage of the diagonal size. Defaults to 1.
        target_scale_diagonal_mm: Target diagonal size of the mesh. Defaults to None.
        rotation_callback: Rotation callback that accepts a mesh and returns a mesh
        smoothing_callback: Smoothing callback that accepts a mesh and returns a mesh
        smoothin_unite_with_original: Unite with the original after smoothing to ensure
        mesh does not get smaller.

    """

    mesh_path: str
    cylinder_axis_tol: float = 0.01
    voxel_size_diagonal_percent: float = 0.4
    simplify_max_error_diagonal_percent: float = 1
    target_scale_diagonal_mm: Optional[float] = None
    rotation_callback: Optional[Callable[[mm.Mesh], mm.Mesh]] = partial(
        rotate_about_2nd_principal, rotation_rad=0
    )
    smoothing_callback: Optional[Callable[[mm.Mesh], mm.Mesh]] = None
    smoothing_unite_with_original: bool = True


def build_lining(
    mesh_params: ScrollMesh,
    *,
    _debug=False,
) -> tuple[mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, mm.Mesh, float, float]:
    """Build lining given a scroll mesh.

    Args:
        mesh_params: ScrollMesh parameters.

    Returns:
        Lining meshes for the positive and negative side, cavity meshes for the
        positive and negative side, aligned scroll mesh, cylinder radius, and height.
    """
    # TODO(akoen): support different upper and lower margin
    # NOTE(akoen): This method seems to work for all meshes. However, some things that
    # may someday cause trouble: tunnels, holes, etc.
    # Look at Meshlib GitHub "heal", "tunnel"

    mesh_scroll = load_mesh(mesh_params.mesh_path)

    # Optionally scale mesh
    if mesh_params.target_scale_diagonal_mm:
        scale = (
            mesh_params.target_scale_diagonal_mm
            / mesh_scroll.computeBoundingBox().diagonal()
        )

        logger.info("Scaling mesh by factor %.2f", scale)
        scale_mat = mm.AffineXf3f.linear(mm.Matrix3f.scale(scale))
        mesh_scroll.transform(scale_mat)
    elif mesh_scroll.computeBoundingBox().diagonal() < 5:
        logger.warning("Mesh is very small. Have you forgotten to scale?")

    voxel_size = (
        mesh_scroll.computeBoundingBox().diagonal()
        * mesh_params.voxel_size_diagonal_percent
        / 100
    )

    logger.debug(f"Initial mesh: {count_vertices(mesh_scroll)} vertices")

    # Simplify mesh
    decimate_settings = mm.DecimateSettings()
    decimate_settings.maxError = mesh_params.simplify_max_error_diagonal_percent / 100
    decimate_settings.packMesh = True
    mm.decimateMesh(mesh_scroll, decimate_settings)

    logger.debug(f"Simplified mesh: {count_vertices(mesh_scroll)} vertices")

    # Fit minimum boundign cylinder
    logger.info("Aligning mesh")
    points = np.array([(point.x, point.y, point.z) for point in mesh_scroll.points])
    rotation_mat, translation_mat, radius, height = alignment.fit_cylinder(
        points, mesh_params.cylinder_axis_tol, _debug=_debug
    )

    A = mm.Matrix3f(*[mm.Vector3f(*d) for d in rotation_mat])
    mesh_scroll.transform(mm.AffineXf3f(A, mm.Vector3f(*translation_mat)))

    # Optimize Rotation
    if mesh_params.rotation_callback is not None:
        mesh_scroll = mesh_params.rotation_callback(mesh_scroll)

    # Smooth mesh
    if (sc := mesh_params.smoothing_callback) is not None:
        mesh_scroll_smoothed = sc(mesh_scroll)
        if mesh_params.smoothing_unite_with_original:
            mesh_scroll = mm.voxelBooleanUnite(
                mesh_scroll, mesh_scroll_smoothed, voxel_size
            )
        else:
            mesh_scroll = mesh_scroll_smoothed

    # Offset 2mm
    logger.info("Offsetting mesh")
    params = mm.OffsetParameters()
    params.voxelSize = voxel_size
    mesh_offset_2mm = mm.offsetMesh(mesh_scroll, offset=2, params=params)  # type: ignore

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

    # Build mesh
    logger.info("Building lining")
    cavity_mesh_pos_offset = mm.offsetMesh(cavity_mesh_pos, offset=2, params=params)  # type: ignore
    cavity_mesh_neg_offset = mm.offsetMesh(cavity_mesh_neg, offset=2, params=params)  # type: ignore
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


def combine_brep_case_lining(
    case: bd.Solid,
    cavity_mesh: mm.Mesh,
    lining_mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.01,
):
    """Combine a BRep case mesh with a lining."""
    with tempfile.NamedTemporaryFile(suffix=".stl") as temp_file:
        bd.export_stl(case, temp_file.name)
        case_mesh = load_mesh(temp_file.name)
        return combine_case_lining(
            case_mesh, cavity_mesh, lining_mesh, voxel_size_diagonal_percent
        )


def combine_case_lining(
    case_mesh: mm.Mesh,
    cavity_mesh: mm.Mesh,
    lining_mesh: mm.Mesh,
    voxel_size_diagonal_percent: float = 0.4,
):
    "Combine a case mesh with a lining."
    voxel_size = (
        lining_mesh.computeBoundingBox().diagonal() * voxel_size_diagonal_percent / 100
    )

    combined_mesh = mm.boolean(case_mesh, cavity_mesh, mm.BooleanOperation.DifferenceAB)
    combined_mesh = mm.boolean(
        combined_mesh.mesh, lining_mesh, mm.BooleanOperation.Union
    )
    return combined_mesh.mesh
