from build123d import *
import meshlib.mrmeshpy as mm
import numpy as np


def generic_mount_disc() -> Solid:
    with BuildPart() as mount:
        with BuildSketch() as sketch:
            Circle(65)
            Rectangle(20, 20, mode=Mode.SUBTRACT)
        extrude(amount=10)
    return mount


def alignment_ring(radius: float):
    """Generate an alignment ring.

    Args:
        radius: Minor radius

    Returns:
        Alignment ring
    """
    with BuildPart() as part:
        Cylinder(radius + 2, 3)
        Cylinder(radius + 3, 1)
        Cylinder(radius, 3, mode=Mode.SUBTRACT)

    return part


def honeycomb_cylinder(
    r: float, h: float, t: float, nr: float = 6, gap: float = 2
) -> Compound:
    """Build a hollow cylinder with honeycomb cutouts.

    Args:
        r: Cylinder minor radius.
        h: Cylinder height.
        t: Cylinder thickness.
        nr: Number of radial cutouts. Defaults to 6.
        gap: Gap between cutouts. Defaults to 1.

    Returns:
        Hhoneycomb cylinder.
    """
    # NOTE(akoen): This is not quite right since this assumes that the plane is flat,
    # which it is not.
    width = (4 * np.pi * r) / (3 * nr) - 2 * gap / np.sqrt(3)
    spacing = 3 / 2 * width + np.sqrt(3) * gap

    height = width * np.cos(np.pi / 6)

    nz = int(np.ceil(2 * h / (height + gap))) + 1

    all_hexes = []
    for iz in range(nz):
        start_angle = 360 / (2 * nr) * (iz % 2)
        hex = (
            (Plane.YZ * RegularPolygon(width / 2, 6, major_radius=True))
            .face()
            .translate((0, 0, iz * (height / 2 + gap / 2)))
        )
        hex = extrude(hex, 3, both=True)
        hexes = PolarLocations(r, nr, start_angle=start_angle) * hex
        all_hexes.extend(hexes)

    all_hexes = [h for h in all_hexes]

    cyl = Cylinder(r + t, h, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(
        r, h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )

    honey_cyl = cyl - all_hexes

    return honey_cyl
