import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from build123d import *

logger = logging.getLogger(__name__)


@dataclass
class ScrollCase:
    scroll_height_mm: float
    scroll_radius_mm: float
    upper_margin_mm: float = 20
    lower_margin_mm: float = 20
    radial_margin_mm: float = 4
    lining_thickness_mm: float = 2
    scroll_offset_mm: float = 2
    case_thickness_mm: float = 2
    mount_disc_diameter_mm: float = 112.5
    mount_disc_height_mm: float = 12.5
    mount_disc_hole_depth_mm: float = 5.75
    mount_disc_hole_diameter_mm = 6.8
    mount_disc_box_height_mm: float = 7
    mount_disc_box_width_mm: float = 13.5
    alignment_ring_spacing_mm: Optional[float] = (
        50  # Relative to the base of the scroll
    )
    label_line_1: str = ""
    label_line_2: str = ""
    nub_size_mm: float = 4
    nub_depth_mm: float = 1.5
    nub_margin_mm: float = 0.5
    square_height_mm: float = 20
    square_edge_fillet: float = 5
    oring_width: float = 5
    oring_depth: float = 2

    @property
    def lining_outer_radius(self):
        return self.scroll_radius_mm + self.scroll_offset_mm + self.lining_thickness_mm

    @property
    def cylinder_height(self):
        return self.scroll_height_mm + self.lower_margin_mm + self.upper_margin_mm

    @property
    def cylinder_inner_radius(self):
        return self.lining_outer_radius + self.radial_margin_mm

    @property
    def cylinder_outer_radius(self):
        return self.lining_outer_radius + self.radial_margin_mm + self.case_thickness_mm

    @property
    def cylinder_bottom(self):
        return -(self.lower_margin_mm)

    @property
    def square_loft_radius(self):
        return max(self.mount_disc_diameter_mm / 2, self.cylinder_outer_radius)


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

    all_hexes = list(all_hexes)

    cyl = Cylinder(r + t, h, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(
        r, h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )

    honey_cyl = cyl - all_hexes

    return honey_cyl


def build_case(case: ScrollCase) -> tuple[Solid, Solid]:
    """Build the scroll case.

    Args:
        case (ScrollCase): The scroll case parameters.

    Returns:
        tuple[Solid, Solid]: The left and right halves of the scroll case.
    """
    logger.info(
        f"Constructing case with scroll radius: {case.scroll_radius_mm}, height: {case.scroll_height_mm}"
    )

    honeycomb = None
    honeycomb = honeycomb_cylinder(
        case.cylinder_inner_radius,
        case.cylinder_height,
        case.case_thickness_mm,
        gap=3.0,
    )

    with BuildPart(Location((0, 0, case.cylinder_bottom))) as case_part:
        if honeycomb:
            add(honeycomb)

        # Parting rect
        Box(
            2 * (case.cylinder_outer_radius),
            2 * case.case_thickness_mm,
            case.scroll_height_mm + case.lower_margin_mm + case.upper_margin_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Alignment rings
        if case.alignment_ring_spacing_mm:
            with Locations(
                [
                    (0.0, 0.0, x)
                    for x in np.arange(
                        case.lower_margin_mm,
                        case.cylinder_height,
                        case.alignment_ring_spacing_mm,
                    )
                ]  # type: ignore
            ):
                add(alignment_ring(case.cylinder_inner_radius))

        # Top Cap
        with BuildSketch(
            Location((0, 0, case.scroll_height_mm + case.upper_margin_mm))
        ) as l2:
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), 20)

        extrude(amount=case.square_height_mm)

        with BuildSketch(
            Location(
                (
                    0,
                    0,
                    case.scroll_height_mm
                    + case.upper_margin_mm
                    + case.square_height_mm / 2,
                )
            )
        ):
            Rectangle(
                2 * case.square_loft_radius,
                2 * case.square_loft_radius,
            )
            with BuildSketch(mode=Mode.SUBTRACT):
                r = Rectangle(
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                    # mode=Mode.SUBTRACT,
                )
                fillet(r.vertices(), 20)

        extrude(amount=case.oring_width / 2, both=True, mode=Mode.SUBTRACT)

        # Bottom Cap

        with BuildSketch(Location((0, 0, -case.lower_margin_mm))) as l2:
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), 20)

        extrude(amount=-case.square_height_mm)
        with BuildSketch(
            Location(
                (
                    0,
                    0,
                    -case.lower_margin_mm - case.square_height_mm / 2,
                )
            )
        ):
            Rectangle(
                2 * case.square_loft_radius,
                2 * case.square_loft_radius,
            )
            with BuildSketch(mode=Mode.SUBTRACT):
                r = Rectangle(
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                )
                fillet(r.vertices(), 20)

        extrude(amount=-case.oring_width / 2, both=True, mode=Mode.SUBTRACT)

        split(bisect_by=Plane.XZ, keep=Keep.BOTH)

    with BuildPart(
        Location((0, 0, -case.lower_margin_mm - case.square_height_mm))
    ) as mount_disc:
        cyl = Cylinder(
            case.mount_disc_diameter_mm / 2,
            case.mount_disc_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )

        # Alignment Cube
        with BuildSketch(cyl.faces().sort_by()[0]):
            Rectangle(case.mount_disc_box_width_mm, case.mount_disc_box_width_mm)
        extrude(amount=-case.mount_disc_box_height_mm, mode=Mode.SUBTRACT)

        # Text
        with BuildSketch(cyl.faces().sort_by()[0]):
            with Locations((0, 40)):
                Text(case.label_line_1, 8)
            with Locations((0, 30)):
                Text(case.label_line_2, 8)
            with Locations((0, 20)):
                Text(
                    f"{case.scroll_radius_mm * 2:.2f}Dx{case.scroll_height_mm:.2f}Z", 8
                )
        extrude(amount=-5, mode=Mode.SUBTRACT)

        # Bolt holes
        with Locations(
            (0, case.mount_disc_diameter_mm / 2, -case.mount_disc_height_mm / 2)
        ):
            Cylinder(
                case.mount_disc_hole_diameter_mm / 2,
                2 * case.mount_disc_hole_depth_mm,
                rotation=(90, 0, 0),
                mode=Mode.SUBTRACT,
            )
        with Locations(
            (0, -case.mount_disc_diameter_mm / 2, -case.mount_disc_height_mm / 2)
        ):
            Cylinder(
                case.mount_disc_hole_diameter_mm / 2,
                2 * case.mount_disc_hole_depth_mm,
                rotation=(-90, 0, 0),
                mode=Mode.SUBTRACT,
            )

    with BuildPart(mode=Mode.PRIVATE) as nubs:
        with Locations((0, 0, -case.lower_margin_mm / 2)):
            with Locations((10, 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-10, 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    with BuildPart() as hollows:
        with Locations((0, 0, -case.lower_margin_mm / 2)):
            with Locations((10, 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-10, 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    left = case_part.solids()[0] + mount_disc.solids()[0] + nubs.solids()
    right = case_part.solids()[1] - hollows.solids()

    return left, right
