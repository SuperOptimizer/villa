import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from build123d import *

logger = logging.getLogger(__name__)


@dataclass
class ScrollCase:
    # Z=0 is the scroll bottom

    # Scroll dimensions defined per-scroll
    scroll_height_mm: float
    scroll_radius_mm: float

    # Labels defined per-scroll (remaining params fixed across scrolls)
    label_line_1: str = ""
    label_line_2: str = ""

    # Gap/offset between scroll and lining wall interior
    lining_offset_mm: float = 2
    # Wall thicknesses
    wall_thickness_mm: float = 2

    # Between lining exterior and top/bottom caps
    upper_margin_mm: float = 3
    lower_margin_mm: float = 3

    # Between lining exterior and cylinder wall interior
    # Just enough space to not touch/create trap points for glass beads
    # Otherwise want to minimize exterior diameter
    radial_margin_mm: float = 0.5

    # Mounting disc dimensions
    mount_disc_diameter_mm: float = 112.5
    mount_disc_height_mm: float = 10
    mount_disc_hole_depth_mm: float = 5.75
    mount_disc_hole_diameter_mm: float = 6.8
    mount_disc_box_height_mm: float = 7
    mount_disc_box_width_mm: float = 13.5

    # Alignment nubs
    nub_size_mm: float = 3
    nub_depth_mm: float = 2
    nub_margin_mm: float = 1

    # Square caps
    square_height_mm: float = 10
    square_edge_fillet: float = 20
    right_cap_buffer: float = 1
    # Based on M4 bolts
    cap_bolt_hole_diameter_mm: float = 5
    cap_bolt_counter_bore_diameter_mm: float = 8
    cap_bolt_counter_bore_depth_mm: float = 2
    cap_bolt_nut_diameter_mm: float = 9
    cap_bolt_nut_depth_mm: float = 3.5

    # Text properties
    text_font_size: float = 8
    text_depth_mm: float = 0.5

    @property
    def lining_outer_radius(self):
        return self.scroll_radius_mm + self.lining_offset_mm + self.wall_thickness_mm

    @property
    def cylinder_height(self):
        return (
            self.scroll_height_mm
            + 2 * self.lining_offset_mm
            + 2 * self.wall_thickness_mm
            + self.lower_margin_mm
            + self.upper_margin_mm
        )

    @property
    def cylinder_inner_radius(self):
        return self.lining_outer_radius + self.radial_margin_mm

    @property
    def cylinder_outer_radius(self):
        return self.cylinder_inner_radius + self.wall_thickness_mm

    @property
    def cylinder_bottom(self):
        return -self.lining_offset_mm - self.wall_thickness_mm - self.lower_margin_mm

    @property
    def square_loft_radius(self):
        return max(self.mount_disc_diameter_mm / 2, self.cylinder_outer_radius)

    @property
    def cylinder_outer_diameter(self):
        return 2 * self.cylinder_outer_radius

    @property
    def lining_interior_height(self):
        return 2 * self.lining_offset_mm + self.scroll_height_mm

    @property
    def cylinder_top_to_lining_bottom(self):
        return self.square_height_mm + self.lower_margin_mm + self.wall_thickness_mm


def hex_nut(diameter_mm: float, depth_mm: float):
    with BuildPart() as hex_part:
        with BuildSketch() as hex_sketch:
            RegularPolygon(radius=diameter_mm / 2, side_count=6)
        extrude(amount=depth_mm)

    return hex_part


def cap(case: ScrollCase):
    with BuildPart() as cap_part:
        with BuildSketch():
            # Main rectangle
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), case.square_edge_fillet)

            # Left bolt protrusion
            with Locations((-case.square_loft_radius, 0)):
                r2 = Rectangle(
                    case.square_height_mm,
                    case.square_height_mm * 2,
                    align=(Align.MAX, Align.CENTER),
                )
                fillet(r2.vertices(), case.square_height_mm / 2)

            # Right bolt protrusion
            with Locations((case.square_loft_radius, 0)):
                r3 = Rectangle(
                    case.square_height_mm,
                    case.square_height_mm * 2,
                    align=(Align.MIN, Align.CENTER),
                )
                fillet(r3.vertices(), case.square_height_mm / 2)
        extrude(amount=case.square_height_mm)

        # Bolt holes
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                0,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                0,
                case.square_height_mm / 2,
            ),
        ):
            Cylinder(
                case.cap_bolt_hole_diameter_mm / 2,
                4 * case.square_height_mm,
                rotation=(90, 0, 0),
                mode=Mode.SUBTRACT,
            )

        # Bolt head cutouts
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                case.square_height_mm - case.cap_bolt_counter_bore_depth_mm,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                case.square_height_mm - case.cap_bolt_counter_bore_depth_mm,
                case.square_height_mm / 2,
            ),
        ):
            Cylinder(
                case.cap_bolt_counter_bore_diameter_mm / 2,
                2 * case.square_height_mm,
                mode=Mode.SUBTRACT,
                rotation=(90, 0, 0),
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

        # Hexagonal nut cutouts
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                -case.square_height_mm + case.cap_bolt_nut_depth_mm,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                -case.square_height_mm + case.cap_bolt_nut_depth_mm,
                case.square_height_mm / 2,
            ),
        ):
            hex = hex_nut(
                case.cap_bolt_nut_diameter_mm, case.cap_bolt_nut_depth_mm + 10
            )
            add(hex, mode=Mode.SUBTRACT, rotation=(90, 0, 0))

    return cap_part


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

    with BuildPart(Location((0, 0, case.cylinder_bottom))) as case_part:
        Cylinder(
            case.cylinder_inner_radius,
            case.cylinder_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Top and bottom caps
        with Locations((0, 0, case.cylinder_height), (0, 0, -case.square_height_mm)):
            add(cap(case))

        # Text
        top_face = case_part.faces().sort_by(Axis.Z)[-1]
        with BuildSketch(top_face):
            with Locations((0, 40)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 40 - case.square_loft_radius)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 30 - case.square_loft_radius)):
                Text(case.label_line_2, case.text_font_size)
            with Locations((0, 20 - case.square_loft_radius)):
                Text(
                    f"{case.cylinder_outer_diameter:.2f}D x {case.lining_interior_height:.2f}H",
                    case.text_font_size,
                )
        extrude(amount=-case.text_depth_mm, mode=Mode.SUBTRACT)

        split(bisect_by=Plane.XZ, keep=Keep.BOTH)

    # Base
    with BuildPart(
        Location((0, 0, case.cylinder_bottom - case.square_height_mm))
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

    # Alignment nubs
    with BuildPart(mode=Mode.PRIVATE) as nubs:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm / 2)):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

        with Locations(
            (
                0,
                0,
                case.cylinder_bottom + case.cylinder_height + case.square_height_mm / 2,
            )
        ):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    # Alignment nub hollows
    with BuildPart() as hollows:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm / 2)):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

        with Locations(
            (
                0,
                0,
                case.cylinder_bottom + case.cylinder_height + case.square_height_mm / 2,
            )
        ):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    # Extra space at bottom of left case half
    with BuildPart() as left_bottom_margin:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm)):
            Box(
                case.square_loft_radius * 2,
                case.square_loft_radius * 2,
                case.right_cap_buffer,
                align=(Align.CENTER, Align.MIN, Align.MAX),
            )

    left = (
        case_part.solids()[0]
        + mount_disc.solids()[0]
        + nubs.solids()
        - left_bottom_margin.solids()
    )
    right = case_part.solids()[1] - hollows.solids()

    return left, right
