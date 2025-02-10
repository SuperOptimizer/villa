from dataclasses import dataclass
from build123d import *
import meshlib.mrmeshpy as mm
import numpy as np

@dataclass
class ScrollCase:
    scroll_height: float
    scroll_radius: float
    upper_margin: float = 20
    lower_margin: float = 20
    radial_margin: float = 4
    lining_thickness: float = 2
    scroll_offset: float = 2
    case_thickness: float = 2
    base_radius: float = 65
    base_height: float = 10

    @property
    def lining_outer_radius(self):
        return self.scroll_radius + self.scroll_offset + self.lining_thickness

    @property
    def cylinder_height(self):
        return self.scroll_height + self.lower_margin + self.upper_margin

    @property
    def cylinder_inner_radius(self):
        return self.lining_outer_radius + self.radial_margin

    @property
    def cylinder_outer_radius(self):
        return self.lining_outer_radius + self.radial_margin + self.case_thickness

    @property
    def cylinder_bottom(self):
        return -(self.lower_margin)

    @property
    def cap_bottom(self):
        return self.cylinder_bottom - self.case_thickness


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

def build_case(case: ScrollCase) -> tuple[Solid, Solid]:
    honeycomb = None
    honeycomb = honeycomb_cylinder(
        case.cylinder_inner_radius, case.cylinder_height, case.case_thickness, gap=3.0
    )

    with BuildPart(Location((0, 0, case.cylinder_bottom))) as case_part:
        if honeycomb:
            add(honeycomb)

        # Bottom Cap
        Cylinder(
            case.cylinder_outer_radius,
            case.case_thickness,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        # Top Cap
        with Locations((0, 0, case.scroll_height + case.lower_margin + case.upper_margin)):
            Cylinder(
                case.cylinder_outer_radius,
                case.case_thickness,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

        # Parting rect
        Box(
            2 * (case.cylinder_outer_radius),
            2 * case.case_thickness,
            case.scroll_height + case.lower_margin + case.upper_margin,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Alignment rings
        with Locations(
            [
                (0.0, 0.0, x)
                for x in np.arange(case.lower_margin, case.cylinder_height, 50.0)
            ]
        ):
            add(alignment_ring(case.cylinder_inner_radius))

        split(bisect_by=Plane.XZ, keep=Keep.BOTH)


    with BuildPart(Location((0, 0, case.cap_bottom))) as base_part:
        cyl = Cylinder(
            case.base_radius,
            case.base_height,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )

        with BuildSketch(cyl.faces().sort_by()[0]) as sk1:
            Rectangle(5, 5)
            with Locations((0, -20)):
                Text(f"{case.scroll_radius:.2f}Rx{case.scroll_height:.2f}Z", 10)
        extrude(amount=-5, mode=Mode.SUBTRACT)

    with BuildPart(mode=Mode.PRIVATE) as nubs:
        with Locations((0, 0, -case.lower_margin / 2)):
            with Locations((10, 0, 0)):
                Box(4, 4, 4)
            with Locations((-10, 0, 0)):
                Box(4, 4, 4, rotation=(0, 45, 0))

    left = case_part.solids()[0] + base_part.solids()[0] + nubs.solids()
    right = case_part.solids()[1] - nubs.solids()

    return left, right