import logging
from build123d import *

logger = logging.getLogger(__name__)


def divider_curve(case):
    pts = [
        (-case.lining_outer_radius, 0),
        (-case.lining_outer_radius / 2, case.square_height_mm),
        (0, 0),
        (case.lining_outer_radius / 2, -case.square_height_mm),
        (case.lining_outer_radius, 0),
    ]
    ln1 = ThreePointArc(pts[0], pts[1], pts[2])
    ln2 = ThreePointArc(pts[2], pts[3], pts[4])
    return ln1 + ln2


def divider_solid(case):
    with BuildPart() as divider:
        with BuildSketch():
            with BuildLine() as divider_ln:
                ln1 = Line(
                    (-case.square_loft_radius * 2, 0),
                    (-case.lining_outer_radius, 0),
                )
                ln2 = divider_curve(case)
                ln3 = Line(
                    (case.lining_outer_radius, 0),
                    (case.square_loft_radius * 2, 0),
                )

                ln4 = Line(
                    (case.square_loft_radius * 2, 0),
                    (case.square_loft_radius * 2, case.square_loft_radius * 2),
                )
                ln5 = Line(
                    (case.square_loft_radius * 2, case.square_loft_radius * 2),
                    (-case.square_loft_radius * 2, case.square_loft_radius * 2),
                )
                ln6 = Line(
                    (-case.square_loft_radius * 2, case.square_loft_radius * 2),
                    (-case.square_loft_radius * 2, 0),
                )
            make_face()
        extrude(amount=case.cylinder_height + case.square_height_mm * 2)

    return divider
