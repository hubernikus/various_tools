"""
Draw 3D -> 2D directional space using matplotlib & tools
"""
from math import pi

import numpy as np

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import *


def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
        [np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors]
    )
    codes = np.concatenate(
        [ring_coding(polygon.exterior)] + [ring_coding(r) for r in polygon.interiors]
    )
    return Path(vertices, codes)


def circular_space_setup(
    ax,
    circ_radius: float = pi / 2,
    space_radius: float = pi,
    circle_background_color="white",
    outer_boundary_color=None,
):
    """Draws circle space on axis with with (dotted) circle-radius 'circ-radius' and
    outer space-boundaries of radius 'space_radius'.
    """
    ax.spines.left.set_position("center")
    ax.spines.right.set_color("none")
    ax.spines.bottom.set_position("center")
    ax.spines.top.set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # circ_var = np.linspace(0, 2*pi, 100)
    # rad = pi
    # ax.plot(np.cos(circ_var)*circ_radius, np.sin(circ_var)*circ_radius, 'k--')

    if outer_boundary_color is not None:
        polygon = (
            Point(0, 0).buffer(space_radius).difference(Point(0, 0).buffer(circ_radius))
        )
        path = pathify(polygon)
        patch = PathPatch(path, facecolor=outer_boundary_color, linewidth=0)
        ax.add_patch(patch)

    polygon = Point(0, 0).buffer(circ_radius)
    path = pathify(polygon)
    patch = PathPatch(
        path, facecolor=circle_background_color, linewidth=1, linestyle="--"
    )
    ax.add_patch(patch)

    # polygon = Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    polygon = Point(0, 0).buffer(10.0).difference(Point(0, 0).buffer(space_radius))
    path = pathify(polygon)
    patch = PathPatch(path, facecolor="white", linewidth=0)
    ax.add_patch(patch)

    ax.axis("equal")

    dx = dy = 0.1
    outer_circle_limits = space_radius
    ax.set_xlim([-outer_circle_limits - dx, outer_circle_limits + dx])
    ax.set_ylim([-outer_circle_limits - dy, outer_circle_limits + dy])
