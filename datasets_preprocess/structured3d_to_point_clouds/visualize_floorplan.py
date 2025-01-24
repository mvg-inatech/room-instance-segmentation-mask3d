"""
Most parts of this code were taken from the Structured3D paper:
Jia Zheng and Junfei Zhang and Jing Li and Rui Tang and Shenghua Gao and Zihan Zhou / Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling / Proceedings of The European Conference on Computer Vision (ECCV) / 2020
See https://github.com/bertjiazheng/Structured3D/blob/master/visualize_floorplan.py

The original Structured3D paper code is under the following license:
    MIT License

    Copyright (c) 2019 Structured3D Group

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import argparse
import json
import os
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

semantics_cmap = {
    "undefined": "#bbb",  # Dark gray
    "outwall": "#ddd",  # Light gray
    "living room": "#FF9999",  # Pastel red
    "kitchen": "#99CCCC",  # Pastel green
    "bedroom": "#FFFF99",  # Pastel yellow
    "bathroom": "#99CCFF",  # Pastel blue
    "balcony": "#FFCC99",  # Pastel orange
    "corridor": "#CC99FF",  # Pastel purple
    "dining room": "#99FFFF",  # Pastel cyan
    "study": "#FF99CC",  # Pastel pink
    "studio": "#CCFF99",  # Pastel lime
    "store room": "#FFCC99",  # Pastel peach
    "garden": "#99FFCC",  # Pastel teal
    "laundry room": "#CC99FF",  # Pastel lavender
    "office": "#FFCC99",  # Pastel coral
    "basement": "#FFFF99",  # Pastel lemon
    "garage": "#CC9999",  # Pastel brown
    "door": "#CCCC99",  # Pastel olive
    "window": "#FFCC99",  # Pastel apricot
}


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices"""
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def visualize_floorplan(args):
    """visualize floorplan"""
    with open(os.path.join(args.data_root, f"scene_{args.scene:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)

    if not args.floor_only:
        with open(os.path.join(args.data_root, f"scene_{args.scene:05d}", "bbox_3d.json")) as file:
            boxes = json.load(file)

    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos["semantics"]:
        for planeID in semantic["planeID"]:
            if annos["planes"][planeID]["type"] == "floor":
                planes.append({"planeID": planeID, "type": semantic["type"]})

        if semantic["type"] == "outwall":
            outerwall_planes = semantic["planeID"]

    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc["coordinate"] for junc in annos["junctions"]])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[0].tolist()
        junction_pairs = [np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane["type"]])
        # print(plane["type"])

    outerwall_floor = []
    if not args.floor_only:
        for planeID in outerwall_planes:
            lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
            lineIDs = np.setdiff1d(lineIDs, lines_holes)
            junction_pairs = [np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist() for lineID in lineIDs]
            for start, end in junction_pairs:
                if start in junction_floor and end in junction_floor:
                    outerwall_floor.append([start, end])

        outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
        polygons.append([outerwall_polygon[0], "outwall"])

    matplotlib.use("PDF")
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    for polygon, poly_type in polygons:
        polygon = np.array(
            polygon
            + [
                polygon[0],
            ]
        )
        polygon = Polygon(junctions[polygon])
        plot_polygon(polygon, ax=ax, add_points=False, facecolor=semantics_cmap[poly_type], linewidth=0, alpha=1)

    if not args.floor_only:
        for bbox in boxes:
            basis = np.array(bbox["basis"])
            coeffs = np.array(bbox["coeffs"])
            centroid = np.array(bbox["centroid"])

            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners = corners[[0, 1, 2, 3, 0], :2]

            polygon = Polygon(corners)
            plot_polygon(polygon, ax=ax, add_points=False, facecolor=colors.rgb2hex(np.random.rand(3)), alpha=0.5)

    plt.axis("equal")
    plt.axis("off")
    plt.title(f"Scene {args.scene:05d}")

    # Do not show "outwall" in legend if it is not plotted
    if not args.floor_only:
        legend_classes = semantics_cmap
    else:
        legend_classes = {room_type: color for room_type, color in semantics_cmap.items() if room_type != "outwall"}

    legend_handles = [mpatches.Patch(color=color, label=room_type) for room_type, color in legend_classes.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1))

    # plt.show()
    fig.savefig(f"scene_{args.scene:05d}.pdf", bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D Floorplan Visualization")
    parser.add_argument("--data_root", required=True, help="Path to the dataset", metavar="DIR")
    parser.add_argument("--scene", required=True, help="scene id", type=int)
    parser.add_argument("--floor_only", default=False, action="store_true", help="Only plot floor polygons, no furniture or other objects")
    return parser.parse_args()


def main():
    args = parse_args()

    visualize_floorplan(args)


if __name__ == "__main__":
    main()
