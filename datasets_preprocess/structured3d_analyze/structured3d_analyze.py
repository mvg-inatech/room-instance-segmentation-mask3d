"""
Some parts of this file were taken from the Structured3D paper:
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
import numpy as np
from shapely.geometry import Polygon
import statistics


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


def analyze_scene(args, scene: str):
    """visualize floorplan"""
    with open(os.path.join(args.data_root, scene, "annotation_3d.json")) as file:
        annos = json.load(file)

    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos["semantics"]:
        for planeID in semantic["planeID"]:
            if annos["planes"][planeID]["type"] == "floor":
                planes.append({"planeID": planeID, "type": semantic["type"]})

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

    rooms = []

    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    for polygon, poly_type in polygons:
        polygon = np.array(
            polygon
            + [
                polygon[0],
            ]
        )
        polygon = Polygon(junctions[polygon])

        no_room_id_semantic_types = ["door", "window", "outwall"]

        if poly_type not in no_room_id_semantic_types:
            rooms.append({"type": poly_type, "area": polygon.area / 1000000})  # area in m^2

    return rooms


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D dataset analysis")
    parser.add_argument("--data_root", required=True, help="Path to the dataset", metavar="DIR")
    parser.add_argument(
        "--scenes_file",
        default=None,
        type=str,
        help="Path to a file that contains one scene per line that should be downsampled. Either this argument or --scene needs to be specified.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scenes = []
    with open(args.scenes_file) as file:
        for line in file.readlines():
            line_cleaned = line.strip().replace("\n", "")
            if len(line_cleaned) > 0:
                scenes.append(line_cleaned)

    print(f"Analyzing {len(scenes)} scenes...")

    scenes_with_undefined_rooms = []
    total_num_undefined_rooms = 0
    scenes_area = []
    scenes_undefined_area = []
    undefined_rooms_area = []

    ROOM_UNDEFINED_AREA_THRESHOLD = 1  # in m^2
    scenes_undefined_thresholded_area = []
    scenes_num_rooms = []

    for scene in scenes:
        scene_rooms = analyze_scene(args, scene)

        scene_rooms_undefined = [r for r in scene_rooms if r["type"] == "undefined"]

        total_num_undefined_rooms += len(scene_rooms_undefined)
        if len(scene_rooms_undefined) > 0:
            scenes_with_undefined_rooms.append((scene, len(scene_rooms_undefined)))

        scene_total_area = 0
        scene_undefined_area = 0
        scene_undefined_thresholded_area = 0

        scene_num_rooms = 0

        for r in scene_rooms:
            scene_total_area += r["area"]

            if r["type"] == "undefined":
                scene_undefined_area += r["area"]
                undefined_rooms_area.append(r["area"])

                if r["area"] > ROOM_UNDEFINED_AREA_THRESHOLD:
                    scene_undefined_thresholded_area += r["area"]
                    scene_num_rooms +=1
            else:
                scene_num_rooms +=1

        scenes_area.append(scene_total_area)
        scenes_undefined_area.append(scene_undefined_area)
        scenes_undefined_thresholded_area.append(scene_undefined_thresholded_area)
        scenes_num_rooms.append(scene_num_rooms)

    print(f"")
    print(f"Results:")
    print(f"num_scenes_with_undefined_rooms: {len(scenes_with_undefined_rooms)}")
    print(f"total_num_undefined_rooms: {total_num_undefined_rooms}")
    print(f"First few scenes with undefined rooms: {scenes_with_undefined_rooms[:20]}")
    print(f"Average number of undefined rooms per scene: {total_num_undefined_rooms / len(scenes)}")
    print("")

    assert len(scenes_area) == len(scenes)
    assert len(scenes_undefined_area) == len(scenes)

    avg_scene_area = statistics.mean(scenes_area)
    print(f"Average scene area: {avg_scene_area}")

    avg_scene_undefined_area = statistics.mean(scenes_undefined_area)
    print(f"Average scene undefined area: {avg_scene_undefined_area}, this is fraction {avg_scene_undefined_area / avg_scene_area}")

    avg_scene_undefined_area_thresholded = statistics.mean(scenes_undefined_thresholded_area)
    print(
        f"Average scene undefined area, thresholded (only consider a room as undefined if it is larger than {ROOM_UNDEFINED_AREA_THRESHOLD} m^2): {avg_scene_undefined_area_thresholded}, this is fraction {avg_scene_undefined_area_thresholded / avg_scene_area}"
    )

    avg_undefined_rooms_area = statistics.mean(undefined_rooms_area)
    print(f"The average undefined room (no threshold) has area: {avg_undefined_rooms_area}")
    median_undefined_rooms_area = statistics.median(undefined_rooms_area)
    print(f"The median undefined room (no threshold) has area: {median_undefined_rooms_area}")

    print("")
    print(f"No. rooms per scene: min {min(scenes_num_rooms)}, max {max(scenes_num_rooms)}, avg {statistics.mean(scenes_num_rooms)}, median {statistics.median(scenes_num_rooms)}")

    # Note: "rooms" are just floor-level polygons excluding the `no_room_id_semantic_types` types. I assume that they are rooms. Sometimes they are very tiny.
    # TODO use lower bound 1m^2?


if __name__ == "__main__":
    main()
