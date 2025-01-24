"""
Parts of this code were taken from the RoomFormer paper:
Y. Yue, T. Kontogianni, K. Schindler, and F. Engelmann, “Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries.” arXiv, Mar. 2023. Accessed: Mar. 19, 2024. [Online]. Available: http://arxiv.org/abs/2211.15658
See https://github.com/ywyue/RoomFormer/blob/main/data_preprocess/stru3d/PointCloudReaderPanorama.py

The RoomFormer paper code is under the following license:
    MIT License

    Copyright (c) 2022 Yuanwen Yue

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

import json
from multiprocessing import Pool
import traceback
from typing import Any
import cv2
import os
import numpy as np
import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import BufferCapStyle, BufferJoinStyle
from itertools import pairwise
from joblib import Parallel, delayed
import laspy

ROOM_ID_INVALID = 0


class PointCloudReaderPanorama:
    def __init__(
        self,
        scene_path,
        resolution="full",
        random_level=0,
        generate_color=False,
        generate_normal=False,
        debug_mode=False,
    ):
        self.debug_mode = debug_mode
        self.scene_path = scene_path
        self.random_level = random_level
        self.resolution = resolution
        self.generate_color = generate_color
        self.generate_normal = generate_normal
        sections = [p for p in os.listdir(os.path.join(scene_path, "2D_rendering"))]
        self.depth_paths = [
            os.path.join(
                *[
                    scene_path,
                    "2D_rendering",
                    p,
                    "panorama",
                    self.resolution,
                    "depth.png",
                ]
            )
            for p in sections
        ]
        self.rgb_paths = [
            os.path.join(
                *[
                    scene_path,
                    "2D_rendering",
                    p,
                    "panorama",
                    self.resolution,
                    "rgb_coldlight.png",
                ]
            )
            for p in sections
        ]
        self.normal_paths = [
            os.path.join(
                *[
                    scene_path,
                    "2D_rendering",
                    p,
                    "panorama",
                    self.resolution,
                    "normal.png",
                ]
            )
            for p in sections
        ]
        self.camera_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.camera_centers = self.read_camera_center()

        annotations_path = os.path.join(scene_path, "annotation_3d.json")

        with open(annotations_path) as file:
            self.annotations = json.load(file)

        self.polygons = self.get_polygons()

        self.semantic_type_int_map = {
            "undefined": 0,
            "living room": 1,
            "kitchen": 2,
            "bedroom": 3,
            "bathroom": 4,
            "balcony": 5,
            "corridor": 6,
            "dining room": 7,
            "study": 8,
            "studio": 9,
            "store room": 10,
            "garden": 11,
            "laundry room": 12,
            "office": 13,
            "basement": 14,
            "garage": 15,
            # Note: index 16 is left out here in favor of index 0, its meaning is "undefined" in the original Structured3D data
            "door": 17,
            "window": 18,
            "outwall": 19,
            "other": 20,
            "invalid": 21,
        }

    def get_polygons(self):
        polygons = []
        room_id_counter = 1
        for semantic in self.annotations["semantics"]:
            for plane_id in semantic["planeID"]:
                allowed_plane_types = ["floor"]
                if self.annotations["planes"][plane_id]["type"] in allowed_plane_types:
                    # IDs of the lines that belong to this plane
                    plane_line_ids = np.where(np.array(self.annotations["planeLineMatrix"][plane_id]))[0].tolist()

                    # IDs of the junctions (=points) that form this plane (=polygon)
                    junction_id_pairs = [np.where(np.array(self.annotations["lineJunctionMatrix"][line_id]))[0].tolist() for line_id in plane_line_ids]
                    # Example content: [[66, 67], [67, 68], [68, 69], [66, 69]]

                    junctions_ids = self._convert_lines_to_vertices(junction_id_pairs)
                    # Example content: [[66, 67, 68, 69]]

                    if not "wall" in allowed_plane_types:
                        assert len(junctions_ids) == 1

                    # len(junctions_ids) may be > 1 for the "wall" plane type.
                    for junction_ids in junctions_ids:
                        polygon_coordinates = []
                        for junction_id in junction_ids:
                            junctions = [j for j in self.annotations["junctions"] if j["ID"] == junction_id]
                            assert len(junctions) == 1
                            polygon_coordinates.append(junctions[0]["coordinate"])

                        polygon_coords_2d = [[coord_3d[0], coord_3d[1]] for coord_3d in polygon_coordinates]

                        polygon = {
                            "shapely_polygon_2d": Polygon(polygon_coords_2d),
                            "semantic_type": semantic["type"],
                        }

                        if polygon["semantic_type"] == "undefined":
                            if polygon["shapely_polygon_2d"].area >= 1500000:  # 1.5 m^2
                                # The polygon is assumed to be a valid room, but we don't know the type
                                polygon["semantic_type"] = "other"
                            else:
                                # The polygon is assumed to be something we don't know. Points should be discarded.
                                polygon["semantic_type"] = "invalid"

                        assert polygon["semantic_type"] != "undefined"

                        no_room_id_semantic_types = ["door", "window", "outwall", "invalid"]
                        if polygon["semantic_type"] not in no_room_id_semantic_types:
                            # Do not assign a room id (since it's not a room), but keep the type information for informational purposes.
                            # Consuming applications must filter points for having a room ID != ROOM_ID_INVALID, instead of checking the type.
                            polygon["room_id"] = room_id_counter
                            room_id_counter += 1
                        else:
                            polygon["room_id"] = ROOM_ID_INVALID

                        polygon = self._enlarge_polygon(polygon)

                        # print(f"Found polygon: {polygon}")
                        polygons.append(polygon)
        return polygons

    def _enlarge_polygon(self, polygon):
        """Enlarges a shapely polygon to prevent the missing pixels in walls problem.

        Args:
            polygon (shapely.Polygon): The polygon to enlarge.

        Returns:
            shapely.Polygon: The enlarged polygon.
        """
        enlarge_distance = 15  # Manually tuned on scene 2, 19, 20, 21
        polygon["shapely_polygon_2d"] = polygon["shapely_polygon_2d"].buffer(
            enlarge_distance, join_style=BufferJoinStyle.mitre, cap_style=BufferCapStyle.square
        )
        return polygon

    def _convert_lines_to_vertices(self, lines):
        """Converts the lines of a polygon to its vertices. Lines are pairs of vertices. Vertices are identified by an id (int).

        Args:
            lines (list): List of lists with 2 int elements each, representing the vertex ids.

        Returns:
            list: List of vertex ids (int).
        """
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

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], "r") as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers

    def _get_matching_polygons(self, shapely_point):
        matching_polygons = []
        for polygon in self.polygons:
            # Check whether the point is within the polygon formed by the lines of this plane

            # TODO This piece of code is too slow!
            # Begin slow code
            if polygon["shapely_polygon_2d"].contains(shapely_point):
                # if shapely_point.within(polygon["shapely_polygon_2d"]):
                matching_polygons.append(polygon)
            # End slow code

        # print(f"{len(matching_polygons)} matching polygons for point {shapely_point}: {matching_polygons}")
        return matching_polygons

    def _generate_point(self, x, y, x_tick, y_tick, depth_img, rgb_img, image_idx, random_level):
        # need 90 to -90 deg
        theta = x * x_tick  # Definition as in the thesis
        alpha = 90 - theta  # Definition as in RoomFormer
        phi = y * y_tick  # Definition as in the thesis
        beta = phi - 180  # Definition as in RoomFormer

        depth = depth_img[x, y] + np.random.random() * random_level

        # Don't output points too close - why?
        if depth > 500.0:
            z_offset = depth * np.sin(np.deg2rad(alpha))  # Thesis uses cos(theta), it's equivalent
            xy_offset = depth * np.cos(np.deg2rad(alpha))  # Thesis uses sin, its equivalent
            x_offset = xy_offset * np.sin(np.deg2rad(beta))  # ...
            y_offset = xy_offset * np.cos(np.deg2rad(beta))
            point_wrt_camera = np.asarray([x_offset, y_offset, z_offset])
            point_wrt_global = point_wrt_camera + self.camera_centers[image_idx]

            # Determine type of hit object and store it in the `types` var
            # Idea: Take this xyz point and project it down to the xy layer (just don't consider z). Match it with self.planes, checking on which plane the point is.

            matching_polygons = self._get_matching_polygons(Point(point_wrt_global[0], point_wrt_global[1]))

            if len(matching_polygons) == 0:
                return {
                    "coords": point_wrt_global,
                    "color": rgb_img[x, y],
                    "type": self.semantic_type_int_map["undefined"],
                    "room_id": ROOM_ID_INVALID,
                }
            elif len(matching_polygons) == 1:
                return {
                    "coords": point_wrt_global,
                    "color": rgb_img[x, y],
                    "type": self.semantic_type_int_map[matching_polygons[0]["semantic_type"]],
                    "room_id": matching_polygons[0]["room_id"],
                }
            else:
                # Filter out undefined/invalid type polygons and retry to do an unambiguous assignment
                non_undefined_invalid_type_polygons = []
                for p in matching_polygons:
                    if p["semantic_type"] != "undefined" and p["semantic_type"] != "invalid":
                        non_undefined_invalid_type_polygons.append(p)

                if len(non_undefined_invalid_type_polygons) == 0:
                    return {
                        "coords": point_wrt_global,
                        "color": rgb_img[x, y],
                        "type": self.semantic_type_int_map["undefined"],
                        "room_id": ROOM_ID_INVALID,
                    }
                elif len(non_undefined_invalid_type_polygons) == 1:
                    return {
                        "coords": point_wrt_global,
                        "color": rgb_img[x, y],
                        "type": self.semantic_type_int_map[non_undefined_invalid_type_polygons[0]["semantic_type"]],
                        "room_id": non_undefined_invalid_type_polygons[0]["room_id"],
                    }
                else:
                    for p1, p2 in pairwise(non_undefined_invalid_type_polygons):
                        lower_priority_types = ["door", "window"]  # TODO add outwall?
                        if p1["semantic_type"] != p2["semantic_type"]:
                            if p1["semantic_type"] not in lower_priority_types and p2["semantic_type"] not in lower_priority_types:
                                raise Exception(
                                    f"Polygon p1 and p2 have different semantic types - don't know which one to choose. No door polygon available that can be skipped. p1 = {p1}, p2 = {p2}"
                                )

                            # At least one of the elements of non_undefined_type_polygons is a door
                            # Filter out door type polygons and retry to do an unambiguous assignment
                            high_priority_type_polygons = []
                            for p in non_undefined_invalid_type_polygons:
                                if p["semantic_type"] not in lower_priority_types:
                                    high_priority_type_polygons.append(p)

                            if len(high_priority_type_polygons) == 0:
                                assert p1["semantic_type"] in lower_priority_types and p2["semantic_type"] in lower_priority_types
                                return {
                                    "coords": point_wrt_global,
                                    "color": rgb_img[x, y],
                                    "type": self.semantic_type_int_map[p1["semantic_type"]],
                                    "room_id": p1["room_id"],
                                }

                            elif len(high_priority_type_polygons) == 1:
                                return {
                                    "coords": point_wrt_global,
                                    "color": rgb_img[x, y],
                                    "type": self.semantic_type_int_map[high_priority_type_polygons[0]["semantic_type"]],
                                    "room_id": high_priority_type_polygons[0]["room_id"],
                                }

                            else:
                                # There are at least 2 polygons inside non_door_type_polygons, and none of them is a door or other lower priority class.
                                # We can only unambiguously identify the point type of all polygons have the same type.
                                for p1, p2 in pairwise(high_priority_type_polygons):
                                    if p1["semantic_type"] != p2["semantic_type"]:
                                        raise Exception(
                                            f"Polygon p1 and p2 have different semantic types - don't know which one to choose. Door polygons have already been skipped. p1 = {p1}, p2 = {p2}"
                                        )

                                    # All polygons have the same type
                                    return {
                                        "coords": point_wrt_global,
                                        "color": rgb_img[x, y],
                                        "type": self.semantic_type_int_map[high_priority_type_polygons[0]["semantic_type"]],
                                        "room_id": high_priority_type_polygons[0]["room_id"],
                                    }

                        # Else:
                        # There is no mismatch between the types of the polygons inside non_undefined_type_polygons.

                    # The non-existing else branch one line above was called every singe time.
                    return {
                        "coords": point_wrt_global,
                        "color": rgb_img[x, y],
                        "type": self.semantic_type_int_map[non_undefined_invalid_type_polygons[0]["semantic_type"]],
                        "room_id": non_undefined_invalid_type_polygons[0]["room_id"],
                    }
        return None

    def _generate_depth_img_point_cloud(self, task):
        try:
            depth_img = cv2.imread(self.depth_paths[task["image_idx"]], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            x_tick = 180.0 / depth_img.shape[0]  # index 0 is the height
            y_tick = 360.0 / depth_img.shape[1]  # index 1 is the width
            # The RoomFormer code somehow uses x and y the wrong way around...

            rgb_img = cv2.imread(self.rgb_paths[task["image_idx"]])
            rgb_img = cv2.cvtColor(rgb_img, code=cv2.COLOR_BGR2RGB)

            coords = []
            colors = []
            types = []
            room_ids = []

            with tqdm.tqdm(
                # position has +1, because position 0 is used by the module that calls this module
                total=depth_img.shape[0] * depth_img.shape[1],
                position=task["image_idx"] + 1,
                leave=False,
                disable=not self.debug_mode,
            ) as pbar_pixels:
                pbar_pixels.set_description(self.depth_paths[task["image_idx"]])

                for x in range(0, depth_img.shape[0]):
                    for y in range(0, depth_img.shape[1]):
                        point = self._generate_point(
                            x,
                            y,
                            x_tick,
                            y_tick,
                            depth_img,
                            rgb_img,
                            task["image_idx"],
                            task["random_level"],
                        )
                        if point:
                            coords.append(point["coords"])
                            colors.append(point["color"])
                            types.append(point["type"])
                            room_ids.append(point["room_id"])

                        pbar_pixels.update()

            return {
                "success": True,
                "coords": coords,
                "colors": colors,
                "types": types,
                "room_ids": room_ids,
            }

        except Exception:
            return {"success": False, "exception": traceback.format_exc()}

    def _generate_point_cloud(self, random_level=0, color=False, normal=False, debug_mode=False, num_workers=1):
        # Getting single points
        tasks = []
        for image_idx in range(len(self.depth_paths)):
            tasks.append(
                {
                    "image_idx": image_idx,
                    "random_level": random_level,
                }
            )

        # Pool creates processes instead of threads (ThreadPool class), so the computation is a lot faster.
        worker_results: list[Any] = Parallel(n_jobs=num_workers)(delayed(self._generate_depth_img_point_cloud)(task) for task in tasks)  # type: ignore

        # Combine worker results
        coords = []
        colors = []
        types = []
        room_ids = []
        for worker_result in worker_results:
            if worker_result["success"]:
                coords.extend(worker_result["coords"])
                colors.extend(worker_result["colors"])
                types.extend(worker_result["types"])
                room_ids.extend(worker_result["room_ids"])
            else:
                raise Exception(f"Exception during inner multiprocessing: {worker_result['exception']}")

        coords = np.asarray(coords)
        colors = np.asarray(colors) / 255.0
        types = np.asarray(types)
        room_ids = np.asarray(room_ids)

        coords[:, :2] = np.round(coords[:, :2] / 10) * 10.0  # Round x and y
        coords[:, 2] = np.round(coords[:, 2] / 100) * 100.0  # Round z
        unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)

        coords = coords[unique_ind]
        colors = colors[unique_ind]
        types = types[unique_ind]
        room_ids = room_ids[unique_ind]

        points = {
            "coords": coords,
            "colors": colors,
            "types": types,
            "room_ids": room_ids,
        }

        # print("Pointcloud size:", points["coords"].shape[0])
        return points

    def export(self, destination_file_path_ply: str, destination_file_path_las: str, num_workers=1):
        """Exports the point cloud in the .ply file format.

        Args:
            destination_file_path (str): Path to the destination file path where the point cloud should be saved.
        """
        point_cloud = self._generate_point_cloud(
            self.random_level, color=self.generate_color, normal=self.generate_normal, debug_mode=self.debug_mode, num_workers=num_workers
        )

        self.export_ply(destination_file_path_ply, point_cloud)
        self.export_las(destination_file_path_las, point_cloud)

    def export_ply(self, destination_file_path: str, point_cloud):
        with open(destination_file_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % point_cloud["coords"].shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property int type\n")
            f.write("property int room_id\n")
            if self.generate_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            if self.generate_normal:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            f.write("end_header\n")

            for i in range(point_cloud["coords"].shape[0]):
                normal = []
                color = []
                coord = point_cloud["coords"][i].tolist()

                if self.generate_color:
                    color = list(map(int, (point_cloud["colors"][i] * 255).tolist()))

                if self.generate_normal:
                    normal = point_cloud["normals"][i].tolist()

                type_value = [point_cloud["types"][i]]  # Turn to list s.t. concatenation works
                room_id_value = [point_cloud["room_ids"][i]]  # Turn to list s.t. concatenation works
                data = coord + type_value + room_id_value + color + normal
                f.write(" ".join(list(map(str, data))) + "\n")

    def export_las(self, destination_file_path: str, point_cloud):
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.offsets = np.min(point_cloud["coords"], axis=0)
        header.scales = np.array([1, 1, 1])
        header.add_extra_dim(laspy.ExtraBytesParams(name="type", type=np.int32))
        header.add_extra_dim(laspy.ExtraBytesParams(name="room_id", type=np.int32))
        las = laspy.LasData(header)

        las.x = point_cloud["coords"][:, 0]
        las.y = point_cloud["coords"][:, 1]
        las.z = point_cloud["coords"][:, 2]

        las["type"] = point_cloud["types"]
        las["room_id"] = point_cloud["room_ids"]

        las.write(destination_file_path)
