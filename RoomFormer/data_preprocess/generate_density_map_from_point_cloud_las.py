import argparse
import cv2
import numpy as np
from common_utils import read_scene_pc
from stru3d.stru3d_utils import generate_density
import laspy


def config():
    a = argparse.ArgumentParser(description="Generate density map from 3D point cloud")
    a.add_argument("--input", default="point_cloud.las", type=str, help="Path to input 3D point cloud .las file")
    a.add_argument("--output", default="density_map.png", type=str, help="Path to output density map")

    args = a.parse_args()
    return args


def export_density(density_map, out_path):
    density_uint8 = (density_map * 255).astype(np.uint8)
    cv2.imwrite(out_path, density_uint8)


def read_las(ply_in_path: str):
    las_file = laspy.read(ply_in_path)

    # Load the 'x', 'y', 'z', 'type', 'room_id' properties of the point cloud
    # Ignore the 'r', 'g', 'b' properties as we don't need them
    coordinates = np.vstack((las_file.x, las_file.y, las_file.z)).T  # Shape: (num_points, 3)
    types = np.array(las_file["type"])  # Shape: (num_points)
    room_ids = np.array(las_file["room_id"])  # Shape: (num_points)
    return coordinates, types, room_ids


def main(args):
    ### begin processing
    points, _, _ = read_las(args.input)
    xyz = points[:, :3]

    ### project point cloud to density map
    density, normalization_dict = generate_density(xyz, width=256, height=256)

    export_density(density, args.output)


if __name__ == "__main__":
    main(config())
