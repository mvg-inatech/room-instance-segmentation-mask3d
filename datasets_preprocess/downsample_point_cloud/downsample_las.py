import os
import traceback
import laspy
import numpy as np

from downsample_ply import config, downsample_point_cloud, main, save_rastered_cuboid_npy, save_sparse_point_cloud_las, save_sparse_point_cloud_ply


def read_las(ply_in_path: str):
    las_file = laspy.read(ply_in_path)

    # Load the 'x', 'y', 'z', 'type', 'room_id' properties of the point cloud
    # Ignore the 'r', 'g', 'b' properties as we don't need them
    coordinates = np.vstack((las_file.x, las_file.y, las_file.z)).T  # Shape: (num_points, 3)
    types = np.array(las_file["type"])  # Shape: (num_points)
    room_ids = np.array(las_file["room_id"])  # Shape: (num_points)
    return coordinates, types, room_ids


def downsample_scene(task):
    os.nice(5)  # Send worker process slightly to background

    try:
        las_in_path = os.path.join(task["data_root"], task["scene"], "point_cloud.las")
        npy_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.npy")
        ply_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.ply")
        las_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.las")

        rasterized_cuboid, sparse_points = downsample_point_cloud(las_in_path, task["voxel_size"], read_las)
        save_rastered_cuboid_npy(rasterized_cuboid, npy_out_path)
        save_sparse_point_cloud_ply(sparse_points, ply_out_path)
        save_sparse_point_cloud_las(sparse_points, las_out_path)
    except Exception:
        return {"scene": task["scene"], "success": False, "exception": traceback.format_exc()}

    return {"scene": task["scene"], "success": True}  # indicates success


if __name__ == "__main__":
    main(config(), downsample_scene)
