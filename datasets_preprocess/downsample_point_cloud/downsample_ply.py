import argparse
from multiprocessing import Pool
import os
import traceback
import numpy as np
import tqdm
import laspy
from plyfile import PlyData, PlyElement


def save_rastered_cuboid_npy(rasterized_cuboid, npy_out_path):
    np.save(npy_out_path, rasterized_cuboid)


def save_sparse_point_cloud_ply(points, ply_out_path):
    ply_element = PlyElement.describe(points, "vertex")
    PlyData([ply_element], text=True).write(ply_out_path)


def save_sparse_point_cloud_las(points, las_out_path):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.min(np.array([points["x"], points["y"], points["z"]]), axis=1)
    header.scales = np.array([1, 1, 1])
    header.add_extra_dim(laspy.ExtraBytesParams(name="type", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="room_id", type=np.int32))

    las = laspy.LasData(header)

    las.x = points["x"]
    las.y = points["y"]
    las.z = points["z"]

    las["type"] = points["type"]
    las["room_id"] = points["room_id"]

    las.write(las_out_path)


def read_ply(ply_in_path: str):
    plydata = PlyData.read(ply_in_path)

    # Load the 'x', 'y', 'z', 'type', 'room_id' properties of the point cloud
    # Ignore the 'r', 'g', 'b' properties as we don't need them
    in_points = plydata["vertex"]
    coords = np.array([in_points[property] for property in ["x", "y", "z"]]).T  # Shape: (num_points, 3)
    types = in_points["type"]
    room_ids = in_points["room_id"]
    return coords, types, room_ids


def downsample_point_cloud(original_point_cloud_path, voxel_size, read_original_point_cloud_fn) -> tuple:
    """Loads a .ply point cloud and downsamples it. Returns the downsampled point cloud in the following different formats:
    1) Rasterized cuboid consisting of voxels, as .npy format. -> Good for deep learning
    2) Unordered point cloud, as .ply format. -> Good for visualization with CloudCompare

    Args:
        ply_in_path (str): Path to the .ply in file.
        voxel_size (int): Voxel size for downsampling.

    Returns:
        tuple: (rasterized cuboid, unordered points)
    """
    # Load the original point cloud
    coords, types, room_ids = read_original_point_cloud_fn(original_point_cloud_path)

    # Downsampling: We simply grid the space and keep one point per voxel grid as a simple downsampling method.
    # The 3D space that will be rasterized below is a cuboid.

    input_min_coord_3d = np.min(coords, axis=0)  # Has shape 3, stores the min value for x,y,z dimension
    input_max_coord_3d = np.max(coords, axis=0)

    # Sample points from the dense input point cloud.
    # Floor s.t. we can remove unwanted points in the next step using np.unique.
    downsampled_coords = np.floor((coords - input_min_coord_3d) / voxel_size).astype(int)
    _, downsampled_unique_point_indices = np.unique(downsampled_coords, axis=0, return_index=True)

    cuboid_size_3d = np.ceil((input_max_coord_3d + 1 - input_min_coord_3d) / voxel_size).astype(int)  # +1 to ensure coverage

    rasterized_cuboid = np.zeros(shape=(*cuboid_size_3d, 2))  # 2 for the 2 labels (room_type, room_instance_id)

    for idx in downsampled_unique_point_indices:
        downsampled_point_coords = downsampled_coords[idx]
        rasterized_cuboid[downsampled_point_coords[0], downsampled_point_coords[1], downsampled_point_coords[2], 0] = types[idx]  # Set type
        rasterized_cuboid[downsampled_point_coords[0], downsampled_point_coords[1], downsampled_point_coords[2], 1] = room_ids[idx]  # Set room instance id
        # The rasterized cloud is initialized with zeros.
        # The semantics of the type and room_id information are set in the same way: 0 means undefined/unknown.
        # So effectively, we're only setting non-zero values in the cuboid where there was data in the point cloud.

    # Extract downsampled points and their properties
    unique_downsampled_coords = downsampled_coords[downsampled_unique_point_indices]

    # Whether the unordered sparse output (for .ply file) should use the original coordinates from the input point cloud (True) or the downsampled coordinates as the rasterized output (False).
    # Setting this to True reduces the possible batch size that can be used with Mask3D from 16 to something between 4 and 8.
    unordered_output_use_original_coords = False

    # Shift coordinates back to the original non-zero aligned place
    if unordered_output_use_original_coords:
        downsampled_coords = unique_downsampled_coords * voxel_size + input_min_coord_3d
    else:
        downsampled_min_coord_3d = np.min(unique_downsampled_coords, axis=0)
        downsampled_coords = unique_downsampled_coords + downsampled_min_coord_3d

    downsampled_types = types[downsampled_unique_point_indices]
    downsampled_room_ids = room_ids[downsampled_unique_point_indices]

    # Write the downsampled point cloud as sparse .ply file
    dtype = [("x", "i4"), ("y", "i4"), ("z", "i4"), ("type", "i4"), ("room_id", "i4")]
    sparse_points = np.empty(len(downsampled_coords), dtype=dtype)
    sparse_points["x"], sparse_points["y"], sparse_points["z"] = downsampled_coords.T
    sparse_points["type"] = downsampled_types
    sparse_points["room_id"] = downsampled_room_ids

    return rasterized_cuboid, sparse_points


def downsample_scene(task):
    os.nice(5)  # Send worker process slightly to background

    try:
        ply_in_path = os.path.join(task["data_root"], task["scene"], "point_cloud.ply")
        npy_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.npy")
        ply_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.ply")
        las_out_path = os.path.join(task["data_root"], task["scene"], f"point_cloud_rasterized_{task['voxel_size']}.las")

        rasterized_cuboid, sparse_points = downsample_point_cloud(ply_in_path, task["voxel_size"], read_ply)
        save_rastered_cuboid_npy(rasterized_cuboid, npy_out_path)
        save_sparse_point_cloud_ply(sparse_points, ply_out_path)
        save_sparse_point_cloud_las(sparse_points, las_out_path)
    except Exception:
        return {"scene": task["scene"], "success": False, "exception": traceback.format_exc()}

    return {"scene": task["scene"], "success": True}  # indicates success


def config():
    a = argparse.ArgumentParser(
        description="Downsample a point cloud in .ply format s.t. there it at most 1 point per voxel remaining. The point position within the voxel is slightly moved to match the voxel coordinates."
    )
    a.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Path to Structured3D dataset directory",
    )
    a.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of parallel workers to use for the conversion.",
    )
    a.add_argument(
        "--scene",
        default=None,
        type=str,
        help="Instead of providing a valid scenes file, only downsample a single scene with the given name. Either this argument or --scenes_file needs to be specified.",
    )
    a.add_argument(
        "--scenes_file",
        default=None,
        type=str,
        help="Path to a file that contains one scene per line that should be downsampled. Either this argument or --scene needs to be specified.",
    )
    a.add_argument("--voxel_size", default=100, type=int, help="Voxel size to use")
    args = a.parse_args()
    return args


def main(args, downsample_function):
    assert args.scene is not None or args.scenes_file is not None
    assert not (args.scene is not None and args.scenes_file is not None)

    data_root = args.data_root

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = []
        with open(args.scenes_file) as file:
            for line in file.readlines():
                line_cleaned = line.strip().replace("\n", "")
                if len(line_cleaned) > 0:
                    scenes.append(line_cleaned)

    print(f"Using {args.num_workers} workers")
    print(f"Downsampling point clouds with voxel size {args.voxel_size}...")

    assert args.num_workers >= 1

    tasks = []
    for scene in scenes:
        tasks.append(
            {
                "scene": scene,
                "data_root": data_root,
                "voxel_size": args.voxel_size,
            }
        )

    failed_items = []

    # One process per scene
    # Pool creates processes instead of threads (ThreadPool class), so the computation is a lot faster.
    with Pool(processes=args.num_workers) as pool:
        worker_results = list(tqdm.tqdm(pool.imap(downsample_function, tasks), total=len(scenes), position=0))
        for worker_result in worker_results:
            if not worker_result["success"]:
                print("-------")
                print(f"Failed processing of {worker_result['scene']}")
                print(worker_result["exception"])
                failed_items.append(worker_result)

    print("")
    print("Done")

    if len(failed_items) > 0:
        print("")
        print(f"Downsampling failed for {len(failed_items)} scenes. Summary:")
        for failed_item in failed_items:
            print("-------")
            print(f"Failed processing of {failed_item['scene']}")
            print(failed_item["exception"])


if __name__ == "__main__":
    main(config(), downsample_scene)
