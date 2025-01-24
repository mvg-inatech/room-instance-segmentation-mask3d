import shutil
import laspy
import numpy as np
from plyfile import PlyData
import argparse
import os
from zipfile import ZipFile
import glob
import pathlib
from tqdm import tqdm

# download_mp.py is the original Matterport3D download script. We cannot publish it for licensing reasons. You must obtain it yourself at https://niessner.github.io/Matterport/#download
from download_mp import (
    BASE_URL,
    RELEASE,
    download_scan,
    get_release_scans,
)

RELEVANT_FILE_TYPES = ["region_segmentations"]


def preprocess_scene(
    scene_id: str, scene_out_dir: str, preprocessing_las_out_path: str
):
    # print(f"Preprocessing scene {scene_id}...")

    # print("Unzipping...")
    zip_path = os.path.join(scene_out_dir, "region_segmentations.zip")
    unzip_path = os.path.join(scene_out_dir, "region_segmentations")
    with ZipFile(zip_path, "r") as zip_object:
        zip_object.extractall(path=unzip_path)
    os.unlink(zip_path)

    # print("Loading and merging .ply files...")
    temp_scene_path = os.path.join(unzip_path, scene_id)
    ply_file_paths = glob.glob(
        os.path.join(temp_scene_path, "region_segmentations", "*.ply")
    )
    assert len(ply_file_paths) > 0

    merged_points = []
    for ply_file_idx, ply_file_path in enumerate(
        sorted(ply_file_paths)
    ):  # Sort to ensure consistent point order in the output point cloud. Not sure whether we really need this (maybe as a workaround for reproducibility issues with the Minkowski Engine package).
        # print(f"  Loading {ply_file_path}")
        plydata = PlyData.read(ply_file_path)
        num_points = len(plydata["vertex"]["x"])
        points = np.stack(
            [
                plydata["vertex"]["x"],
                plydata["vertex"]["y"],
                plydata["vertex"]["z"],
                plydata["vertex"]["red"],
                plydata["vertex"]["green"],
                plydata["vertex"]["blue"],
                np.ones(num_points),  # room type
                np.ones(num_points) * (ply_file_idx + 1),  # room instance id
            ],
            axis=-1,
        )
        merged_points.append(points)

    merged_points_np = np.concatenate(merged_points, axis=0)

    header = laspy.LasHeader(point_format=3, version="1.4")
    # Do not set header.offsets to keep the original dataset coordinates
    header.scales = np.array([1, 1, 1])  # Millimeter resolution (see *1000 below)
    header.add_extra_dim(laspy.ExtraBytesParams(name="type", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="room_id", type=np.int32))

    las = laspy.LasData(header)

    las.x = merged_points_np[:, 0] * 1000  # Convert meters to millimeters
    las.y = merged_points_np[:, 1] * 1000  # Convert meters to millimeters
    las.z = merged_points_np[:, 2] * 1000  # Convert meters to millimeters

    las.red = merged_points_np[:, 3]
    las.green = merged_points_np[:, 4]
    las.blue = merged_points_np[:, 5]

    las["type"] = merged_points_np[:, 6]
    las["room_id"] = merged_points_np[:, 7]

    pathlib.Path(preprocessing_las_out_path).mkdir(parents=True, exist_ok=True)
    las.write(os.path.join(preprocessing_las_out_path, "point_cloud.las"))

    # Delete temp files
    shutil.rmtree(scene_out_dir)


def process_scan(scan_id: str, out_dir: str):
    preprocessing_input_dir = os.path.join(out_dir, "temp", RELEASE, scan_id)
    preprocessing_output_dir = os.path.join(out_dir, "preprocessed", RELEASE, scan_id)
    download_scan(scan_id, preprocessing_input_dir, RELEVANT_FILE_TYPES)
    preprocess_scene(scan_id, preprocessing_input_dir, preprocessing_output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Downloads MP public data release and preprocesses for room instance segmentation.
        """,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="directory in which to save the preprocessed files",
    )
    parser.add_argument(
        "--id",
        default="ALL",
        help="specific scan id to download or ALL to download entire dataset",
    )
    args = parser.parse_args()

    release_file = BASE_URL + RELEASE + ".txt"
    print("Getting release scans...")
    release_scans = get_release_scans(release_file)

    # download task data
    # task_data=["region_classification_data"]
    # out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
    # download_task_data(task_data, out_dir)

    if args.id and args.id != "ALL":  # download single scan
        scan_id = args.id
        if scan_id not in release_scans:
            print("ERROR: Invalid scan id: " + scan_id)
        else:
            process_scan(scan_id, args.out_dir)

    elif args.id == "ALL" or args.id == "all":  # download entire release
        print("WARNING: You are downloading all scenes.")
        print(
            "Existing scan directories will be skipped. Delete partially downloaded directories to re-download."
        )
        out_dir = os.path.join(args.out_dir, RELEASE)

        print("Downloading entire dataset to " + out_dir + "...")
        for scan_id in tqdm(release_scans, desc="Downloading scans", unit="scan"):
            process_scan(scan_id, args.out_dir)


if __name__ == "__main__":
    main()
