"""
Parts of this code were taken from the RoomFormer paper:
Y. Yue, T. Kontogianni, K. Schindler, and F. Engelmann, “Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries.” arXiv, Mar. 2023. Accessed: Mar. 19, 2024. [Online]. Available: http://arxiv.org/abs/2211.15658
See https://github.com/ywyue/RoomFormer/blob/main/data_preprocess/stru3d/generate_point_cloud_stru3d.py

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

import time
import traceback
import argparse
import os
import tqdm
from point_cloud_reader_stru3d import PointCloudReaderPanorama
from multiprocessing import Pool


def convert_scene(task):
    os.nice(5)  # Send worker process slightly to background

    if task["debug_mode"]:
        print(f"Starting processing of {task['scene']}")

    try:
        scene_path = os.path.join(task["data_root"], task["scene"])
        reader = PointCloudReaderPanorama(scene_path, random_level=0, generate_color=True, generate_normal=False, debug_mode=task["debug_mode"])
        save_path_ply = os.path.join(task["data_root"], task["scene"], "point_cloud.ply")
        save_path_las = os.path.join(task["data_root"], task["scene"], "point_cloud.las")
        reader.export(save_path_ply, save_path_las)
    except Exception:
        return {"scene": task["scene"], "success": False, "exception": traceback.format_exc()}

    return {"scene": task["scene"], "success": True}


def config():
    a = argparse.ArgumentParser(description="Generate point cloud for Structured3D")
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
        help="Number of parallel workers to use for the conversion. The program will start the conversion of --num_workers many scenes in parallel.",
    )
    a.add_argument(
        "--scene",
        default=None,
        type=int,
        help="Instead of all scenes, only convert a single scene with the given id",
    )
    args = a.parse_args()
    return args


def main(args):
    data_root = args.data_root

    if args.scene is not None:
        scenes = [f"scene_{args.scene:05}"]
    else:
        scenes = sorted(os.listdir(data_root))

    start_timestamp = int(time.time())
    with open(f"run_{start_timestamp}_log.txt", "a") as log_file:
        log(f"Using up to {args.num_workers} workers", log_file)
        log("Creating point clouds from perspective views...", log_file)

        assert args.num_workers >= 1

        debug_mode = len(scenes) == 1 or args.num_workers == 1

        tasks = []
        for idx, scene in enumerate(scenes):
            tasks.append(
                {
                    "scene": scene,
                    "idx": idx,
                    "data_root": data_root,
                    "debug_mode": debug_mode,
                }
            )

        failed_items = []

        with open(f"run_{start_timestamp}_valid_scenes.txt", "a") as valid_scenes_file:
            # One process per scene
            with Pool(processes=args.num_workers) as pool:
                # The convert_scene function will start args.num_workers many workers as well
                # Therefore, the arg is named like this
                worker_results = list(tqdm.tqdm(pool.imap(convert_scene, tasks), total=len(scenes), position=0))
                for worker_result in worker_results:
                    if worker_result["success"]:
                        valid_scenes_file.write(worker_result["scene"] + "\n")
                        valid_scenes_file.flush()
                    else:
                        log("-------", log_file)
                        log(f"Failed processing of {worker_result['scene']}", log_file)
                        log(worker_result["exception"], log_file)
                        failed_items.append(worker_result)

        log("", log_file)
        log("Done", log_file)

        if len(failed_items) > 0:
            print("")
            log(f"Conversion failed for {len(failed_items)} items.", log_file)
            print("Summary:")
            for failed_item in failed_items:
                print("-------")
                print(f"Failed processing of {failed_item['scene']}")
                print(failed_item["exception"])


def log(text, file):
    print(text)
    file.write(text + "\n")
    file.flush()


if __name__ == "__main__":
    main(config())
