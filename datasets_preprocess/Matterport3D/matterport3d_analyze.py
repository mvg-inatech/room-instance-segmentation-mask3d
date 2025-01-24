import os
import laspy
import numpy as np


def main():
    dataset_root = "/data/Matterport3D/preprocessed/v1/scans/"

    files_to_analyze = []
    highest_room_id_scene_name = None
    highest_room_id = -1

    for scene_name in os.listdir(dataset_root):
        files_to_analyze.append({"scene_name": scene_name, "las_path": os.path.join(dataset_root, scene_name, "point_cloud.las")})

    # Load each of the las files
    for scene in files_to_analyze:
        las_file = laspy.read(scene["las_path"])

        # Load the 'x', 'y', 'z', 'type', 'room_id' properties of the point cloud
        room_ids = np.array(las_file["room_id"])  # Shape: (num_points)
        if max(room_ids) > highest_room_id:
            highest_room_id = max(room_ids)
            highest_room_id_scene_name = scene["scene_name"]

    print(f"Scene with highest room id: {highest_room_id_scene_name}, with max room id: {highest_room_id}")


if __name__=="__main__":
    main()
