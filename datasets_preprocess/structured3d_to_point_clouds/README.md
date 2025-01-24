# Generate point clouds from the Structured3D dataset

For other parts of this work, point clouds of building indoor areas are required.

This script loads the [Structured3D](https://structured3d-dataset.org/) dataset.

Parts of the code were taken from the RoomFormer paper, especially from [this file](https://github.com/ywyue/RoomFormer/blob/main/data_preprocess/stru3d/PointCloudReaderPanorama.py).

The implementation of the RoomFormer team was extended by the following features:

* Add floor plan annotations to the output point cloud
* Resolve ambiguities
* See our paper for more details

Annotated point clouds are exported in the .ply and .las file formats.

## Running the conversion

1. Open this repository in Visual Studio Code.
2. Use the Dev Containers extension to create a Docker container as specified in the `.devcontainer/` directory.
3. Download the Structured3D dataset. You need to download and unzip the following parts:
    * Panorama images (all parts that you wish to convert)
    * Structure annotations
    * 3D bounding box and instance annotations (only required for visualizing the floor plan of a scene without the `--floor_only` parameter)

4. Move the extracted dataset to `/home/your-username/data/Structured3D/`. Cross-check the file `.devcontainer/devcontainer.json` to ensure that this path is mounted inside the dev container.
5. Start the created Docker container that comes with all dependencies installed. Start a shell inside. You will probably do this using the Dev Containers extension for Visual Studio Code.
6. Run the script (adapt the parameters accordingly): `$ python generate_point_cloud_stru3d.py --data_root /data/Structured3D --num_workers 8`

## Visualizing the floor plan of a scene

For debugging purposes, you can plot the floor plan annotations from the dataset. You can compare the plot with the generated point clouds (use CloudCompare to visualize them).

1. Follow the steps above and shart a shell inside the dev container.
2. Run the script (adapt the parameters accordingly): `$ python visualize_floorplan.py --data_root /data/Structured3D --scene 19`. Optionally, you can add the `--floor_only` parameter to only visualize floor polygons (no furniture or other objects).
