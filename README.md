# Three-Dimensional Instance Segmentation of Rooms in Indoor Building Point Clouds Using Mask3D

<div align="center">
Michael Brunklaus<sup>1,2</sup>,
Maximilian Kellner<sup>1,2</sup>,
Alexander Reiterer<sup>1,2</sup>


<sup>1</sup>University of Freiburg, Germany
<sup>2</sup>Fraunhofer Institute for Physical Measurement Techniques IPM, Germany

We segmented room instances in 3D indoor building point clouds without the need for orthographic projections into 2D density maps.


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


</div>

This repository belongs to the following paper: Remote Sens. 2025, 17(7), 1124; DOI: [10.3390/rs17071124](https://doi.org/10.3390/rs17071124)


## Abstract

While most recent work in room instance segmentation relies on orthographic top-down projections of 3D point clouds to 2D density maps, leading to information loss of one dimension, 3D instance segmentation methods based on deep learning were rarely considered. We explore the potential of the general 3D instance segmentation deep learning model Mask3D for room instance segmentation in indoor building point clouds. We show that Mask3D generates meaningful predictions for multi-floor scenes. After hyperparameter optimization, Mask3D outperforms the current state-of-the-art method RoomFormer evaluated in 3D on the synthetic Structured3D dataset. We provide generalization results of Mask3D trained on Structured3D to the real-world S3DIS and Matterport3D datasets, showing a domain gap. Fine-tuning improves the results. In contrast to related work in room instance segmentation, we employ the more expressive mean average precision (mAP) metric, and we propose the more intuitive successfully detected rooms (SDR) metric, which is an absolute recall measure. Our results indicate potential for the digitization of the construction industry.


## Code structure
We adapt the codebases of [Mask3D](https://github.com/JonasSchult/Mask3D), a 3D instance segmentation model for point clouds, and of [RoomFormer](https://github.com/ywyue/RoomFormer), the current state-of-the-art in floor plan reconstruction.

```
├── datasets_preprocess               <- dataset preprocessing code
│   ├── downsample_point_cloud        <- downsampling point clouds
│   ├── Matterport3D                  <- Matterport3D dataset related
│   ├── structured3d_analyze		  <- Analyzes properties of the Structured3D dataset
│   ├── structured3d_to_point_clouds  <- Generates point clouds from the Structured3D dataset
├── mask3d                            <- Mask3D model code, adapted for our room instance segmentation task
├── RoomFormer                        <- RoomFormer model code, adapted to the metrics of this work
```

Please find the valid_scenes_file used for Structured3D [here](structured3d_valid_scenes_class21.txt). It contains a list of all scenes that were used for training because their preprocessing was successful.


## Setting up the development environment

We use Visual Studio Code and run our code within devcontainers (provided by the [remote development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)) to avoid interference with operating system packages.

The following devcontainer configurations are used:
* `dataset_preprocess/.devcontainer`
	* Used for all dataset preprocessing
	* For the `Matterport3D` subdirectory, you need to create a virtual Python environment within the devcontainer. See the `Matterport3D/README.md` for details.
* `mask3d/.devcontainer`
	* Used for the Mask3D model
* `RoomFormer/.devcontainer`
	* Used for the RoomFormer model


Within the Dockerfile of each devcontainer configuration, you have to change the following parameters to match the user logged in to your local dev machine, in order to avoid permission problems:
```Docker
ARG USER_UID=610212133
ARG USER_GID=610200513
```

Within the devcontainer.json file of each devcontainer configuration, you need to mount your dataset folder into the Docker container so that it is available at `/data` from inside the container:
```JSON
"mounts": [
	"source=/your/data_folder/,target=/data,type=bind"
]
```

<!--
## BibTeX
```
@article{
  TODO
}
```
-->
