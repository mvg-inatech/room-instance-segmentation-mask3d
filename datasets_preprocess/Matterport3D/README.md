# Matterport3D dataset preprocessing

This script downloads the [Matterport3D dataset](https://niessner.github.io/Matterport/) and preprocesses it for 3D room instance segmentation.
Output files are saved in .las format.

## Instructions

### Installation

This script was run with Python 3.12.3.

1. `pip install -U virtualenv`
2. `virtualenv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Obtain the download_mp.py file. It is the original Matterport3D download script. We cannot publish it for licensing reasons. You must obtain it yourself at https://niessner.github.io/Matterport/#download. Save it to the same directory as this `README.md` file.

### Running

1. `source .venv/bin/activate`
2. `python download_and_preprocess.py -o base_dir --id 17DRP5sb8fy` for a single scene, or `python download_and_preprocess.py -o base_dir` for all scenes

Only run one instance of the script at the same time with the same output directory (`-o` parameter).
