# Mask3D for Room Instance Segmentation

This code is based on the [original Mask3D repository](https://github.com/JonasSchult/Mask3D), commit id `11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5`.

It was adapted for room instance segmentation.

## How to run the experiments from our paper?
1. Open devcontainer in vscode
2. Run the bash scripts in `./experiments_launch_scripts`

## How to run own experiments?
1. Open devcontainer in vscode
2. Get familiar with hydra so you know how to configure the application
3. The main config is at the following path: `conf/config_base_instance_segmentation.yaml`. If the `train_mode` config param is set to `true`, it will train the model and evaluate both on the train and validation splits. If the `train_mode` config param is set to `false`, it will test a checkpoint on the test split.
4. Run by calling: `python main_instance_segmentation.py`

* To load a custom checkpoint, you can call the script as following: `python -m main_instance_segmentation 'general.checkpoint="saved/run9/epoch=219-mapval_mean_ap_50=0.430.ckpt"'`. Further read that explains the CLI parameters: [here](https://hydra.cc/docs/advanced/override_grammar/basic/#quoted-values).

* Be cautious when training and then testing. You should increase the version number between any two calls. Otherwise, the CSV logs (`metrics.csv`) will be overwritten. This behavior of the lightning CSVLogger cannot be changed. However, tensorboard logs do not have a static filename like `metrics.csv`, but they include a timestamp. Therefore, for the Tensorboard logs, there is no risk for overwriting.

## Good to know

* You can ignore `RuntimeWarning: Mean of empty slice` console output during execution.
