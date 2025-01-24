# RoomFormer

## Installation

* With conda, as described in the [original RoomFormer repo](https://github.com/ywyue/RoomFormer), I didn't manage to get it to work
* Without virtualenv, the make.sh will give an error because it tries to install to site. Permissions are denied. But with sudo, it doesn't find the installed torch and fails as well.
* Solution (how RoomFormer was installed for this work):
    * open the devcontainer (use Visual Studio Code and use the dev containers extension)
    * pip install -U virtualenv
    * Add to .bashrc: PATH=$PATH:~/.local/bin
    * Restart terminal
    * virtualenv .venv
    * source .venv/bin/activate
    * pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    * pip install -r requirements.txt
    * continue with models/ops etc. as written in the original RoomFormer README:
        * cd models/ops
        * sh make.sh
        * cd ../../diff_ras
        * python setup.py build develop
        * cd ../models/ops
        * python test.py
            * -> README says it needs to print true
            * -> I'm getting CUDA OOM even on an NVIDIA A100 80GB GPU, but the first tests returned true
                * Apparently, this is not a problem. Just continue.
    * Continue with downloading the data, see the original RoomFormer README
