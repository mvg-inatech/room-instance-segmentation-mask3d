How I did the setup in devcontainer

Use nvidia docker image, comes with python 3.10 and no conda

* with conda, I didn't manage to get it to work
* without virtualenv, the make.sh will give an error because it tries to install to site. permissions denied. But with sudo, it doesn't find the installed torch and fails as well
* Solution:
    * open the devcontainer
    * pip install virtualenv
    * Add to .bashrc: PATH=$PATH:~/.local/bin
    * Restart terminal
    * virtualenv .venv
    * source .venv/bin/activate
    * pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    * pip install -r requirements.txt
    * continue with models/ops etc. as written in the README
        * cd models/ops
        * sh make.sh
        * cd ../../diff_ras
        * python setup.py build develop
        * cd ../models/ops
        * python test.py
            * -> README says it needs to print true
            * -> CUDA OOM even on the DL1 A100 80GB GPU, but the first tests returned true
                * Apparently, this is not a problem. Just continue.
    * continue with downloading the data, see README