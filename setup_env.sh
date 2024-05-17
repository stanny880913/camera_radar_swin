# install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

#install mmdet3d
pip install mmcv-full==1.6.0
pip install mmsegmentation==0.30.0

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.1  
pip install -r requirements/build.txt
pip install -v -e .

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
pip install -v -e .  # or "python setup.py develop"

pip install  numba==0.53.0
pip install numpy==1.23.5

#install mamba
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.1 # current latest version tag
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..


git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.1 # current latest version tag
pip install .

# Uninstall and reinstall yapf
pip uninstall yapf -y
pip install yapf==0.40.1

#install transformer 
pip install transformer==4.30.1

# 參考 https://zhuanlan.zhihu.com/p/687359086可以成功