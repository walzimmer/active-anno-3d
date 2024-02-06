# Installation Guide

An outline to set up the environment and dependencies for this project using Docker.

In the project directory, run the following:

```bash
docker build . -t <image-name>
```
```
docker run -it --gpus all -v $(pwd):/<container_directory> --name <container_name> <image_name>
```
This will open a shell terminal in the docker container, run the following commands:
```
source ~/.bashrc
```

```
xhost+
```

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit==11.3.1 -c pytorch -c conda-forge
```
```
pip install spconv-cu113==2.1.21
```

```
pip uninstall cumm-cu113
```
```
pip install cumm-cu113==0.2.8
```
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
```
conda install jupyter
```
```
pip install scikit-image matplotlib imageio plotly opencv-python
```
```
conda install pytorch3d -c pytorch3d
```
```
pip install open3d==0.16.0
```
```
conda install -c conda-forge plyfile
```
```
pip install addict mayavi
```
```
python setup.py develop
```


```
cd /home
git clone https://github.com/DanielPollithy/pypcd
```
```
cd pypcd && pip install .
```
```
pip install ruamel.yaml
```

```
git config --global --add safe.directory /root
```
Wandb
```
pip install wandb
pip install chardet
```
```
wandb login
```
Copy the API key in the terminal and hit enter
