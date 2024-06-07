# Checking Cuda version along with openCV from Jetson Stat
sudo apt-get update
sudo apt-get upgrade

sudo pip3 install -U jetson-stats
jtop
------------------------------------------

# Activate Cuda on Jetson
## Install OpenCV with CUDA Enable
To make CUDA enable in OpenCV, we need to build the OpenCV with CUDA from Source. Follow this repo to do that : https://github.com/mdegans/nano_build_opencv
or you can follow this step:
```
sudo git clone https://github.com/mdegans/nano_build_opencv.git
gedit build_opencv.sh
```
* follow from this documentation `https://docs/opencv.org` to make sure you're building you cuda on the right opencv version (the default verison is 4.4.0)
```
./build_opencv.sh (your opencv version)
```

------------------------------------------
# check if Cudnn have ininstalled
dpkg -l | grep libcudnn

# To instal Cudnn follow this command
cd ~/downloads
https://developer.nvidia.com/downloads/c102-cudnn-local-repo-ubuntu1804-8708410-1amd64deb
sudo dpkg -i cudnn-local-repo-ubuntu1804-8.7.0.84_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev

------------------------------------------
------------------------------------------
# Install 

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Download for python 3.8 
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics
cd ultralytics

# make a virtual env for torch
python3.8 -m venv venv
source venv/bin/activate #use this command again to activate venv
pip list #to see what packages have installed

pip install -U pip wheel gdown

------------------------------------------

# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
python3.8 -m pip install torch-*.whl torchvision-*.whl

pip install .
yolo task=detetct mode predict=yolov8n.pt sourch=0 show=true
yolo task=segment mode predict=yolov8n-seg.pt sourch=0 show=true

# python3 -m pip install -r requirements.txt

>> nano ~/VENV/python/0/bin/activate
#add these paths to venv to access cuda and cudnn
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda/bin:$PATH

------------------------------------------
------------------------------------------

# RUN 
# run with images
python3 run-yolo.py --weights yolov5n.pt --source data/images --device 0

#run with webcam
python3 run-yolo.py --weights yolov5n.pt --source 0 --device 0 --conf-thres 0.25 --imgsz 640
