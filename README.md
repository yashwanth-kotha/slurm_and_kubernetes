# slurm_and_kubernetes
This project is about  installing SLURM (High performance computing) on bare metal and Kubernetes to deploy Machine learning model

I completed installing SLURM on Bare metal

Ubuntu == 20.4
CUDA == 12.1
OpenMPI == 4.17
NCCL == 2.18 latest version
Python == 3.9
PIP == Latest
Pytorch == 2.4.0
NumPy
colossal.ai == Download using git

I am able to install all the libraries but there are issues with versions and bug with the colossal.ai

Colossal.ai installation on local Ubuntu machine
What is colossal.ai?
Colossal-AI is an open-source system designed to make it easier to train and deploy large-scale AI models efficiently. It provides a user-friendly interface to scale deep learning models across distributed systems, supporting multiple parallel training methods like data, pipeline, and tensor parallelism. Colossal-AI also optimizes resource use through advanced techniques, making large model training more accessible and efficient for AI developers.

Here are the versions compatibility for installing colossal.ai
Ubuntu == 20.4 
CUDA == 12.1 
OpenMPI == 4.1.7 
NCCL == 2.23
Python == 3.9 
PIP == Latest 
Pytorch == 2.4.0 
colossal.ai 

Installation scripts

Ubuntu == 20.4 
#Currently installing Ubuntu and related libraries through MAAS

 
CUDA == 12.1 

https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


After you run those scripts, you will encounter following error
cuda : Depends: cuda-12-1 (>= 12.1.0) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.
root@a02411:/# nvcc --version

Command 'nvcc' not found, but can be installed with:

apt install nvidia-cuda-toolkit


Then use this script to download the CUDA
apt clean 
apt update 
sudo apt purge cuda 
sudo apt purge nvidia-* 
sudo apt autoremove 
sudo apt install cuda


Try the nvidia-smi command and check the CUDA version on the top right. It will change to 12.1 from 12.2

OpenMPI == 4.1.7 

wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz
tar -xzf openmpi-4.1.7.tar.gz
cd openmpi-4.1.7


./configure --with-cuda


make -j12  # Adjust the number based on your CPU cores nproc - to get the number

sudo make install



NCCL 

Go to /usr/local

git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
sudo apt install build-essential devscripts debhelper fakeroot
sudo make pkg.debian.build
ls build/pkg/deb/
sudo dpkg -i build/pkg/deb/libnccl2_*.deb
sudo dpkg -i build/pkg/deb/libnccl-dev_*.deb
cd ..




Python == 3.9 
PIP == Latest 

The default Python 3.8 is installed while installing Ubuntu through MAAS
We need to install 3.9 and create vevn using python 3.9 version
Have pip to use python 3.9 version instead of the default one

Pytorch == 2.4.0

Install Pytorch in the venv so that it won’t disturb the other libraries

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121



colossal.ai
 
git clone https://github.com/hpcaitech/ColossalAI.git 
cd ColossalAI 
python3.9 -m pip install .



Sample colossal.ai script

Create/activate a new venv if running locally

git clone https://github.com/hpcaitech/ColossalAI.git 
cd ColossalAI/examples/images/resnet


pip install -r requirements.txt


training
# train with torch DDP with fp32
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp16 -p torch_ddp_fp16

# train with low level zero
colossalai run --nproc_per_node 2 train.py -c ./ckpt-low_level_zero -p low_level_zero


Evaluating
# evaluate fp32 training
python eval.py -c ./ckpt-fp32 -e 80

# evaluate fp16 mixed precision training
python eval.py -c ./ckpt-fp16 -e 80

# evaluate low level zero training
python eval.py -c ./ckpt-low_level_zero -e 80



Testing

Nvidia-smi

Nvcc -version

colossalai check -i





