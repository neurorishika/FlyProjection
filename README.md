# FlyProjection

<!-- badges: start -->
<!-- badges: end -->

Author: [Rishika Mohanta](https://neurorishika.github.io/)

Latest Build Date: 2024-09-18 14:31:22

## About the Project

Project description is being updated. Please check back later.

## Instructions

This is a Poetry-enabled python project. Make sure you have poetry installed from https://python-poetry.org/.


First, you need to setup a git alias for tree generation by running the following command on the terminal:

```
git config --global alias.tree '! git ls-tree --full-name --name-only -t -r HEAD | sed -e "s/[^-][^\/]*\//   |/g" -e "s/|\([^ ]\)/|-- \1/"'
```

## Install Instructions (Assumes Ubuntu 22.04 LTS on Nvidia GPU)

### PART 1: Make sure Nvidia Software is Installed

Check the current version using: `cat /proc/driver/nvidia/version` If not discovered, follow instructions at: https://ubuntu.com/server/docs/nvidia-drivers-installation

Verify by calling `nvidia-smi`. Output should look something like.

```
Date-Time      
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
|  0%   41C    P8              26W / 450W |     11MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2300      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+

```

After this install the **Nvidia CUDA Toolkit**:

```
sudo apt update && sudo apt upgrade
sudo apt-get install nvidia-cuda-toolkit
sudo apt-get install nvidia-gds # this is optional but recommended (if it doesnt work, follow CUDA install instructions on Nvidia's Website)
sudo reboot 
```

Verify installation using: `nvcc --version`

```
sudo apt-get install zlib1g
sudo apt-get install nvidia-cudnn
```

### PART 2: Build FFMPEG

Follow the instructions here: https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/ffmpeg-with-nvidia-gpu/index.html build and install ffmpeg.

**AFTER** install, If there is an error with missing library path, do as follows:
```
sudo find / -name xxxxxxxxx.so.xx # find the library path using this, say /usr/local/lib, and change it in the next command
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/mylibs.conf; sudo ldconfig
```

Verify that FFMPEG works using `ffmpeg -version` making sure the cuda flag is enabled (See example below).

```
ffmpeg version N-116392-g53d0f9afb4 Copyright (c) 2000-2024 the FFmpeg developers
built with gcc 11 (Ubuntu 11.4.0-1ubuntu1~22.04)
configuration: --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
...
```

### PART 3: Setup Camera (Only Basler Cameras are Supported)

Install the latest version of Pylon using *.deb installer from: https://www2.baslerweb.com/en/downloads/software-downloads/#type=pylonsoftware;language=all;version=all;os=linuxx8664bit;series=all;model=all.
Also get the *.deb for the MPEG4 suppplementary package: https://www2.baslerweb.com/en/downloads/software-downloads/#type=pylonsupplementarypackageformpeg4;language=all;version=all;os=linuxx8664bit.

 **Prior to install, read all instructions from Bassler.**

Before running install run the following commands:
```
sudo apt install libgl1-mesa-dri libgl1-mesa-glx libxcb-xinerama0 libxcb-xinput0 libxcb-cursor0
```

#### CoaXPress Cameras
If you have a Coaxpress Camera make sure to install the supplementary package (additional instructions might be available on the 

```
sudo apt-get install /opt/pylon/share/pylon/menable-dkms_*.deb
```

#### GigE Cameras

Remember to run the following command before any camera's are connected.

```sudo apt install network-manager ethtool```

After camera is connected, go to network settings for the now connected ethernet adapter and set the IPv4 and IPv6 settings to "Local-Link Only".

Go to Pylon opt (default:/opt/pylon/bin) install and run the following commands:

```
sudo ./PylonGigEConfigurator list
```
Use this to find out which ethernet port is connected and shows an IP. Lets say its called 'eno1'.

```
sudo ./PylonGigEConfigurator auto-all -a "eno1"
```

### PART 4: Make sure Miniconda is Installed

Install Mambaforge (a faster version of Miniconda), by running the following command:

```
wget -nc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && bash Mambaforge-Linux-x86_64.sh -b && ~/mambaforge/bin/conda init bash
```

### PART 5: Configure poetry to recognize the conda environment

```
poetry config virtualenvs.path $CONDA_ENV_PATH
poetry config virtualenvs.create false
```
where `$CONDA_ENV_PATH` is the location of the conda envs, usually `/home/username/miniforge3/envs` but can be verified by running `conda info --envs`. If the environment variable is not defined, make sure to define it and add it to ~/.bashrc first.

### PART 6: Clone the Repository

Move to the directory where you want to clone the repository and run the following commands:

```
git clone https://github.com/neurorishika/FlyProjection.git
cd FlyProjection
```

### PART 7: Create the Conda Environment

Start by creating the conda environment that includes cudatooolkit and cudnn (cross-check with [SLEAP](https://sleap.ai/) for the latest installation instructions).

```
conda create --name flyprojection pip python=3.9 cudatoolkit=11.3 cudnn=8.2
```

Activate the environment:

```
conda activate flyprojection
```

### PART 8: Install the Project Dependencies

Install the project dependencies using poetry:

```
poetry install
```

### PART 9: Install SLEAP

Install SLEAP using pip as per the instructions at: https://sleap.ai/

```
conda activate flyprojection # make sure the conda environment is activated
pip install sleap[pypi]==1.4.1a2 # or the latest version
```

Verify the installation by running `sleap-label` to see the GUI. Additioanlly, run `python -c "import sleap; sleap.versions()"` to see the versions of the installed packages.


## Project Organization

The project is organized as follows:
```
.DS_Store
.gitignore
LICENSE
README.md
analysis
   |-- .gitkeep
   |-- 20hr-wingless-orco-yy
   |   |-- analysis.ipynb
   |   |-- arena.json
   |   |-- video_gen.ipynb
   |   |-- yang_props.json
   |   |-- ying_props.json
   |-- 20hrs-wingless-orcoctrl-yy2024-04-26_13-24
   |   |-- analysis.ipynb
   |   |-- arena.json
   |   |-- video_gen.ipynb
   |   |-- yang_props.json
   |   |-- ying_props.json
   |-- OLD METHOD
   |   |-- analysis.ipynb
   |-- Thin-Trails_ORCO
   |   |-- 20hr-wingless-orco-tt
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- big_ring_props.json
   |   |   |-- small_ring_props.json
   |   |-- 20hr-wingless-orcoctrl-tt
   |   |   |-- analysis-archived-2.ipynb
   |   |   |-- analysis-archived.ipynb
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- big_ring_props.json
   |   |   |-- small_ring_props.json
   |-- archived
   |   |-- 20hr-wingless-orcoctrl-yy-BADTRACKING
   |   |   |-- analysis.ipynb
   |   |   |-- arena.json
   |   |   |-- video_gen.ipynb
   |   |   |-- yang_props.json
   |   |   |-- ying_props.json
   |-- process_bands.ipynb
   |-- process_thin_trails.ipynb
   |-- process_ying-yang-oc.ipynb
   |-- simulation
   |   |-- analysis.ipynb
   |   |-- simulation.ipynb
   |   |-- test.ipynb
configs
   |-- archived_configs
   |   |-- rig_config_20240416120834.json
   |   |-- rig_config_20240419111528.json
   |-- rig_config.json
data
   |-- .gitkeep
deprecated
   |-- flyprojection
   |   |-- controllers
   |   |   |-- camera.py
   |-- prereqs_for_pyspin
   |   |-- spinnaker-3.2.0.62-amd64
   |   |   |-- README
   |   |   |-- configure_gentl_paths.sh
   |   |   |-- configure_spinnaker.sh
   |   |   |-- configure_spinnaker_paths.sh
   |   |   |-- configure_usbfs.sh
   |   |   |-- install_spinnaker.sh
   |   |   |-- libgentl_3.2.0.62_amd64.deb
   |   |   |-- libspinnaker-c-dev_3.2.0.62_amd64.deb
   |   |   |-- libspinnaker-c_3.2.0.62_amd64.deb
   |   |   |-- libspinnaker-dev_3.2.0.62_amd64.deb
   |   |   |-- libspinnaker_3.2.0.62_amd64.deb
   |   |   |-- libspinvideo-c-dev_3.2.0.62_amd64.deb
   |   |   |-- libspinvideo-c_3.2.0.62_amd64.deb
   |   |   |-- libspinvideo-dev_3.2.0.62_amd64.deb
   |   |   |-- libspinvideo_3.2.0.62_amd64.deb
   |   |   |-- remove_spinnaker.sh
   |   |   |-- spinnaker-doc_3.2.0.62_amd64.deb
   |   |   |-- spinnaker_3.2.0.62_amd64.deb
   |   |   |-- spinupdate-dev_3.2.0.62_amd64.deb
   |   |   |-- spinupdate_3.2.0.62_amd64.deb
   |   |   |-- spinview-qt-dev_3.2.0.62_amd64.deb
   |   |   |-- spinview-qt_3.2.0.62_amd64.deb
   |   |-- spinnaker_python-3.2.0.62-cp310-cp310-linux_x86_64
   |   |   |-- Examples
   |   |   |   |-- Python3
   |   |   |   |   |-- AcquireAndDisplay.py
   |   |   |   |   |-- Acquisition.py
   |   |   |   |   |-- AcquisitionMultipleCamera.py
   |   |   |   |   |-- BufferHandling.py
   |   |   |   |   |-- ChunkData.py
   |   |   |   |   |-- CounterAndTimer.py
   |   |   |   |   |-- DeviceEvents.py
   |   |   |   |   |-- Enumeration.py
   |   |   |   |   |-- EnumerationEvents.py
   |   |   |   |   |-- Enumeration_QuickSpin.py
   |   |   |   |   |-- Exposure_QuickSpin.py
   |   |   |   |   |-- FileAccess_QuickSpin.py
   |   |   |   |   |-- ImageChannelStatistics.py
   |   |   |   |   |-- ImageEvents.py
   |   |   |   |   |-- ImageFormatControl.py
   |   |   |   |   |-- ImageFormatControl_QuickSpin.py
   |   |   |   |   |-- Inference.py
   |   |   |   |   |-- Injected_Image_Classification.raw
   |   |   |   |   |-- Injected_Image_Classification_Daisy.jpg
   |   |   |   |   |-- Injected_Image_Detection.raw
   |   |   |   |   |-- Injected_Image_Detection_Aeroplane.jpg
   |   |   |   |   |-- Logging.py
   |   |   |   |   |-- LookupTable.py
   |   |   |   |   |-- Network_Classification
   |   |   |   |   |-- Network_Detection
   |   |   |   |   |-- NodeMapCallback.py
   |   |   |   |   |-- NodeMapInfo.py
   |   |   |   |   |-- NodeMapInfo_QuickSpin.py
   |   |   |   |   |-- SaveToAvi.py
   |   |   |   |   |-- Sequencer.py
   |   |   |   |   |-- SpinUpdate.py
   |   |   |   |   |-- Trigger.py
   |   |   |   |   |-- Trigger_QuickSpin.py
   |   |   |-- README.txt
   |   |   |-- docs
   |   |   |   |-- FLIR Camera Getting Started.html
   |   |   |   |-- PySpinDoc.pdf
   |   |   |-- licenses
   |   |   |   |-- FFmpeg_compliance_doc.txt
   |   |   |   |-- Spinnaker-Open-Source-Licenses.pdf
   |   |   |-- spinnaker_python-3.2.0.62-cp310-cp310-linux_x86_64.whl
experiments
   |-- dual_band
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- dual_trail
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- multi_trail_ece
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- multi_trail_mov
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- ortho_circle
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- patches
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- random_flash
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- trail_test
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- yin_yang
   |   |-- config.py
   |   |-- experiment_logic.py
   |-- yy_oc
   |   |-- config.py
   |   |-- experiment_logic.py
flyprojection
   |-- __init__.py
   |-- app
   |   |-- webapp.py
   |-- config.py
   |-- controllers
   |   |-- __init__.py
   |   |-- basler_camera.py
   |   |-- camera.py
   |   |-- spinnaker_camera.py
   |-- experiment_logic.py
   |-- main.py
   |-- reanalysis.py
   |-- rig-reconfig.py
   |-- utils.py
   |-- utils
   |   |-- rdp_client.py
poetry.lock
poetry.toml
processed_data
   |-- .gitkeep
push_script.sh
pyproject.toml
scripts
   |-- .gitkeep
tests
   |-- GUItest.py
   |-- __init__.py
   |-- basler_camera.py
   |-- camera_test.ipynb
   |-- configurationeventprinter.py
   |-- error.txt
   |-- fps.npy
   |-- gpu_test.ipynb
   |-- imageeventprinter.py
   |-- spinnaker_camera.py
utils
   |-- build.py
   |-- quickstart.py
   |-- update.py
```
