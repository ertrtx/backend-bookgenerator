# backend-bookgenerator
This is the repo for the back-end component of the Baby Book generator app.

## Summary
The back-end is architected as a REST API (Flask-based app) that can receive requests. The request includes a user prompt which serves as an input to a text generator function.  The text generator outputs a short story parsed into sentences and corresponding noun prompts derived from each sentence. Noun prompts are passed to an image generator function to create images. The originating prompt and sentence are attached to each image as metadata. Images are uploaded as blob objects to a GCS bucket.

### Text generator
GPT-J-6B


### Image generator

Implementation of Imagen by @cene555


## Hardware 
GCP Deep Learning VM

Pytorch GPU image

| GPU model | Machine Type | GPUs | GPU memory | Available vCPUs | Available memory |
|---------- | ------------ | -----| ---------- | --------------- | ---------------- |
| NVIDIA A100| a2-highgpu-1g |	1 GPU	|40 GB HBM2|	12 vCPUs|	85 GB |





## Installation requirements
- Install GPU driver
- Check driver installation
```
nvidia-smi
gcc --version
nvcc --version
```
- Create new conda environment, install Pytorch & CUDA dependencies, and test installation
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
python test_gpu_torch_install.py 
```
- Install modules for text generator
```
pip install transformers
conda install spacy
python -m spacy download en_core_web_sm
```
-  Check version of Linux on VM and install Git LFS
```
uname -r
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar -xf git-lfs-linux-amd64-v3.2.0.tar.gz
cd git-lfs-3.2.0/
sudo ./install.sh
```
- Install modules for image generator
```
git lfs install
git clone https://huggingface.co/Cene655/ImagenT5-3B
pip install git+https://github.com/cene555/Imagen-pytorch.git
pip install git+https://github.com/openai/CLIP.git
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```
- Install flask modules
```
pip install Flask
pip install flask-restful
```



