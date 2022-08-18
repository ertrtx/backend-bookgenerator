# backend-bookgenerator
This is the repo for the back-end component of the Baby Book generator app.

## Summary
The back-end is architected as a REST API (Flask-based app) that can receive requests. The request includes a user prompt which serves as an input to a text generator function.  The text generator outputs a short story parsed into sentences and corresponding noun prompts derived from each sentence. Noun prompts are passed to an image generator function to create images. The originating prompt and sentence are attached to each image as metadata. Images are uploaded as blob objects to a GCS bucket.

### Text generator
[GPT-J-6B](https://github.com/kingoflolz/mesh-transformer-jax/). Based on the example implementation with HuggingFace's Transformer library by [@mallorbc](https://github.com/mallorbc/gpt-j-6b).


### Image generator
Implementation of [Imagen](https://imagen.research.google/) by [@cene555](https://github.com/cene555/Imagen-pytorch).


## Hardware 
GCP Deep Learning VM

Pytorch GPU image

| GPU model | Machine Type | GPUs | GPU memory | Available vCPUs | Available memory |
|---------- | ------------ | -----| ---------- | --------------- | ---------------- |
| NVIDIA A100| a2-highgpu-1g |	1 GPU	|40 GB HBM2|	12 vCPUs|	85 GB |


## Installation requirements
- Install GPU driver and test
```
nvidia-smi
gcc --version
nvcc --version
```
- Create new conda environment, install Pytorch & CUDA dependencies, and [test installation](https://stackoverflow.com/a/70946283)
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
- Install modules for image generator ([original notebook](https://github.com/cene555/Imagen-pytorch/blob/main/notebooks/Imagen_pytorch_inference_new.ipynb))
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
## Running the app
```
conda activate <new env>
python flask_api.py
```

## Troubleshooting
Installing the GPU NVIDIA driver on the VM instance was problematic. The VM seemed to lose the driver's configuration once it was restarted. To address this problem, we ran GCP's script to manually reinstall the driver and modified it by commenting out the following statement `if check_driver_installed() and not args.force:` ...
```
sudo lsmod | grep -i nvidia
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
# modify script
sudo python3 install_gpu_driver.py
```


