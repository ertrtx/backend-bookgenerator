# backend-bookgenerator
This is the repo for the back-end component of the Baby Book generator app.

### Summary
The back-end is architected as a REST API (Flask-based app) that can receive requests. The request includes a user prompt which serves as an input to a text generator function.  The text generator outputs a short story parsed into sentences and corresponding noun prompts derived from each sentence. Noun prompts are passed to an image generator function to create images. The originating prompt and sentence are attached to each image as metadata. Images are uploaded as blob objects to a GCS bucket.

### Text generator
GPT-J-6B


### Image generator



### Hardware 
GCP Deep Learning VM

Pytorch GPU image

| GPU model | Machine Type | GPUs | GPU memory | Available vCPUs | Available memory |
|---------- | ------------ | -----| ---------- | --------------- | ---------------- |
| NVIDIA A100| a2-highgpu-1g |	1 GPU	|40 GB HBM2|	12 vCPUs|	85 GB |


### Installation requirements
- Install GPU driver
- Check driver installation
```
nvidia-smi
gcc --version
nvcc --version
```
- Create new conda environment, install Pytorch/CUDA dependencies, and test modules
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
python test_gpu_toch_install.py 
```




