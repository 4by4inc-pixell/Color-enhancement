# Color-enhancement

## Overview
Color-enhancement is a task that improves the color quality of images and videos that lack color information. 
This process aims to restore or improve the sharpness, saturation, and overall color balance of colors to create visually more natural and rich results. 
This model was developed to effectively perform these color-enhances.
To illustrate the model in detail, we decompose RGB input into YCbCr color space, denoise chrominance components (Cb, Cr) separately, and use spatial feature extraction blocks and global attention mechanisms to improve luminance features (Y). 
The model handles triple frames to promote temporal consistency, and optimizes output quality, color fidelity, and inter-frame consistency using a comprehensive set of loss functions including perceptual, MS-SSIM, HSV color, edge, and optical flow-based time loss.
The system supports PyTorch and ONNX workflows for large-scale applications and an efficient multi-GPU inference pipeline.
 
## Installation

- Make Conda Environment
```bash
conda create -n ColEn python=3.9 -y
conda activate ColEn
```
- Install Dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install onnx==1.14.0 / pip install onnxruntime-gpu==1.15.0

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm torchmetrics pytorch_msssim
```

## Download Pre-trained Models

Download pytorch models: [Pytorch model](https://drive.google.com/file/d/1bSs45h3zDqmimXkJAgueDzIUNG5I7KZQ/view?usp=sharing).
Download onnx models: [ONNX model](https://drive.google.com/file/d/1wyKLu4RdF-uBw-QJWNcV9AxreGFC9CnE/view?usp=sharing).

## File Paths

## Quantitative evaluation


<div align="center">
  
| Dataset - 300 images testset |  Pytorch model  |    ONNX model   |          
|:----------------------------:|:---------------:|:---------------:|
|         PSNR / SSIM          |  23.76 / 0.9540 |  23.76 / 0.9540 |
|       Inference time         |  350~400 ms/frame | 575~625 ms/frame |

</div>


## Qualitative evaluation

### Compare Model Outputs - Image 1

<table>
  <tr>
    <td><strong>Input</strong></td>
    <td><img src="https://github.com/user-attachments/assets/797781a6-68a4-47e9-a25a-c84c508d99a7" width="700"/></td>
  </tr>
  <tr>
    <td><strong>Version-0326</strong></td>
    <td><img src="https://github.com/user-attachments/assets/a2317926-6922-4008-a962-fa122cc556fc" width="700"/></td>
  </tr>
</table>

---


### Compare Model Outputs - Image 2

<table>
  <tr>
    <td><strong>Input</strong></td>
    <td><img src="https://github.com/user-attachments/assets/38f1b0e3-84e5-42e3-b799-6db49c4d8310" width="700"/></td>
  </tr>
  <tr>
    <td><strong>Version-0326</strong></td>
    <td><img src="https://github.com/user-attachments/assets/6126a620-77fa-4782-91d2-a1e162ed11b6" width="700"/></td>
  </tr>
</table>

---


### Compare Model Outputs - Image 3

<table>
  <tr>
    <td><strong>Input</strong></td>
    <td><img src="https://github.com/user-attachments/assets/b605a1bd-cfa8-4f30-a309-ea8b001f3eec" width="700"/></td>
  </tr>
  <tr>
    <td><strong>Version-0326</strong></td>
    <td><img src="https://github.com/user-attachments/assets/010c9134-f6e3-4e04-b496-8e0258b435bd" width="700"/></td>
  </tr>
</table>

---


### Compare Model Outputs - Image 4

<table>
  <tr>
    <td><strong>Input</strong></td>
    <td><img src="https://github.com/user-attachments/assets/d7242af7-067e-471c-8b65-bddda6df2103" width="700"/></td>
  </tr>
  <tr>
    <td><strong>Version-0326</strong></td>
    <td><img src="https://github.com/user-attachments/assets/bb6991de-089f-46f5-bfac-0729da5000b3" width="700"/></td>
  </tr>
</table>

---

### Compare Model Outputs - Image 5

<table>
  <tr>
    <td><strong>Input</strong></td>
    <td><img src="https://github.com/user-attachments/assets/d6a86b46-d85d-456b-b1df-10a0529c7986" width="700"/></td>
  </tr>
  <tr>
    <td><strong>Version-0326</strong></td>
    <td><img src="https://github.com/user-attachments/assets/429f94c0-5a04-4d30-83ae-1c61238d2a82" width="700"/></td>
  </tr>
</table>



## Qualitative evaluation

### Example 1

<div align="center">
  <img src="https://github.com/user-attachments/assets/950cc092-a728-4efb-824a-cda1b6ddef62" width="1200"/><br/>
  <strong>Input: original</strong>
</div>

<br/>

<div align="center">
  <img src="https://github.com/user-attachments/assets/f980d2c0-e9d5-4c52-98e6-1ade92473652" width="1200"/><br/>
  <strong>Output: enhanced</strong>
</div>

---

### Example 2

<div align="center">
  <img src="https://github.com/user-attachments/assets/c4359cdb-a16f-4ee1-a0ac-50e295916763" width="1200"/><br/>
  <strong>Input: original</strong>
</div>

<br/>

<div align="center">
  <img src="https://github.com/user-attachments/assets/257cae13-1da5-4dd1-8519-15f47b00dbf8" width="1200"/><br/>
  <strong>Output: enhanced</strong>
</div>

---

### Example 3

<div align="center">
  <img src="https://github.com/user-attachments/assets/727575d4-f0b8-4ef3-8d50-33fd21653efc" width="1200"/><br/>
  <strong>Input: original</strong>
</div>

<br/>

<div align="center">
  <img src="https://github.com/user-attachments/assets/8d3a1e3f-9800-4e0a-8866-5cd9f80566ed" width="1200"/><br/>
  <strong>Output: enhanced</strong>
</div>


