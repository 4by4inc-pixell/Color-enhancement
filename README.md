# Color-enhancement

## Requirements
- CUDA 11.8
- CUDNN 8.9
- Python 3.9

## Installation

- Make Conda Environment
```bash
conda create -n ColEn python=3.9 -y
conda activate ColEn
```
- Install Dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c nvidia cudatoolkit=11.8 cudnn=8.9 -y

pip install -r requirements.txt

```

<!--
## Download Pre-trained Models

Download pytorch models: [Pytorch model](https://drive.google.com/file/d/12kmIeep10ok5y0SRfWQsQjikMUfY2to6/view?usp=sharing).



Download onnx models: [ONNX model](https://drive.google.com/file/d/1VZJf6QmqvIm1DJEXWO6kBfSzgl73FpvF/view?usp=sharing).
-->

## Test

Run the following command for test:

- Pytorch model video test:
```bash
python pytorch_video_test.py --input<input video path> --output_dir<generated color enhance video path> --ckpt<pytorch color-enhancement model> --gpu_ids<ex)0,1,2,3>
```

- ONNX model video test:
```bash
python onnx_video_test.py --input<input video path> ---output_dir<generated color enhance video path> --onnx_step<onnx color-enhancement model> --gpu_ids<ex)0,1,2,3>
```


