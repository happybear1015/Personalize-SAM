# Personalize Segment Anything with 1 Shot in 10 Seconds

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/personalize-segment-anything-model-with-one/personalized-segmentation-on-perseg)](https://paperswithcode.com/sota/personalized-segmentation-on-perseg?p=personalize-segment-anything-model-with-one)

Official implementation of ['Personalize Segment Anything Model with One Shot'](https://arxiv.org/pdf/2305.03048.pdf).

💥 Try out the [web demo](https://huggingface.co/spaces/justin-zk/Personalize-SAM) 🤗 of PerSAM and PerSAM-F: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/justin-zk/Personalize-SAM)


🎉 Try out the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PerSAM) in colab for your own dataset. Great thanks to [@NielsRogge](https://github.com/NielsRogge)!


## News
* **TODO**: Release the PerSAM-assisted [Dreambooth](https://arxiv.org/pdf/2208.12242.pdf) for better fine-tuning [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 📌.
* We release the code of PerSAM and PerSAM-F 🔥. Check our [video](https://www.youtube.com/watch?v=QlunvXpYQXM) here!
* We release a new dataset for personalized segmentation, [PerSeg](https://drive.google.com/file/d/18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio/view?usp=sharing) 🔥.

## Introduction
*How to customize SAM to automatically segment your pet dog in a photo album?*

In this project, we propose a training-free **Per**sonalization approach for [Segment Anything Model (SAM)](https://ai.facebook.com/research/publications/segment-anything/), termed as **PerSAM**. Given only a single image with a reference mask, PerSAM can segment specific visual concepts, e.g., your pet dog, within other images or videos without any training. 
For better performance, we further present an efficient one-shot fine-tuning variant, **PerSAM-F**. We freeze the entire SAM and introduce two learnable mask weights, which only trains **2 parameters** within **10 seconds**. 
在这个项目中，我们提出了一种无需训练的个性化方法，用于Segment Anything Model (SAM) 的分割。这个方法被称为PerSAM。给定只有一张带有参考掩码的图像，PerSAM可以在其他图像或视频中分割特定的视觉概念，例如您的宠物狗，而无需进行任何训练。为了获得更好的性能，我们进一步提出了一种高效的单次微调变体，即PerSAM-F。我们冻结整个SAM，并引入两个可学习的掩码权重，仅在10秒内训练2个参数。



<div align="center">
  <img src="figs/fig_persam.png"/ width="97%"> <br>
</div>

Besides, our approach can be utilized to assist [DreamBooth](https://arxiv.org/pdf/2208.12242.pdf) in fine-tuning better [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for personalized image synthesis. We adopt PerSAM to segment the target object in the user-provided few-shot images, which eliminates the **background disturbance** and benefits the target representation learning.

<div align="center">
  <img src="figs/fig_db.png"/ width="97%"> <br>
</div>

## Requirements
### Installation
Clone the repo and create a conda environment:
```bash
git clone https://github.com/ZrrSkywalker/Personalize-SAM.git
cd Personalize-SAM

conda create -n persam python=3.8
conda activate persam

pip install -r requirements.txt
```

Similar to Segment Anything, our code requires `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.



### Preparation
Please download our constructed dataset **PerSeg** for personalized segmentation from [Google Drive](https://drive.google.com/file/d/18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1X-czD-FYW0ELlk2x90eTLg) (code `222k`), and the pre-trained weights of SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Then, unzip the dataset file and organize them as
```
data/
|–– Annotations/
|–– Images/
sam_vit_h_4b8939.pth
```

## Getting Started

### Personalized Segmentation

For the training-free 🧊 **PerSAM**, just run:
```bash
python persam.py --outdir <output filename>
```

For 10-second fine-tuning of 🚀 **PerSAM-F**, just run:
```bash
python persam_f.py --outdir <output filename>
```

For **Multi-Object** segmentation of the same category by PerSAM-F (Great thanks to [@mlzoo](https://github.com/mlzoo)), just run:
```bash
python persam_f_multi_obj.py --outdir <output filename>
```

After running, the output masks and visualzations will be stored at `outputs/<output filename>`. 

### Evaluation
Then, for mIoU evaluation, please run:
```bash
python eval_miou.py --pred_path <output filename>
```

### Personalized Stable Diffusion
Our approach can enhance DreamBooth to better personalize Stable Diffusion for text-to-image generation.

Comming soon.

## Citation
```bash
@article{zhang2023personalize,
  title={Personalize Segment Anything Model with One Shot},
  author={Zhang, Renrui and Jiang, Zhengkai and Guo, Ziyu and Yan, Shilin and Pan, Junting and Dong, Hao and Gao, Peng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2305.03048},
  year={2023}
}
```

## Acknowledgement
This repo benefits from [Segment Anything](https://github.com/facebookresearch/segment-anything) and [DreamBooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
