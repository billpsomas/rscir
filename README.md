# Composed Image Retieval for Remote Sensing
Official PyTorch implementation and benchmark dataset for IGARSS 2024 ORAL paper. [[`arXiv`](https://arxiv.org/abs/2405.15587)]

<div align="center">
  <img width="100%" alt="WeiCom" src=".github/method.PNG">
</div>

## Overview

### Motivation

In recent years, earth observation (EO) through remote sensing (RS) has witnessed an enormous growth in data volume, creating a challenge in managing and extracting relevant information. Remote sensing image retrieval (RSIR), which aims to search and retrieve images from RS image archives, has emerged as a key solution. However, RSIR methods encounter a major limitation:
the reliance on a query of single modality. This constraint often restricts users from fully expressing their specific requirements. 

### Approach

To tackle this constraint, we introduce a new task, remote sensing composed image retrieval. RSCIR, integrating both image and text in the search query, is designed to retrieve images that are not only visually similar to the query image but also relevant to the details of the accompanying query text. Our RSCIR approach, called WeiCom, is expressive, flexible and training-free based on a vision-language model, utilizing a weighting parameter λ for more image- or text-oriented results, with λ → 0 or λ → 1 respectively.

<div align="center">
  <img width="100%" alt="lambda" src=".github/teaser.PNG">
</div>

In this work, we recognize, present and qualitatively evaluate the capabilities and challenges of RSCIR. We demonstrate how users can now pair a query image with a query text specifying modifications related to color, context, density, existence, quantity, shape, size or texture of one or more classes.

<div align="center">
  <img width="100%" alt="Attributes" src=".github/attr_use_cases_v2.png">
</div>

Quantitatively, we focus on color, context, density, existence, quantity, and shape
modifications, establishing a new benchmark dataset, called PatterCom and an evaluation protocol.

### Contributions

In summary, we make the following contributions:
- We introduce remote sensing composed image retrieval (RSCIR), accompanied with PatterCom, a new benchmark dataset.
- We introduce WeiCom, a training-free method utilizing a modality control parameter for more image- or text-oriented results according to the needs of each search.
- We evaluate both qualitatively and quantitatively the performance of WeiCom, setting the state-of-the-art on RSCIR.

## Pre-trained models

For our experiments, you need to download [CLIP]() and [RemoteCLIP](), both with a ViT-L/14 image encoder. After downloading, place them inside the `models/` folder.

## Environment

Create this environment for our experiments:

```bash
conda create -n rscir python=3.9 -y
conda activate rscir
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install open_clip_torch
```

## Dataset

PatterCom is based on [PatterNet](https://sites.google.com/view/zhouwx/dataset), a large-scale, high-resolution remote sensing dataset that comprises 38 classes, with each class containing 800 images of 256×256 pixels. 

Download PatterNet from [here](https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977) and unzip it into `PatterNet/` folder. Download `patternnet.csv` for [here](https://drive.google.com/file/d/1sIdH0DVR2JxCEYQgp8041sUOYfLdGgyV/view?usp=sharing) and place it in the same folder too. Finally, download PatternCom from [here](https://drive.google.com/drive/folders/1NP2Ryj4V2L_wwInQB6HjzPDPRH2k0J5U?usp=sharing) and place it into the same folder too.

The `PatterNet/` folder structure should look like this:

```
PatterNet/
    |-- images/
    |-- PatternCom/
        |-- color.csv
        |-- context.csv
        |-- density.csv
        |-- existence.csv
        |-- quantity.csv
        |-- shape.csv
    |-- patternnet.csv
    |-- patternnet_description.pdf
```

## Feature Extraction

To extract CLIP or RemoteCLIP features from PatternNet dataset, run:

```python
python extract_features.py --model_name clip --dataset_path /path/to/PatternNet/
```

Replace `clip` with `remoteclip` for RemoteCLIP features.

Note that this will save features as pickle files inside `PatterNet/features/` folder. Thus, the new folder structure should look like this:

```
PatterNet/
    |-- features/
        |-- patternnet_clip.pkl
        |-- patternnet_remoteclip.pkl
    |-- images/
    |-- PatternCom/
    |-- patternnet.csv
    |-- patternnet_description.pdf
```

## Evaluation

