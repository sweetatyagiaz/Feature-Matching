# DINOv2: A Self-supervised Vision Transformer Model

## Overview

**DINOv2 (Self-Supervised Vision Transformer)** is a self-supervised training method from **Meta Research** that learns rich visual features directly from images without requiring any labels, removing one of the most time-consuming bottlenecks in building computer vision models. Pre-trained checkpoints support depth estimation, semantic segmentation, image classification, and instance retrieval, and the architecture produces dense embeddings that can underpin a wide range of downstream applications. The post covers how the method works, what tasks it supports out of the box, and how to get started with the available weights ranging from 84 MB to 4.2 GB.

This project explores the use of **DINOv2** for image feature extraction, image similarity search, face recognition, and visual retrieval tasks.

DINOv2 generates highly discriminative image embeddings without requiring task-specific training. These embeddings can be indexed using FAISS for large-scale similarity search and recognition systems.

---

## Key Features

* Image Feature Extraction using DINOv2
* Face Feature Representation
* Image-to-Image Similarity Search
* Partial Image Matching
* FAISS Integration
* Large Scale Image Retrieval
* Person Identification
* Duplicate Image Detection
* Image Clustering

---

## How Does DINOv2 Work?

DINOv2 leverages a technique called self-supervised learning, where a model is trained using images without labels. There are two major benefits of a model not requiring labels.

First, a model can be trained without investing the significant time and resources to label data. Second, the model can derive more meaningful and rich representations of the image input data since the model is trained directly on the image.

By training a model directly on the images, the model can learn all the context in the image. Consider the following image of solar panels:

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/data-src-image-a0e8a980-c0d3-4998-b219-1245ddc9972e.png" width="800">
</p>

We could assign this image a label such as “an aerial photograph of solar panels” but this misses out on a lot of the information in the image; documenting deeper knowledge for a large dataset is difficult. But, DINOv2 shows that labels are not necessary for many tasks such as classification: instead, you can train on the unlabelled images directly.

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/data-src-image-314e15d1-8445-412e-97ec-5f6a9944300c.png" width="800">
</p>

To prepare data for the model, Meta researchers used data from curated and uncurated data sources. These images were then embedded. The uncurated images were deduplicated, then the deduplicated images were combined with the curated images to create the initial dataset used for training the model.

Prior to DINOv2, a common method of building general computer vision models that embed semantics of an entire image has been using a pre-training step with image-text pairs. For example, OpenAI’s CLIP model, which can work on tasks from image retrieval to classification, was trained on 400 million image-text pairs.

CLIP learned semantic information about the contents of images based on the text pairs. DINOv2, in contrast, was trained on 142,109,386 images. The table below, featured in the DINOv2 paper, shows how the data was sourced for the dataset:

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/Screenshot-2023-05-31-at-09.04.45-1.png" width="800">
</p>

DINOv2 circumvents the requirement to have these labels, enabling researchers and practitioners to build large models without requiring a labeling phase.

---

## What Can We Do With DINOv2?

As a result of the increased information DINOv2 learns by training directly on images, the model has been found to perform effectively on many image tasks, including depth estimation, a task for which separate models are usually employed.

Meta AI researchers wrote custom model heads to accomplish depth estimation, **image classification** (using linear classification and KNN), **image segmentation**, and instance retrieval. There are no out-of-the-box heads available for depth estimation or segmentation, which means one would need to write a custom head to use them.

Let’s talk through a few of the use cases of the DINOv2 model.


### Depth Estimation

DINOv2 can be used for predicting the depth of each pixel in an image, achieving state-of-the-art performance when evaluated on the NYU Depth and SUN RGB-D depth estimation benchmark datasets. Meta Research created a depth estimation model with a DPT decoder for use in their repository, although this is not open source. Thus, it is necessary to write the code that uses the model backbone to construct a depth estimation model.

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/data-src-image-709dc95f-f2ae-436d-b14e-e8ff6245886f.png" width="800">
</p>


### Image Segmentation

DINOv2 is capable of segmenting objects in an image. Meta Research evaluated DINOv2 against the ADE20K and Cityscapes benchmarks and achieved “competitive results” without any fine-tuning when compared to other relevant models, according to the <a href="https://dinov2.metademolab.com/demos?category=segmentation&ref=blog.roboflow.com">instance segmentation example in the model playground</a>. There is no official image segmentation head that accompanies the repository.

Such code would need to be written manually in order to use DINOv2 for segmentation tasks.

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/data-src-image-82a71d16-49ca-419f-ab8b-6155560436dc.png" width="800">
</p>


### Classification

DINOv2 is suitable for use in image classification tasks. According to Meta Research, the performance of DINOv2 is “competitive or better than the performance of text-image models such as CLIP and OpenCLIP on a wide array of tasks”.

For instance, consider a scenario where you want to classify images of <a href="https://dinov2.metademolab.com/demos?category=segmentation&ref=blog.roboflow.com">vehicles on a construction site</a>. You could use DINOv2 to classify vehicles into specified classes using a nearest neighbor approach or linear classification.

With that said, a custom classifier head is needed to work with the DINOv2 embeddings.


### Instance Retrieval

DINOv2 can be used as part of an image information retrieval system that accepts images and returns related images. To do so, one would embed all of the images in a dataset. For each search, the provided image would be embedded and then images with a high cosine similarity to the embedded query image would be returned.

In the Meta Research playground accompanying DINOv2, there is a system that retrieves images related to landmarks. This is used to find art pieces that are similar to an image.

<p align="center">
  <img src="https://storage.ghost.io/c/2c/8d/2c8d8c0d-1c15-4b6d-825e-02b78d61d40a/content/images/2023/05/data-src-image-ed388222-d060-494e-843c-b8a2c1941894.png" width="800">
</p>

---

# Project Structure

```text
DINOv2/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── embeddings/
│   ├── faiss_index.bin
│   └── metadata.pkl
│
├── examples/
│   ├── extract_embeddings.py
│   ├── search_similar.py
│   └── recognize_person.py
│
├── src/
│   ├── models/
│   │   └── dinov2_model.py
│   │
│   ├── feature_extractors/
│   │   └── dinov2_extractor.py
│   │
│   ├── indexing/
│   │   ├── faiss_builder.py
│   │   └── faiss_search.py
│   │
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── visualization.py
│   │
│   └── config.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## DINOv2 Models

| Model    | Parameters | Embedding Size |
| -------- | ---------- | -------------- |
| ViT-S/14 | 22M        | 384            |
| ViT-B/14 | 86M        | 768            |
| ViT-L/14 | 300M       | 1024           |
| ViT-G/14 | 1.1B       | 1536           |

Recommended:

```text
dinov2_vitb14
```

for balanced speed and accuracy.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/DINOv2.git

cd DINOv2
```

### Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Packages

```bash
torch
torchvision
opencv-python
numpy
pillow
faiss-cpu
matplotlib
scikit-learn
tqdm
```

Install manually:

```bash
pip install torch torchvision
pip install opencv-python pillow numpy
pip install faiss-cpu
pip install matplotlib scikit-learn tqdm
```

---

## Loading DINOv2

```python
import torch

model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitb14'
)

model.eval()
```

---

## Extract Features

```python
from PIL import Image
import torch
from torchvision import transforms

image = Image.open("image.jpg")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = transform(image).unsqueeze(0)

with torch.no_grad():
    embedding = model(img)

print(embedding.shape)
```

Output:

```text
(1,768)
```

---

## Create FAISS Index

```python
import faiss

dimension = 768

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "faiss_index.bin"
)
```

---

## Search Similar Images

```python
query_embedding = get_embedding(query_image)

distance, ids = index.search(
    query_embedding,
    k=5
)

print(ids)
```

---

## Recommended Dataset Structure

For person recognition:

```text
database/
│
├── 1.Rakesh_Ranjan/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│
├── 2.Ajay_Kumar/
│   ├── img1.jpg
│   ├── img2.jpg
│
└── 3.Rahul_Sharma/
    ├── img1.jpg
    ├── img2.jpg
```

---

## Metadata Mapping

Store mapping separately:

```python
metadata = {
    0: {
        "person_id": 1,
        "name": "Rakesh Ranjan",
        "image": "img1.jpg"
    }
}
```

This allows:

* One person → Multiple images
* Multiple embeddings per person
* Easy identification after FAISS search

---

## Face Recognition Workflow

```text
Image
   │
   ▼
Face Detection
   │
   ▼
Face Cropping
   │
   ▼
DINOv2 Embedding
   │
   ▼
FAISS Search
   │
   ▼
Top-K Matches
   │
   ▼
Person Identification
```

---

## Partial Image Matching

DINOv2 works reasonably well for:

* Partial Face Matching
* Object Retrieval
* Scene Retrieval
* Similar Image Search

For very small cropped regions combine:

```text
DINOv2
+
SIFT
+
ORB
```

or

```text
DINOv2
+
SuperPoint
+
LightGlue
```

for improved robustness.

---

## Advantages

### Strengths

* No training required
* Strong image representation
* Fast feature extraction
* Works on unseen images
* Suitable for FAISS retrieval
* Excellent visual similarity search

### Limitations

* Not specifically trained for faces
* Weaker than ArcFace for identity verification
* Performance drops for extreme pose variations
* Sensitive to heavy occlusion

---

## Recommended Usage

### General Image Search

```text
DINOv2
```

### Face Recognition

```text
RetinaFace
      +
ArcFace
      +
FAISS
```

### Partial Image Search

```text
DINOv2
      +
SIFT
      +
LightGlue
```

### Large Surveillance Systems

```text
RetinaFace
      +
ArcFace
      +
FAISS
      +
DINOv2 Re-ranking
```

---

## Future Work

* Face Clustering
* Video Analytics
* Multi-Camera Tracking
* Person Re-Identification (ReID)
* Hybrid ArcFace + DINOv2 Retrieval
* Distributed FAISS Index
* Billion-Scale Image Search

---

## References

* DINOv2 Research Paper
* Facebook AI Research (FAIR)
* FAISS
* Vision Transformer (ViT)
* Self-Supervised Learning


