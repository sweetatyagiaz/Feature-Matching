# DINOv2: A Self-supervised Vision Transformer Model

## Overview

DINOv2 is a self-supervised training method from Meta Research that learns rich visual features directly from images without requiring any labels, removing one of the most time-consuming bottlenecks in building computer vision models. Pre-trained checkpoints support depth estimation, semantic segmentation, image classification, and instance retrieval, and the architecture produces dense embeddings that can underpin a wide range of downstream applications. The post covers how the method works, what tasks it supports out of the box, and how to get started with the available weights ranging from 84 MB to 4.2 GB.

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

## Project Structure

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


