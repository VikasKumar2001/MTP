# QRFusionCAD: QR-Guided Feature Extraction from 3D CAD Models

> A research project that converts 3D CAD models into compact QR representations by extracting geometric, structural, and semantic features from STL models.

---

## Overview

QRFusionCAD is an AI-assisted CAD understanding framework that transforms a 3D CAD model into an intelligent feature representation. Instead of storing the complete CAD model, the system extracts meaningful geometric and semantic information and compresses it into a QR code.

This repository implements the **Phase-I** of the proposed framework:

```
CAD Model → Feature Extraction → QR Generation
```

The generated QR code contains a compressed latent representation of the CAD model, which will later be used for CAD reconstruction and context-aware CAD generation using Large Language Models (LLMs).

---

## Motivation

Modern CAD files are large and difficult to share efficiently. Existing approaches either generate CAD models from scratch or require complete CAD files for reconstruction.

The goal of this project is to develop a lightweight representation of CAD models by:

- Extracting geometric features
- Understanding structural components
- Generating semantic information
- Compressing important information
- Encoding it into a portable QR representation

---

# Project Pipeline

```
               STL CAD Model
                      │
                      ▼
               Geometry Loader
                      │
                      ▼
            3D Slicing Engine
                      │
                      ▼
       Structural Feature Extraction
                      │
                      ▼
       Semantic Feature Intelligence
                      │
                      ▼
      Feature Compression (zlib)
                      │
                      ▼
             QR Code Generator
```

---

# Project Structure

```
QRFusionCAD/
│
├── Data/
│   ├── chair_1.stl
│   ├── chair_2.stl
│   └── ...
│
├── results/
│   └── <model_name>/
│       ├── summary.json
│       ├── slice_result.json
│       ├── features.json
│       ├── slice_graph.png
│       └── latent_qr.png
│
├── load.py
├── slicer.py
├── feature.py
├── qr_encoder.py
├── main.py
└── README.md
```

---

# Features

- STL Model Loading
- Geometry Analysis
- 3D Layer-wise Slicing
- Slice Signature Generation
- Structural Feature Detection
- Chair Type Classification
- Feature Vector Generation
- Feature Compression
- QR Code Generation

---

# Workflow

## 1. Load STL Model

The system loads STL models using **Trimesh** and extracts:

- Bounding Box
- Dimensions
- Surface Area
- Volume
- Triangle Count
- Vertex Count
- Curvature Statistics
- Mesh Centroid
- Watertightness

---

## 2. Slice the CAD Model

The CAD model is sliced into **1000 horizontal layers**.

For every slice, the system computes:

- Cross-sectional Area
- Perimeter
- Occupancy
- Slice Centroid

These slices are later used to understand the structure of the object.

---

## 3. Structural Feature Extraction

From the slicing information, the system estimates:

- Seat Height
- Backrest Detection
- Armrest Detection
- Backrest Angle
- Leg Configuration
- Stability Score
- Ergonomic Score

These features help describe the object beyond raw geometry.

---

## 4. Semantic Feature Intelligence

The extracted geometry is converted into semantic information.

Current outputs include:

- Chair Type
- Material Estimation
- Dimensions
- Volume
- Surface Area
- Stability Score
- Ergonomic Score
- Slice Signature
- Feature Vector

Supported chair categories:

- Office Chair
- Hospital Chair
- Dining Chair
- Lounge Chair
- Stool
- Wheelchair

---

## 5. Feature Compression

Only the information required for future CAD reconstruction is retained.

The current compressed payload consists of:

```json
{
    "v": "Feature Vector",
    "s": "Slice Signature",
    "t": "Chair Type"
}
```

The payload is:

1. Converted to JSON
2. Compressed using **zlib**
3. Encoded using **Base64**

---

## 6. QR Code Generation

The compressed payload is converted into a QR code.

Unlike conventional QR codes that store URLs or text, this QR stores a compact latent representation of the CAD model.

Example output:

```
results/chair_01/
    latent_qr.png
```

---

# Generated Outputs

For every processed CAD model the following files are generated:

| File | Description |
|------|-------------|
| summary.json | Geometry information |
| slice_result.json | Layer-wise slicing analysis |
| features.json | Extracted semantic features |
| slice_graph.png | Slice analysis visualization |
| latent_qr.png | QR representation of compressed features |

---

# Technologies Used

- Python
- Trimesh
- NumPy
- Shapely
- Matplotlib
- numpy-stl
- qrcode
- zlib
- Base64
- JSON

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/QRFusionCAD.git
cd QRFusionCAD
```

Install dependencies

```bash
pip install trimesh numpy numpy-stl shapely matplotlib qrcode pillow
```

---

# Usage

Place STL models inside:

```
Data/
```

Run:

```bash
python main.py
```

Results will automatically be generated inside:

```
results/
```

---

# Current Progress

- ✅ STL Model Loading
- ✅ Geometry Extraction
- ✅ CAD Slicing
- ✅ Structural Feature Detection
- ✅ Semantic Feature Extraction
- ✅ Feature Vector Generation
- ✅ Feature Compression
- ✅ QR Code Generation

---

# Future Work

The next phase of this research will include:

- QR Code Decoding
- Feature Embedding Retrieval
- Vector Database Integration
- Large Language Model Integration
- Context-aware CAD Editing
- Synthetic CAD Generation
- CAD Reconstruction from QR Representation
- Natural Language Guided CAD Design

Future workflow:

```
Existing CAD
      │
      ▼
Feature Extraction
      │
      ▼
QR Encoding
      │
      ▼
Scan QR
      │
      ▼
Feature Retrieval
      │
      ▼
Large Language Model
      │
      ▼
Synthetic Editable CAD Model
```

---

# Research Objective

The long-term vision of **QRFusionCAD** is to create an intelligent CAD ecosystem where physical objects can be linked to their digital representations through QR codes. These QR-encoded features will enable AI systems to reconstruct, modify, and generate context-aware CAD models using natural language instructions.

Example:

```
Office Chair
      │
      ▼
Generate QR
      │
      ▼
Scan QR
      │
      ▼
"Convert into Hospital Chair"
      │
      ▼
Synthetic CAD Model
```

---

# License

This project is intended for research and academic purposes as part of an M.Tech thesis in Robotics and Artificial Intelligence.
