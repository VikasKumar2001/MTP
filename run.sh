#!/bin/bash

set -e

echo "================================================================="
echo "       MTP PIPELINE: 3D CAD ENCODING & RECONSTRUCTION"
echo "================================================================="
echo ""

echo "[1/5] PREPARING DATASET..."
echo "Extracting raw features and sampling point clouds..."
python 1_extract_features.py

echo ""
echo "Preparing the dataset for autoencoder training..."
python training/prepare_dataset.py

echo ""

echo "[2/5] TRAINING AUTOENCODER..."
echo "Training the 1925-D -> 512-D encoder..."
python training/train_autoencoder.py

echo ""

echo "[3/5] PHASE 1: ENCODING & QR GENERATION..."
echo "Extracting descriptors, using trained encoder, and generating QR codes..."
python main_encode.py

echo ""

echo "[4/5] MERGING LATENT QR CODES..."
echo "Merging all generated QR codes..."
python QR_similarity/merger.py

echo ""

echo "[5/5] LATENT QR SIMILARITY SEARCH..."
echo "Running cosine similarity search on the latent space..."
python QR_similarity/qr_similarity.py

echo ""

echo "================================================================="
echo "               PIPELINE EXECUTION COMPLETE"
echo "================================================================="