# Semantic Compression with Multimodal Foundation Models

Official implementation of the paper "[Compression Beyond Pixels: Semantic Compression with Multimodal Foundation Models]()"

## Overview

This repository contains code for performing semantic compression using multimodal foundation models. The approach compresses CLIP features rather than pixels, enabling more efficient storage and transmission while preserving semantic information.

## Key Components

### Core Files

- **pqvae_shared.py**: Core feature compression framework
- **train.py**: Training code (pre-trained models available in the `checkpoints` directory)
- **validate.py**: Evaluation code
- **extract_clip_features.py**: Script for extracting CLIP features from any dataset supported by torchvision
> **Note**: For convenience, we've already provided pre-extracted CLIP features for datasets mentioned in the paper in the `data/features` directory.

### Pre-trained Models

The `checkpoints` directory contains pre-trained PQVAE-shared models ready for use.

## Downstream Tasks

### Image Captioning

- **train_clipcap.py**: Train a captioning model using compressed CLIP features on COCO2014
- **predict_clipcap.py**: Generate captions from compressed CLIP features
  
> **Note**: Both the CLIP model and our CLIP feature compression framework are frozen during this task.

### Referring Object Identification

- **sam_clip.py**: Demonstrates that compressed CLIP features still work effectively for general object detection and identification
  
> **Note**: This is a plug-and-play model for direct inference - simply modify the designated section as needed.

## Quick Start

Detailed comments are also included in the code, so get started and enjoy!
 