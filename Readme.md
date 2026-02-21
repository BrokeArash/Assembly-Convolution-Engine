🧠 Assembly Convolution Engine & CNN-Based Brain Tumor Detection

A low-level, performance-oriented project combining x86 assembly, C, and Python to build and benchmark convolution operations and deploy a CNN-based brain tumor detector for MRI images.

This project was originally developed for a Computer Structure and Language course and later extended into a complete end-to-end neural inference pipeline, suitable for showcasing systems-level optimization, ML fundamentals, and cross-language engineering.

📌 Project Overview

This repository contains three main components:

1️⃣ Convolution Engine (C vs x86 Assembly)

Manual implementation of 2D convolution and max-pooling

Two versions:

Pure C implementation

Optimized x86 Assembly implementation

Benchmarking to compare:

Execution time

Performance gains from low-level optimization

📊 Python scripts are used to visualize and plot benchmark results.

2️⃣ Pattern Recognition (Classical ML Pipeline)

Image dataset preprocessing

Feature extraction using convolution + pooling

Simple classification logic

Focus on understanding how convolution enables pattern recognition at a low level

This section bridges signal processing concepts with machine learning fundamentals.

3️⃣ CNN-Based Brain Tumor Detection (End-to-End)

A custom CNN trained on MRI brain images

Implemented using:

PyTorch (training & weight export)

C inference engine (no PyTorch at runtime)

The trained network is executed entirely in C + Assembly

Outputs a tumor probability score for MRI images

🧠 The CNN architecture:

2× Convolution layers

ReLU activations

Max-pooling

Fully-connected layers

Sigmoid output for binary classification (tumor / no tumor)

🛠 Technologies Used

Languages

C

x86 Assembly

Python

Libraries / Tools

PyTorch (training only)

stb_image (image loading)

GCC

NumPy / Matplotlib (benchmark plots)

Architecture

x86 (Linux / WSL compatible)

📂 Repository Structure
Assembly-Convolution-Engine/
│
├── cnn.c                    # CNN layers in C
├── infer.c                  # End-to-end inference executable
├── convolution.asm          # Optimized x86 convolution
├── fast_maxpool.asm         # Optimized x86 max-pooling
│
├── cnn_weights/
│   └── c_arrays/            # Exported CNN weights as C arrays
│
├── dataset/                 # MRI brain image dataset
├── images/                  # Test images for inference
│
├── benchmarks/              # Performance test scripts
├── train_cnn.py             # CNN training script (PyTorch)
└── README.md

🚀 How to Build & Run
Compile Inference Engine
gcc infer.c cnn.c convolution.o maxpool.o -O2 -lm

Run Tumor Detection
./a.out images/input_tumor.jpg


Example output:

🧠 Brain Tumor Analysis
----------------------
Tumor probability: 87.42%
Diagnosis: ❗ TUMOR DETECTED

📈 Performance Benchmarking

Assembly implementation significantly outperforms the C version

Demonstrates:

Cache-aware memory access

Reduced loop overhead

SIMD-friendly design (where applicable)

Benchmark results are visualized using Python.

🎓 Academic Context

Course: Computer Structure and Language

Focus Areas:

Low-level performance optimization

Understanding CNN internals

Cross-language system design

Practical application of ML without heavy frameworks

💼 Why This Project Matters (For Recruiters)

This project demonstrates:

✅ Strong understanding of computer architecture

✅ Ability to optimize code at the assembly level

✅ Deep understanding of CNN internals

✅ Bridging ML theory with systems programming

✅ Clean separation between training and inference

✅ Real-world application (medical imaging)

⚠️ Disclaimer

This project is educational and not intended for medical diagnosis.
It should not be used in clinical or healthcare settings.

👤 Author

Arash
Computer Engineering
Focus: Systems Programming, Performance Optimization, Machine Learning