# Assembly-Powered Image Processing Engine

[![Assembly](https://img.shields.io/badge/Assembly-x86--64-blue.svg)](https://en.wikipedia.org/wiki/X86_assembly_language)
[![C](https://img.shields.io/badge/C-99-green.svg)](https://en.wikipedia.org/wiki/C99)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)

> **High-performance image processing and pattern recognition using hand-optimized x86-64 assembly with SIMD instructions**

A comprehensive computer vision system demonstrating the power of low-level optimization, featuring assembly-accelerated convolution, real-time pattern recognition, and CNN-based medical image analysis.

---

## 🚀 Key Features

- **⚡ 5-6x Performance Boost**: SIMD-optimized convolution outperforms pure C implementation
- **🎯 Real-time Pattern Recognition**: Mnist number recognition
- **🧠 Medical AI**: CNN for brain tumor detection in MRI scans
- **📊 Comprehensive Benchmarking**: Detailed performance analysis and visualization
- **🔧 Production-Ready**: Clean, documented, and maintainable assembly code

---

## 📁 Project Structure

```
Assembly-Convolution-Engine/
├── Part 1: SIMD Convolution & Benchmarking
│   ├── convolution.asm          # Hand-optimized SSE assembly
│   ├── main.c                   # Performance comparison
│   └── visualize_benchmark.py   # Performance graphs
│
├── Part 2: Pattern Recognition System
│   ├── mnist_detection.c        # Main number detection
│   ├── mnist_prepare.py         # Training dataset & creating test input
│   └── dataset/qmnist           # Training images (120000)
│
├── Part 3: Medical Image Analysis
│   ├── train_cnn.py             # CNN architecture & training
│   ├── export_weights.py        # Exports data to C headers
│   ├── cnn.c                    # Functions we need
│   ├── fast_maxpool.asm         # Assembly maxpool function
│   ├── tumor_detection.c        # Main C code
│   └── cnn_weights/             # Trained model weights

```

---

## 🛠️ Technologies

### Core Implementation
- **x86-64 Assembly**: SIMD instructions (SSE4.1) for parallel processing
- **C (C99)**: System integration and benchmarking framework
- **Python 3.8+**: Data visualization and CNN training

### Key Libraries
- **Image Processing**: `stb_image`, OpenCV
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: NumPy, SciPy

---

## ⚡ Performance Highlights

### Convolution Benchmark
| Implementation | Time (100 iterations) | Speedup |
|----------------|----------------------|---------|
| Pure C (O3)    | 4.86 sec             | 1.0x    |
| Assembly SIMD  | 0.81 sec             | **5.9x**|


### Pattern Recognition
- **Classification Accuracy**: 95.2%
- **Processing Speed**: 9ms per image
- **Dataset Size**: 120000 training samples

### CNN Tumor Detection
- **Accuracy**: 94.7% on test set
- **Precision**: 92.3%
- **Recall**: 96.1%
- **F1-Score**: 94.2%

---

## 🎯 Part 1: SIMD Convolution Engine

### Features
- **3×3 Kernel Convolution**: Sobel, Gaussian, Sharpen filters
- **SIMD Optimization**: Processes 4 pixels simultaneously
- **Max Pooling**: 2×2 window dimension reduction
- **Edge Detection**: Vertical and horizontal Sobel operators

### Quick Start

```bash
# Compile assembly
nasm -f elf64 convolution.asm -o convolution.o

# Build benchmark
gcc -o my_filter main.c convolution.o -lm -O2

# Run performance test
./my_filter

# Generate visualizations
python3 visualize_benchmark.py
```

### Architecture

The convolution engine uses **SSE SIMD instructions** to process 4 pixels in parallel:

```asm
; Load 4 pixels, zero-extend to int32
pmovzxbd xmm9, [rdi + r13 - 1]

; Convert to float for computation
cvtdq2ps xmm9, xmm9

; Multiply by kernel (4 operations in 1 instruction)
mulps xmm9, xmm0

; Accumulate results
addps xmm13, xmm9
```

**Result**: 5-6x speedup over scalar C code with `-O2` optimization.

---

## 🔍 Part 2: Pattern Recognition System

### Overview
Qmnist number recognition using **normalized cross-correlation**.

### Features
- **Feature Extraction**: Convolution and other functions
- **Visual Analysis**: Output image making borders arounf numbers and detect them
- **Real-time Processing**: <10ms classification time

### Usage

```bash
# Train Dataset and create input image
python3 mnist_prepare.py 150  #150 image per digit

# Compile pattern recognition
gcc -o detect mnist_detection.c convolution.o  -O3 -lm -Iheader -Itemplates/

# Run the detection code
./detect test_qmnist.jpg templates/

```

## 🧠 Part 3: CNN Medical Image Analysis

### Brain Tumor Detection

Deep learning model for automated tumor detection in MRI scans.

### Model Architecture

```python
Conv2D(32) → ReLU → MaxPool
    ↓
Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool
    ↓
Flatten → Dense(256) → Dropout(0.5)
    ↓
Dense(2) → Softmax
```

### Training

```bash
# Assemble maxpool
nasm -f elf64 fast_maxpool.asm -o maxpool.o

# Preprocess MRI dataset
python3 train_cnn.py
python3 export_weights.py

# Evaluate
gcc -o tumor_detection tumor_detection.c cnn.c convolution.o maxpool.o -lm -O3
./tumor_detecion
```

### Results
- Probability: 49.778   Diagnosis: TUMOR DETECTED!!!
- Probability: 3.792    Diagnosis: NO TUMOR

---

[⬆ Back to Top](#assembly-powered-image-processing-engine)

</div>
