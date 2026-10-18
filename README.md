# Computer Vision and CNN-Based Defect Detection in 3D Printing

This repository contains the source code, dataset scripts, evaluation reports, and documentation for the **ICT300 Mini Project** titled **"Computer Vision and CNN-Based Defect Detection in 3D Printing"**. 

The project presents a vision-based framework for real-time, layer-by-layer quality monitoring in extrusion-based Additive Manufacturing (Fused Filament Fabrication / FDM)[cite: 8, 11]. It implements a custom **Multi-Level Spatial Attention Convolutional Neural Network (MLSA-CNN)** alongside five pre-trained transfer learning baseline models (`DenseNet121`, `MobileNetV2`, `ResNet50`, `VGG16`, and `XceptionNet`) to identify critical surface defects (**warping**, **stringing**, and **cracking**) and `No_Defect` states[cite: 5, 8, 10, 11].

---

## 👥 Project Team
* **Alvina Jayaseelan John** (127014005, B.Tech ICT)[cite: 1]
* **Jayabhavana P** (127014022, B.Tech ICT)[cite: 1]
* **Yellamraju Monisha** (127014066, B.Tech ICT)[cite: 1]
* **Project Supervisor:** Dr. Premaladha J, Assistant Professor, School of Computing, SASTRA Deemed to be University[cite: 2]

---

## 📂 Repository Structure

* `custom2.py`: Implementation, training pipeline, and evaluation for the custom MLSA-CNN model.
* `miniproject.py`: Exploratory data analysis, dataset verification, and pre-trained transfer learning baseline models.
* `Computer Vision and CNN-Based Defect Detection in 3D Printing_Report .docx (4) (1).pdf`: Official comprehensive project report[cite: 1].
* `README.md`: Project documentation guide.

---

## 🚀 Methodology & Architecture

1. **Custom MLSA-CNN Architecture:** Incorporates a Multi-Level Spatial Attention block across five hierarchical levels ($C = 16 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$)[cite: 10, 12, 17] with parallel dilated convolutions ($dr = 1, 2, 3$) to capture multi-scale defects (fine filament stringing vs. large structural warping and cracking)[cite: 10, 14, 16].
2. **Synthetic No_Defect Generation:** Utilizes an adaptive 4-pass cropping cascade (corners, edge midpoints, offset grids, and random crops) with local contrast filtering to generate clean background training samples[cite: 24, 34].
3. **Pre-trained Baselines:** Evaluated against five prominent vision architectures (`DenseNet121`, `MobileNetV2`, `ResNet50`, `VGG16`, `XceptionNet`) with frozen feature backbones and custom dense classification heads[cite: 5, 14, 37].

---

## 📊 Summary of Results

| Model / Architecture | Accuracy (%) | Parameters (M) | Model Size (MB) |
| :--- | :---: | :---: | :---: |
| **DenseNet121**[cite: 23] | 95.82%[cite: 23] | 7.04[cite: 23] | 26.86[cite: 23] |
| **MobileNetV2**[cite: 23] | 98.48%[cite: 23] | 3.40[cite: 23] | 14.36[cite: 23] |
| **ResNet50**[cite: 23] | 100.00%[cite: 23] | 25.60[cite: 23] | 98.48[cite: 23] |
| **VGG16**[cite: 23] | 99.62%[cite: 23] | 138.00[cite: 23] | 528.83[cite: 23] |
| **XceptionNet**[cite: 23] | 97.34%[cite: 23] | 22.90[cite: 23] | 88.67[cite: 23] |
| **Custom MLSA-CNN** *(Base)*[cite: 28] | 95.09%[cite: 60] | 7.19[cite: 12] | 27.42[cite: 12] |
| **Custom MLSA-CNN** *(Novelty Tuning)*[cite: 28] | **100.00%**[cite: 28] | **7.19**[cite: 12] | **27.42**[cite: 12] |

---

## 🛠️ Requirements & Execution

Install core dependencies via pip:
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pillow