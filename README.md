<div align="center">

# 🔬 Skin Lesion Segmentation

**Automatic dermoscopic image segmentation using deep learning**

[![Demo](https://img.shields.io/badge/🤗_Live_Demo-Try_it_now-FF6B6B?style=for-the-badge)](https://huggingface.co/spaces/karthik2687/skin-lesion)
[![Model](https://img.shields.io/badge/🤗_Model-Download_.pth-4A90D9?style=for-the-badge)](https://huggingface.co/spaces/karthik2687/skin-lesion/blob/main/skin_lesion_best.pth)
[![Dataset](https://img.shields.io/badge/📦_Dataset-ISIC_2018-34A853?style=for-the-badge)](https://challenge.isic-archive.com/data/#2018)

</div>

---

## 📌 Overview

This project performs **automatic segmentation of skin lesions** from dermoscopic images using a custom deep-learning model trained on the ISIC 2018 benchmark dataset.

- 🧠 Custom **MedGradECLIP_Small** U-Net architecture (encoder-decoder + skip connections)
- 🏋️ Trained from scratch in **PyTorch** on Kaggle T4 GPU
- 🌐 Deployed as an interactive **Gradio** app on Hugging Face Spaces

---

## 🧠 Model Architecture — MedGradECLIP_Small

```
Input (128×128 RGB)
      │
   Encoder
      │ ← skip connections
   Bottleneck
      │ ← skip connections
   Decoder
      │
Output Mask (128×128)
```

A **U-Net** style architecture built from scratch in PyTorch featuring:
- Symmetric encoder–decoder with skip connections for spatial detail recovery
- Lightweight design optimized for dermoscopic image segmentation
- Trained with a combined **Dice + Binary Cross-Entropy** loss

---

## 📊 Results

| Metric | Score |
|---|---|
| 🎯 Accuracy | **87.51%** |
| 🏆 F1 / Dice Score | **78.37%** |
| 🔍 Precision | 76.06% |
| 📡 Recall | 80.83% |
| 🛡️ Specificity | 90.10% |
| 📐 IoU Score | 64.44% |

---

## 📦 Dataset — ISIC 2018 (Task 1)

| Split | Images |
|---|---|
| Training | 2,594 |
| Validation | 1,000 |

### Download

Go to [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018) and download **Task 1**:

| File | Size |
|---|---|
| Training Images | 10.4 GB |
| Training Masks | 26 MB |
| Validation Images | 228 MB |
| Validation Masks | 742 KB |

### Directory Structure

After downloading, organize the dataset as follows:

```
ISIC2018/
├── trainx/    ← training images  (.jpg)
├── trainy/    ← training masks   (.png)
├── valx/      ← validation images (.jpg)
└── valy/      ← validation masks  (.png)
```

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Loss Function | Dice + BCE |
| Optimizer | Adam |
| LR Scheduler | Cosine Annealing |
| Epochs | 50 |
| Batch Size | 64 |
| Image Size | 128 × 128 |
| GPU | Kaggle T4 |

---

## 🗂️ Project Structure

```
Skin-Lesion-Diagnosis/
├── skin_lesion.ipynb      # Full training notebook
├── app.py                 # Hugging Face Gradio app
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/karthik26-Thalari/-Skin-Lesion-Diagnosis.git
cd -Skin-Lesion-Diagnosis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model

Visit the model page and place the file in the project root:
👉 [skin_lesion_best.pth — Hugging Face](https://huggingface.co/spaces/karthik2687/skin-lesion/blob/main/skin_lesion_best.pth)

```
Skin-Lesion-Diagnosis/
└── skin_lesion_best.pth   ← place here
```

### 4. Run the demo

```bash
python app.py
```

### 5. Or train from scratch

```bash
jupyter notebook skin_lesion.ipynb
```

---

## 📄 Citation

If you use this work or the dataset, please cite:

```bibtex
@article{codella2019skin,
  title={Skin Lesion Analysis Toward Melanoma Detection 2018},
  author={Codella, Noel and others},
  journal={arXiv:1902.03368},
  year={2019}
}

@article{tschandl2018ham10000,
  title={The HAM10000 dataset},
  author={Tschandl, Philipp and others},
  journal={Scientific Data},
  doi={10.1038/sdata.2018.161},
  year={2018}
}
```

---

## 🙌 Acknowledgements

| Resource | Role |
|---|---|
| [ISIC Archive](https://www.isic-archive.com/) | Dermoscopy dataset |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [Kaggle](https://www.kaggle.com/) | Free T4 GPU for training |
| [Hugging Face](https://huggingface.co/) | Model hosting & demo |

---

<div align="center">

Made with ❤️ by [karthik26-Thalari](https://github.com/karthik26-Thalari)

</div>
