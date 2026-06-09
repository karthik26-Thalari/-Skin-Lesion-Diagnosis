# 🔬 Skin Lesion Segmentation

Automatic skin lesion segmentation from dermoscopic images using deep learning.

## 🚀 Live Demo
👉 [Try it on Hugging Face](https://huggingface.co/spaces/karthik2687/skin-lesion)

---

## 🧠 Model — MedGradECLIP_Small (U-Net)

Built from scratch in PyTorch — encoder-decoder with skip connections.

## 📥 Download Trained Model
👉 [skin_lesion_best.pth — Hugging Face](https://huggingface.co/spaces/karthik2687/skin-lesion/blob/main/skin_lesion_best.pth)

---

## 📊 Dataset — ISIC 2018

| Split | Images |
|-------|--------|
| Training | 2,594 |
| Validation | 1,000 |

## 📥 Download Dataset
👉 [ISIC 2018 Task 1 — Official Download](https://challenge.isic-archive.com/data/#2018)

Download Task 1:
- Training Images (10.4GB)
- Training Masks (26MB)
- Validation Images (228MB)
- Validation Masks (742KB)

Organize like this after downloading:
\\\
ISIC2018/
├── trainx/    <- training images (.jpg)
├── trainy/    <- training masks  (.png)
├── valx/      <- validation images (.jpg)
└── valy/      <- validation masks  (.png)
\\\

---

## ✅ Results

| Metric | Score |
|--------|-------|
| Accuracy | 87.51% |
| F1 Score | 78.37% |
| Dice Score | 78.37% |
| Precision | 76.06% |
| Recall | 80.83% |
| Specificity | 90.10% |
| IoU Score | 64.44% |

---

## ⚙️ Training

| Setting | Value |
|---------|-------|
| Loss | Dice + BCE |
| Optimizer | Adam |
| LR Scheduler | Cosine Annealing |
| Epochs | 50 |
| Batch Size | 64 |
| Image Size | 128x128 |
| GPU | Kaggle T4 |

---

## 🗂️ Project Structure

\\\
Skin-Lesion-Diagnosis/
├── skin_lesion.ipynb   # Full training notebook
├── app.py              # Hugging Face Gradio app
├── requirements.txt    # Dependencies
└── README.md
\\\

---

## 🚀 Run Locally

\\\ash
git clone https://github.com/karthik26-Thalari/-Skin-Lesion-Diagnosis.git
cd -Skin-Lesion-Diagnosis
pip install -r requirements.txt

# Download model
# Go to https://huggingface.co/spaces/karthik2687/skin-lesion/blob/main/skin_lesion_best.pth
# Place skin_lesion_best.pth in the project folder

# Run demo
python app.py

# Or train from scratch
jupyter notebook skin_lesion.ipynb
\\\

---

## 📄 Citation

\\\
Codella et al., ISIC 2018: https://arxiv.org/abs/1902.03368
Tschandl et al., HAM10000: https://doi.org/10.1038/sdata.2018.161
\\\

---

## 🙌 Acknowledgements
- ISIC Archive — International Skin Imaging Collaboration
- PyTorch — Deep learning framework
- Kaggle — Free GPU for training
- Hugging Face — Model and demo hosting
