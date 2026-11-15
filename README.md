# DermaCheck — AI-Powered Multi-Class Skin Lesion Classifier

DermaCheck is a full-stack deep learning dermatology screening web application capable of classifying **115 distinct skin conditions**. Powered by a fine-tuned **EfficientNet-B2** neural network trained across four major clinical and dermoscopic datasets, the application provides real-time image inference alongside **Top-3 differential predictions**, **per-class F1 scores**, and **2×2 One-vs-Rest confusion matrices**.

> **Medical Disclaimer:** DermaCheck is an academic and research tool designed for educational and screening assistance purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for clinical evaluation.

---

## Key Features

* **115-Class Skin Condition Classification:** Combines clinical and dermoscopic imagery across diverse skin tones and lesion types.
* **Real-Time Model Evaluation Analytics:** Every scan dynamically surfaces validation performance metrics for the predicted condition:
  * **Class F1 Score, Precision, and Recall**
  * **Overall Model Weighted & Macro F1 Scores**
  * **2×2 One-vs-Rest Confusion Matrix** (`TP`, `FP`, `FN`, `TN`) and class support counts
  * **Top-3 Differential Predictions** with softmax confidence percentages
  * **Top Confused Classes** to highlight visual overlap between similar conditions
* **Resilient Training Pipeline:** Supports CUDA GPU acceleration, inverse class-frequency loss weighting for class imbalance, StepLR learning rate decay, best-loss model checkpointing (`skin_model_best.pth`), and `KeyboardInterrupt` (`Ctrl + C`) pause/resume checkpointing.

---

## Datasets & Label Standardization

The training pipeline unifies and deduplicates records from four public dermatology datasets into **115 standardized classes**:

| Dataset | Focus Area | Images / Metadata Used |
| :--- | :--- | :--- |
| **Fitzpatrick17k** | Diverse skin tones (Fitzpatrick I–VI) across 114 clinical skin conditions | `fitzpatrick17k.csv` |
| **HAM10000** | Pigmented skin lesions (7 diagnostic categories) | `HAM10000_metadata.csv` |
| **ISIC 2019** | Dermoscopic skin cancer & lesion imagery *(deduplicated against HAM10000)* | `ISIC_2019_Training_GroundTruth.csv` |
| **PAD-UFES-20** | Smartphone-captured clinical skin lesion images | `padufes.csv` |

---

## Tech Stack

* **Deep Learning & Evaluation:** PyTorch, `timm` (`efficientnet_b2`), Torchvision, `scikit-learn`, Pandas, NumPy, Pillow
* **Backend API:** FastAPI, Uvicorn, Python
* **Frontend Web App:** React, Vite, Tailwind CSS, Lucide Icons

---

## Project Structure

```text
DermaCheck/
├── Backend/
│   ├── app.py                   # FastAPI inference & metrics server (/predict, /health)
│   ├── train.py                 # Multi-dataset training script with GPU & pause/resume support
│   ├── evaluate.py              # Validation split evaluation script generating F1 & confusion matrices
│   ├── label_mapping.json       # Class ID to disease label dictionary (115 classes)
│   ├── evaluation_metrics.json  # Pre-computed validation F1, precision, recall, and confusion matrices
│   └── skin_model_best.pth      # Best trained EfficientNet-B2 model weights
├── Frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI components (ImageUpload, ResultsCard, Disclaimer)
│   │   ├── data/                # Local condition symptoms, causes, and care tips database
│   │   └── pages/               # Application views including Scan.jsx
│   ├── package.json
│   └── vite.config.js
├── Dataset/                     # Local datasets (excluded from Git via .gitignore)
│   ├── fitzpatrick17k/
│   ├── ham10000/
│   ├── isic2019/
│   └── pad_ufes_20/
├── .gitignore
└── README.md
```

---

## Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AsthaSingh-19/DermaCheck.git
cd DermaCheck
```

### 2. Backend Setup (FastAPI + PyTorch)
1. Navigate to the `Backend` directory and install dependencies:
   ```bash
   cd Backend
   pip install torch torchvision timm fastapi uvicorn python-multipart pillow pandas numpy scikit-learn
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   The API will run at `http://127.0.0.1:8000`.

### 3. Frontend Setup (React + Vite)
1. Open a new terminal, navigate to `Frontend`, and install packages:
   ```bash
   cd Frontend
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
   Open the local URL shown in your terminal (typically `http://localhost:5173`) and navigate to `/scan`.

---

## Model Training & Evaluation (Optional)

If you wish to retrain or re-evaluate the model locally:

1. Place the four datasets inside `Dataset/` following the folder layout in the Project Structure above.
2. Run training (supports CUDA GPU and `Ctrl + C` checkpointing):
   ```bash
   cd Backend
   python train.py
   ```
3. Run standalone validation evaluation to regenerate `evaluation_metrics.json`:
   ```bash
   python evaluate.py
   ```
