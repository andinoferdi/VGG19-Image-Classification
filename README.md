# VGG19 Image Classification - Flask Web Application

Aplikasi web berbasis Flask untuk training dan prediksi klasifikasi gambar menggunakan model VGG19 dengan transfer learning. Proyek ini menggunakan PyTorch dengan GPU acceleration dan mendukung 6 kelas klasifikasi.

**Setup Langsung Tanpa Virtual Environment** - Install PyTorch dan dependencies secara global ke system Python. GPU support akan otomatis aktif jika CUDA terinstall dengan benar.

## Dataset

Proyek ini menggunakan dataset dengan 6 kelas:
- buildings
- forest
- glacier
- mountain
- sea
- street

## Requirements

- Python 3.12
- NVIDIA GPU dengan CUDA support (disarankan)
- PyTorch dengan CUDA (untuk GPU training)

## Setup Instructions

### 1. Clone Repository

```powershell
git clone https://github.com/andinoferdi/VGG19-Image-Classification.git
```

```powershell
cd VGG19-Image-Classification
```

### 2. Install PyTorch dengan CUDA

**Untuk CUDA 12.1/12.4:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Alternatif untuk CUDA 12.1:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verifikasi instalasi:**
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Output harus menunjukkan versi PyTorch dengan `+cu124` atau `+cu121` (bukan `+cpu`) dan `True` untuk CUDA availability.

**PENTING:** Jika outputnya menunjukan +cpu, uninstall lalu install CUDA version:

```powershell
pip uninstall torch torchvision torchaudio -y
```

**Contoh output yang benar:**
- `2.9.0+cu124 True` ✓
- `2.9.0+cpu False` ✗ (masih CPU version, perlu uninstall dan reinstall)

### 3. Install Dependencies

Install semua dependencies ke system Python secara global:

```powershell
pip install -r requirements.txt
```

### 4. Test GPU (Opsional)

```powershell
python "test gpu.py"
```

Pastikan output menunjukkan:
- `cuda_available=True`
- `gpu_in_use=True`
- GPU name terdeteksi

## Menjalankan Aplikasi

### 1. Masuk ke Folder Flask App

```powershell
cd flask_app
```

### 2. Jalankan Flask Server

```powershell
python app.py
```

### 3. Buka Browser

Buka: http://localhost:5000

## Penggunaan

### 1. Prepare Dataset

- Klik tombol "Prepare Dataset" di web interface
- Sistem akan otomatis split dataset dari `all_data/` ke struktur `dataset/train/val/test`
- Split ratio: 80% train, 20% test, 20% dari train untuk validation

### 2. Training Model

- Konfigurasi training parameters (opsional):
  - Epochs: 25 (default)
  - Learning Rate: 0.001 (default)
  - Batch Size: 32 (default)
- Klik "Start Training"
- Tunggu training selesai (30-60 menit tergantung GPU)

### 3. Prediksi Gambar

- Upload gambar melalui form
- Klik "Predict"
- Lihat hasil prediksi dengan confidence score

### 4. Lihat Hasil

- Klik "Load Metrics" untuk melihat:
  - Training history plots
  - Confusion matrix
  - Detailed metrics report

## Struktur Proyek

```
Image-Classification-on-small-datasets-in-Pytorch/
├── flask_app/
│   ├── app.py                 # Flask application
│   ├── config.py              # Configuration
│   ├── models.py              # VGG19 model
│   ├── data_utils.py          # Dataset utilities
│   ├── train_utils.py         # Training utilities
│   ├── predict_utils.py       # Prediction utilities
│   ├── templates/
│   │   └── index.html         # Web interface
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
├── all_data/                  # Source dataset (6 classes)
├── dataset/                   # Split dataset (created by app)
├── results/                   # Training outputs
├── test gpu.py               # GPU testing utility
├── test_imports.py           # Import verification
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dataset Structure

```
all_data/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/
```

Setelah prepare dataset, struktur menjadi:
```
dataset/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── val/
│   └── [same classes]
└── test/
    └── [same classes]
```

## Model Configuration

- Architecture: VGG19 (pretrained ImageNet)
- Transfer Learning: Feature extraction mode
- Input Size: 224x224
- Number of Classes: 6
- Optimizer: SGD (lr=0.001, momentum=0.9)
- Scheduler: StepLR (step_size=7, gamma=0.1)
- Loss: CrossEntropyLoss

## Data Augmentation

Training augmentation:
- Horizontal flip (p=0.5)
- Random affine rotation (±20 degrees)
- Random affine shear (20 degrees)
- Zoom out (scale 0.8-1.0)
- Color jitter

## Evaluation Metrics

Aplikasi menghitung:
- Accuracy (overall dan per-class)
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- ROC-AUC (one-vs-rest)
- Confusion Matrix
- Training Time
- Testing Time

