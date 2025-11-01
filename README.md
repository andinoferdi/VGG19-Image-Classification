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

**PENTING:** Jika PyTorch CPU version sudah terinstall, uninstall dulu sebelum install CUDA version:

```powershell
pip uninstall torch torchvision torchaudio -y
```

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

## Troubleshooting

### GPU Not Detected / gpu_in_use=False

1. **Cek versi PyTorch yang terinstall:**
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   
   Jika output menunjukkan `+cpu` dan `False`, berarti PyTorch CPU version masih terinstall.

2. **Solusi: Uninstall PyTorch CPU dan install CUDA version:**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   
   Setelah install, verifikasi lagi:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   
   Harus menunjukkan `+cu124` atau `+cu121` dan `True`.

3. Verifikasi GPU hardware terdeteksi:
   ```powershell
   nvidia-smi
   ```

4. Test dengan script:
   ```powershell
   python "test gpu.py"
   ```
   
   Pastikan output menunjukkan:
   - `cuda_available=True`
   - `gpu_in_use=True`
   - GPU name terdeteksi

### Import Errors

1. Install semua dependencies secara global:
   ```powershell
   pip install -r requirements.txt
   ```

2. Test imports:
   ```powershell
   python test_imports.py
   ```

3. Pastikan menggunakan Python 3.12 yang sudah terinstall PyTorch CUDA

### IDE Masih Menggunakan Venv (Python Path Error)

Jika IDE/editor masih mencoba menggunakan `.venv312` yang sudah tidak ada:

**VS Code / Cursor:**
1. Buka Command Palette (`Ctrl+Shift+P`)
2. Ketik "Python: Select Interpreter"
3. Pilih: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe`
   (atau path Python system Anda)
4. File `.vscode/settings.json` sudah dikonfigurasi untuk menggunakan system Python

**Atau edit manual di `.vscode/settings.json`:**
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\YourUsername\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
    "python.terminal.activateEnvironment": false
}
```

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Pilih "System Interpreter" atau browse ke Python system path

**Verifikasi:**
Jalankan di terminal IDE:
```powershell
python -c "import sys; print(sys.executable)"
```

Pastikan output menunjuk ke system Python, bukan venv.

### CUDA Out of Memory

- Kurangi batch size di training form (coba 16 atau 8)
- Tutup aplikasi lain yang menggunakan GPU

### Port 5000 Already in Use

Edit `flask_app/app.py` line terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Dataset Not Found

- Pastikan folder `all_data/` ada dengan 6 subfolder kelas
- Struktur harus sesuai dengan format di atas

## Monitoring GPU Usage

Untuk monitoring real-time GPU:
```powershell
nvidia-smi -l 1
```

Refresh setiap 1 detik.

## Results Location

Setelah training, hasil tersimpan di `results/`:
- `vgg19_best.pth` - Model weights terbaik
- `training_history.png` - Loss dan accuracy plots
- `confusion_matrix.png` - Confusion matrix heatmap
- `metrics.txt` - Detailed evaluation metrics

## Notes

- Setup ini menginstall PyTorch dan dependencies secara global ke system Python
- Tidak memerlukan virtual environment - GPU akan otomatis terdeteksi jika PyTorch CUDA terinstall dengan benar
- Dataset preparation hanya perlu dilakukan sekali
- Model training bisa memakan waktu 30-60 menit (tergantung GPU)
- Pastikan GPU tersedia untuk training yang lebih cepat
- Task Manager Windows mungkin tidak akurat untuk CUDA usage, gunakan `nvidia-smi`

## License

[Your License Here]

## Author

[Your Name Here]

