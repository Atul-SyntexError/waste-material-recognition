# Waste Material Recognition

A beginner-friendly TensorFlow/Keras project that classifies one waste image into one of six TrashNet categories: **glass, paper, cardboard, plastic, metal, or trash**.

## What's included

- `Waste_Material_Recognition_Optimized.ipynb` — notebook adapted from the uploaded notebook.
- `requirements.txt` — Python packages used by the notebook and prediction script.
- `predict.py` — command-line prediction using a previously trained model.
- `data/` — dataset download/extraction location (created/filled by the notebook).
- `models/` — output location for trained models and class mapping.
- `sample_images/` — optional place for your own test images.

## Important note about testing

The notebook and scripts have been structurally checked, but a full training run was **not** performed in this file-generation step. Model accuracy, graphs, and training time will only exist after you run the notebook; no results are fabricated. MobileNetV2 also needs internet access the first time it downloads ImageNet weights.

## Recommended setup: Windows + VS Code

### 1. Install Python

Install a 64-bit Python version supported by TensorFlow (Python 3.11 is a sensible starting point for Windows). During installation, tick **Add Python to PATH**.

Open PowerShell and verify:

```powershell
py --version
py -m pip --version
```

If `py` is not found, reopen the terminal after installation or reinstall Python with the PATH option enabled.

### 2. Open this folder in VS Code

Extract the project ZIP, then in VS Code choose **File → Open Folder** and select `waste_material_recognition_project`.

Install these VS Code extensions:
- **Python** (Microsoft)
- **Jupyter** (Microsoft)

Git and CUDA are **not required** for the basic workflow. A compatible NVIDIA GPU is optional; TensorFlow on Windows may use CPU unless a supported GPU setup is separately configured. Start with CPU rather than spending time configuring CUDA.

### 3. Create and activate a virtual environment

Run in the VS Code PowerShell terminal, from the project folder:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name waste-material-recognition --display-name "Python (Waste Material Recognition)"
```

If you installed a different supported Python version, replace `-3.11` with the version installed. The `Set-ExecutionPolicy` command applies only to the current PowerShell process.

### 4. Select the notebook kernel

Open `Waste_Material_Recognition_Optimized.ipynb`. Use **Select Kernel** in the top-right and choose **Python (Waste Material Recognition)** or the `.venv` interpreter. Run cells from top to bottom.

### 5. Dataset location

By default, the notebook downloads the resized TrashNet ZIP and extracts it here:

```text
data/
├── dataset-resized.zip
└── raw/
    └── dataset-resized/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

The notebook uses paths relative to the project folder. If you download the dataset manually, place the ZIP at `data/dataset-resized.zip`; the notebook extracts it. If already extracted, ensure the six class folders are directly inside `data/raw/dataset-resized/`.

### 6. Run and review results

Run every cell in order. The notebook:
1. counts actual images and plots class balance and samples;
2. creates stratified train/validation/test splits;
3. compares a baseline CNN, improved CNN, and MobileNetV2 transfer learning;
4. uses early stopping and best-validation checkpoints;
5. selects the model using validation accuracy;
6. produces accuracy/loss curves, a comparison table, test metrics, confusion matrix, and classification report; and
7. saves the selected model as `models/best_waste_classifier.keras` and mapping as `models/class_mapping.json`.

The default optimized settings are 160×160 images, batch size 16, and up to 8 epochs per model. The notebook can still take a while because it trains three models. Early stopping may reduce training time. Smaller images and fewer epochs generally reduce compute time but can also reduce accuracy.

**Memory improvement:** the optimized data pipeline does not cache all decoded images in RAM. This lowers memory pressure, but images are decoded again during epochs and may take a little longer per epoch.

## Predict on a new image without retraining

After training and saving a model, place an image anywhere (or in `sample_images/`) and run:

```powershell
python predict.py --image sample_images\my_waste_photo.jpg
```

Optional explicit paths:

```powershell
python predict.py --image "C:\path\to\photo.jpg" --model models\best_waste_classifier.keras --mapping models\class_mapping.json
```

Expected output format (example format only, not a real prediction):

```text
Predicted class: plastic
Confidence: 87.42%
```

The values above illustrate the output format only. Your prediction will depend on the selected trained model and the image.

## Google Colab option

1. Open Google Colab and upload `Waste_Material_Recognition_Optimized.ipynb`.
2. In **Runtime → Change runtime type**, select **T4 GPU** if available.
3. Upload the project folder or the notebook and let its dataset cell download the dataset. The notebook uses `Path.cwd()`; in Colab, run it from the folder where you uploaded the notebook. If running from `/content`, its `data/` and `models/` folders will be created there.
4. Run all cells. MobileNetV2's ImageNet weights may download on first use.
5. Download `models/best_waste_classifier.keras` and `models/class_mapping.json` from the Colab file browser, then put them in your local project's `models/` folder. You can then use `predict.py` locally without retraining.

Colab sessions are temporary. Download the model and mapping before the session ends. For a more persistent workflow, mount Google Drive and copy the two output files there.

## Common fixes

- **`ModuleNotFoundError`**: activate `.venv`, then run `pip install -r requirements.txt`.
- **Notebook can't find TensorFlow/kernel**: select the `.venv` / `Python (Waste Material Recognition)` kernel, then restart the kernel.
- **Dataset folder error**: confirm the six class folders are directly under `data/raw/dataset-resized/`.
- **Out of memory**: close other applications; set `BATCH_SIZE = 8` and `IMAGE_SIZE = (128, 128)` in the notebook, then restart and run all cells.
- **Slow training on CPU**: use Colab GPU for training; download the `.keras` model and mapping and use `predict.py` locally.
- **Network/download error**: retry when internet is stable, or manually download the dataset archive and place it at `data/dataset-resized.zip`.
- **No saved model yet**: `predict.py` requires a trained `.keras` file and `class_mapping.json`; it cannot predict before those files exist.

## Dataset and limitations

The notebook uses the author-published resized TrashNet archive linked from the project notebook. TrashNet images were collected in relatively controlled backgrounds, so real-world performance may be lower for cluttered bins, poor lighting, occlusion, or mixed materials. Predictions are a project aid, not a reliable safety assessment for hazardous waste.
