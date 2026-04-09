# Deep Learning Systems — Project 4

## Overview
This project implements a **deep learning image classification workflow** using **PyTorch** and the **Fashion-MNIST** dataset. The notebook builds and compares two convolutional neural network (CNN) architectures:

- a **baseline CNN** with two convolutional blocks
- a **deeper CNN** with one additional convolutional layer

The goal is not just final accuracy, but correct implementation, controlled experimentation, and clear interpretation of results.

---

## Project Files

- `deep_learning.ipynb` — executed notebook containing data loading, preprocessing, model training, evaluation, plots, and summary
- `Deep_Learning_Systems_Analysis_Report.md` — written analysis report with citations and interpretation
- `requirements.txt` — reproducibility file generated from `pip freeze`
- `deep_learning.html` — HTML export of the executed notebook
- `data/FashionMNIST/` — downloaded dataset files

---

## Dataset
This project uses **Fashion-MNIST**, a benchmark dataset of `28×28` grayscale clothing images across 10 classes.

The dataset is downloaded automatically in the notebook with `torchvision.datasets.FashionMNIST(..., download=True)` and stored under `data/`.

---

## Environment Setup
Create and activate the local virtual environment, then install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run the Notebook

Open the notebook in Jupyter or VS Code:

```bash
source .venv/bin/activate
jupyter notebook deep_learning.ipynb
```

To execute the notebook from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute deep_learning.ipynb --inplace
```

To export the executed notebook to HTML:

```bash
.venv/bin/jupyter nbconvert --to html deep_learning.ipynb --output deep_learning.html
```

---

## Experimental Comparison
The experiment changes **one major factor only**: model depth.

| Model | Change |
|---|---|
| `Baseline CNN` | Two convolutional layers |
| `Deeper CNN` | Adds a third convolutional layer |

All other settings were kept fixed:

- optimizer: `Adam`
- loss: `CrossEntropyLoss`
- learning rate: `0.001`
- batch size: `128`
- epochs: `4`

---

## Results Summary

| Model | Best Validation Accuracy | Test Accuracy | Macro F1 | Test Loss |
|---|---:|---:|---:|---:|
| `Deeper CNN` | `0.8647` | `0.8626` | `0.8606` | `0.3887` |
| `Baseline CNN` | `0.8643` | `0.8589` | `0.8571` | `0.4004` |

### Key Takeaway
The deeper CNN performed **slightly better** than the baseline across all headline metrics. The most difficult classes remained visually similar upper-body items such as **Shirt**, **Pullover**, and **T-shirt/top**.

---

## Submission Checklist
This repository includes the required deliverables:

- ✅ executed notebook
- ✅ analysis report with citations
- ✅ `requirements.txt`
- ✅ dataset access through notebook download instructions
- ✅ HTML export of the notebook

---

## Author Notes
This project was completed for the **Udacity Deep Learning Systems** module and emphasizes reproducibility, controlled experimentation, and clear communication of findings.