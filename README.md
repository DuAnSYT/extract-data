# Medical Ads Classification Model

AI model for extracting structured information from medical-related articles/posts, including:

- Facility / clinic name
- Address
- Phone number
- License number (GPHD)
- Medical certificate number (CCHN)
- Doctor name

The model is based on a combination of a Named Entity Recognition (NER) model and rule-based heuristics to automatically identify and extract the above entities from unstructured text.

---

## 📂 Repository Structure
```text
EXTRACT-DATA/
├── data-scripts/
├── train/
├── infer_pytorch/
├── export/
└── infer_onnx/
```

---

## Folder Descriptions

### 1. Data Preparation `data-scripts/`
Scripts for data preparation, including:
- Data conversion from raw formats
- Automatic or semi-automatic data labeling  
These scripts are used to prepare training data for the model.

### 2. Model Training `train/`
Training source code for the NER model. 

### 3. PyTorch Inference `infer_pytorch/`
PyTorch-based inference code.  
Used for local testing, debugging, and validating model behavior using the original PyTorch model.

Additionally, this folder contains:
`inspect_model.py` – a debugging and inspection tool for the NER model, which allows you to examine the internal processing steps such as:
- Tokenization results
- Token-to-label mapping
- Raw model logits and predicted labels
- Final extracted entities

This is useful for understanding model decisions and troubleshooting incorrect predictions.

### 4. ONNX Export `export/`
Scripts for exporting the trained PyTorch model to **ONNX format**.  
This step is required before deploying the model to production environments.

### 5. ONNX Inference `infer_onnx/`
ONNX-based inference code (deployment version).  
This folder contains inference logic using ONNX Runtime, intended for production or deployment use.

---