# Part Number Recognition API

## Overview
This project implements a FastAPI-based application for part number recognition using a Vision Transformer (ViT) model. It provides endpoints for image-based prediction, explainability with attention maps, and file uploads for dataset management. The system leverages pre-trained ViT models fine-tuned for part number classification, with features for attention-based explainability and model management.

## Features
- **Prediction Endpoint**: Upload an image to predict the top 5 part numbers with confidence scores.
- **Explainability Endpoint**: Generate attention heatmaps and textual explanations for model predictions.
- **File Upload**: Upload images to create datasets for retraining, stored with versioning.
- **Model Management**: List and load fine-tuned models with accuracy metrics.
- **CORS Support**: Configured for frontend integration (e.g., `http://localhost:3000`).

## Requirements
- Python 3.8+
- Dependencies:
  - `fastapi`: Web framework for API endpoints
  - `torch`: PyTorch for model inference
  - `timm`: For ViT model creation
  - `cv2` (OpenCV): Image processing
  - `albumentations`: Image augmentation
  - `numpy`, `pandas`: Data handling
  - `werkzeug`: Secure file handling
  - `pydantic`: Data validation
- A database (configured in `backend.db`)
- Pre-trained ViT models in `./model-retraining/training_outputs`

## Installation
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn torch timm opencv-python numpy pandas albumentations pydantic
   ```
3. **Set Up Directory Structure**
   - Ensure `./model-retraining/training_outputs` contains fine-tuned model files (e.g., `model_fine_tune_X.pth`) and class label CSVs.
   - Create `./uploaded_images/` for storing uploaded datasets.
   - Ensure `./backend/model_list.json` is writable for model accuracy tracking.
4. **Database Setup**
   - Configure the database connection in `backend/db.py` (not provided in this code).

## Usage
1. **Run the API**
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
   ```
   The API will be available at `http://localhost:8000`.
2. **Endpoints**
   - **GET /**: Health check endpoint.
     - Response: `{"message": "Part Number Recognition API is running!"}`
   - **POST /predict/**: Upload an image for part number prediction.
     - Input: Image file
     - Response: JSON with top 5 predictions (part number and confidence)
   - **POST /explain/**: Upload an image for prediction and attention-based explanation.
     - Input: Image file
     - Response: JSON with prediction, base64-encoded heatmap, and textual explanation
   - **POST /upload-files/**: Upload images for a dataset.
     - Input: `dir_name` (form field) and image files
     - Response: Success message
   - **GET /model-list**: Retrieve a list of models and their accuracies.
     - Response: JSON mapping model paths to accuracy values
   - **POST /load-model**: Load a selected model.
     - Input: JSON with `model_path` field
     - Response: Success or error message
3. **CORS Configuration**
   - Currently allows `http://localhost:3000`. Update `allow_origins` in the CORS middleware for production.

## Model and Data
- **Model**: Uses a Vision Transformer (ViT) base model (`vit_base_patch16_224`) fine-tuned for part number classification.
- **Image Preprocessing**: Resizes to 512x512, normalizes with ImageNet mean/std.
- **Explainability**: Combines ViT-CAM attention maps with edge detection for visual and textual explanations.
- **Dataset**: Uploaded images are stored in `./uploaded_images/<dir_name>/dataset_vX`, with versioning.
- **Class Labels**: Loaded from CSV files (e.g., `class_labels_X.csv`) in the model output directory.

## Directory Structure
- `./model-retraining/training_outputs/`: Stores fine-tuned models and classification reports
- `./uploaded_images/`: Stores uploaded datasets
- `./backend/model_list.json`: Tracks model paths and accuracies
- `./backend/`: Contains database and auth modules (e.g., `db.py`, `auth_routes.py`)

## Notes
- **Device**: Automatically uses CUDA if available, else CPU.
- **Logging**: Errors and events are logged to `app.log`.
- **Security**: Uses `secure_filename` for file uploads. Update CORS and auth for production.
- **Model Loading**: Dynamically loads the latest fine-tuned model from the training outputs directory.

## Limitations
- Requires pre-trained models and class label CSVs in the correct format.
- Database and auth modules (`backend.db`, `auth_routes`) are not included and must be implemented.
- CORS is set for local development; secure for production use.

## Troubleshooting
- **Model Not Found**: Ensure model files exist in `./model-retraining/training_outputs/`.
- **Image Decoding Error**: Check if uploaded images are valid (JPEG, PNG, etc.).
- **Database Issues**: Verify `backend.db` is correctly configured.
- **CORS Errors**: Adjust `allow_origins` in the CORS middleware.
