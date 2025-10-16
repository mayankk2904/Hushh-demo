from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
import pandas as pd
from timm import create_model
import os
from io import BytesIO
import base64
from backend import db
from typing import List
import logging
from werkzeug.utils import secure_filename
import json
import csv
import glob

from fastapi.middleware.cors import CORSMiddleware
from backend.auth.auth_routes import router as auth_router


OUTPUT_DIR = "./model-retraining/training_outputs"
MODEL_LIST_FILE = "./backend/model_list.json"

os.makedirs("./uploaded_images/", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# -------------------------
# Constants and Configurations
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINE_TUNE_IMG_SIZE = (512, 512)

if os.path.exists(OUTPUT_DIR):
    dirs = os.listdir(OUTPUT_DIR)
    latest_folder = max(dirs, key=lambda x: os.path.getctime(os.path.join(OUTPUT_DIR, x)))
    print(latest_folder)
    model_files = glob.glob(os.path.join(OUTPUT_DIR, latest_folder, "*model_fine_tune*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model file found in {os.path.join(OUTPUT_DIR, latest_folder)}")
    MODEL_PATH = model_files[0]
    parent_dir, model_file = os.path.split(MODEL_PATH)
    start, end = model_file.rfind("_") + 1, model_file.find(".")
    NUM_CLASSES = int(model_file[start:end])
    CLASS_LABEL_FILE = os.path.join(parent_dir, f"class_labels_{NUM_CLASSES}.csv")

# Minimal augmentation for inference (resize only)
augmentations_inference = A.Compose([
    A.Resize(FINE_TUNE_IMG_SIZE[0], FINE_TUNE_IMG_SIZE[1])
], p=1.0)

# -------------------------
# Preprocessing Function
# -------------------------
def preprocess_image(image, img_size, augmentations):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = augmentations(image=image)['image']
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    augmented = (augmented / 255.0 - mean) / std
    return augmented.astype(np.float32)

# -------------------------
# Model Loader (cached for efficiency)
# -------------------------
def load_model():
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    # Adjust the image size for patch embedding
    model.patch_embed.img_size = FINE_TUNE_IMG_SIZE
    patch_size = model.patch_embed.patch_size
    grid_height = FINE_TUNE_IMG_SIZE[0] // patch_size[0]
    grid_width  = FINE_TUNE_IMG_SIZE[1] // patch_size[1]
    num_patches = grid_height * grid_width
    embed_dim = model.pos_embed.shape[-1]
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'pos_embed' in state_dict:
        checkpoint_pos_embed = state_dict['pos_embed']
        if checkpoint_pos_embed.shape != model.pos_embed.shape:
            model.pos_embed = torch.nn.Parameter(checkpoint_pos_embed)
            del state_dict['pos_embed']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# Load model once when FastAPI starts
model = load_model()
# Get patch size from the model (used in the explainability function)
patch_size = model.patch_embed.patch_size  # e.g. (16, 16)

# -------------------------
# Label Mapping
# -------------------------
def get_label_mapping(label_file_path):
    global label_mapping
    df = pd.read_csv(label_file_path)
    label_mapping = {i: label for i, label in enumerate(df["Class_Label"])}

get_label_mapping(CLASS_LABEL_FILE)

# -------------------------
# ViT-CAM and Attention Map Functions
# -------------------------
def get_vit_cam(model, input_tensor):
    """
    Computes a class activation map for a ViT model by using the attention weights 
    of the class token from the last transformer block.
    """
    hooks = []
    attention_last = None

    def hook_fn(module, input, output):
        nonlocal attention_last
        attention_last = output.detach()

    hook = model.blocks[-1].attn.register_forward_hook(hook_fn)
    hooks.append(hook)
    with torch.no_grad():
        _ = model(input_tensor)
    for hook in hooks:
        hook.remove()

    if attention_last.ndim == 4:
        cls_attn = attention_last[0, :, 0, 1:]
        cls_attn_mean = cls_attn.mean(dim=0)
    elif attention_last.ndim == 3:
        cls_attn = attention_last[0, 0, 1:]
        cls_attn_mean = cls_attn
    else:
        raise ValueError(f"Unexpected attention tensor shape: {attention_last.shape}")

    num_tokens = cls_attn_mean.shape[0]
    grid_size = int(np.sqrt(num_tokens))
    cam = cls_attn_mean[:grid_size * grid_size].reshape(grid_size, grid_size).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, FINE_TUNE_IMG_SIZE)
    return cam

def analyze_heatmap_regions(heatmap: np.ndarray, threshold: float = None, max_regions: int = 3):
    """
    Analyze the heatmap to find connected regions.
    If threshold is not provided, compute a dynamic threshold based on the heatmap statistics.
    """
    heatmap = np.clip(heatmap, 0, 1)
    if threshold is None:
        # Dynamic threshold: mean + 0.5 * standard deviation
        threshold = np.mean(heatmap) + 0.5 * np.std(heatmap)
        threshold = np.clip(threshold, 0.3, 0.7)
    
    mask = (heatmap >= threshold).astype(np.uint8)
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        cx, cy = centroids[label_idx]
        regions.append({
            "label_idx": label_idx,
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "cx": cx, "cy": cy
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions[:max_regions]

def regions_to_text(regions, heatmap_shape):
    """
    Convert each region's bounding box and centroid into descriptive phrases.
    """
    h, w = heatmap_shape
    descriptions = []
    total_area = sum(r["area"] for r in regions)
    total_area = total_area if total_area > 0 else 1e-6

    for r in regions:
        fraction = r["area"] / total_area
        if fraction > 0.5:
            size_str = "a very large"
        elif fraction > 0.3:
            size_str = "a large"
        elif fraction > 0.15:
            size_str = "a moderately sized"
        else:
            size_str = "a small"

        cx, cy = r["cx"], r["cy"]
        vertical_pos = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "center"
        horizontal_pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "middle"
        descriptions.append(f"{size_str} region near the {vertical_pos}-{horizontal_pos}")
    return descriptions

def get_attention_rollout(model, input_tensor):
    """
    Computes attention rollout map using the model's attention weights.
    """
    attentions = []
    hooks = []
    
    def hook_fn(module, input, output):
        attentions.append(output.detach())
    
    for block in model.blocks:
        hook = block.attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    num_tokens = attentions[0].shape[-1]
    rollout = torch.eye(num_tokens, device=device)
    for att in attentions:
        att_heads = att.mean(dim=1)[0] + torch.eye(num_tokens, device=device)
        att_heads = att_heads / att_heads.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(att_heads, rollout)
    rollout = rollout[0, 1:].cpu().numpy()
    num_patches = rollout.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        factors = []
        for i in range(1, int(np.sqrt(num_patches)) + 1):
            if num_patches % i == 0:
                factors.append((i, num_patches // i))
        grid_h, grid_w = factors[-1] if factors else (num_patches, 1)
    else:
        grid_h, grid_w = grid_size, grid_size
    return rollout.reshape(grid_h, grid_w)

def generate_attention_map_vitcam(model, input_tensor, original_image):
    """
    Generates an attention map using the ViT-CAM approach along with edge detection.
    """
    cam = get_vit_cam(model, input_tensor)
    heatmap = cv2.GaussianBlur(cam, (15, 15), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 300, 600)
    edges = cv2.dilate(edges, None, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heatmap_mask = np.zeros(FINE_TUNE_IMG_SIZE, dtype=np.float32)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(heatmap_mask, [contour], -1, 1.0, thickness=cv2.FILLED)
    
    heatmap = heatmap * heatmap_mask
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap, edges

def generate_text_explanation(heatmap: np.ndarray, label: str, confidence: float, threshold: float = None, max_regions: int = 3) -> str:
    """
    Generate an advanced textual explanation by identifying connected regions
    in the attention heatmap. It describes each region's relative size and location.
    """
    regions = analyze_heatmap_regions(heatmap, threshold=threshold, max_regions=max_regions)
    if not regions:
        explanation = (
            f"The model predicted '{label}' with {confidence*100:.1f}% confidence, "
            "but no significant attention regions were detected."
        )
    else:
        region_descriptions = regions_to_text(regions, heatmap.shape)
        regions_text = "; ".join(region_descriptions)
        overall_intensity = np.mean(heatmap)
        explanation = (
            f"The model predicted '{label}' with {confidence*100:.1f}% confidence. "
            f"The attention map highlights {regions_text}. "
            f"Overall, the average attention intensity is {overall_intensity:.2f}, indicating "
            "where the model focuses most on distinctive features."
        )
    
    if confidence < 0.5:
        explanation += " Note: The prediction confidence is low, so the result might be less reliable."
    
    return explanation

# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
def home():
    return {"message": "Part Number Recognition API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    if image is None:
        raise HTTPException(status_code=400, detail="Error decoding image!")
    
    processed_img = preprocess_image(image, FINE_TUNE_IMG_SIZE, augmentations_inference)
    input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_probs = probabilities[top5_indices]
    top5_labels = [label_mapping[idx] for idx in top5_indices]
    
    if len(top5_labels) >= 5:
        return JSONResponse(content={
            "top_5_predictions": [
                {"part_number": top5_labels[i], "confidence": float(top5_probs[i])}
                for i in range(5)
            ]
        })
    else:
         return JSONResponse(content={
            "top_5_predictions": [
                {"part_number": top5_labels[i], "confidence": float(top5_probs[i])}
                for i in range(len(top5_labels))
            ]
        })

@app.post("/explain/")
async def explain_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise Exception()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    processed_img = preprocess_image(image, FINE_TUNE_IMG_SIZE, augmentations_inference)
    input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top_idx = np.argmax(probs)
    top_label = label_mapping[top_idx]
    top_conf = float(probs[top_idx])
    
    resized_image = cv2.resize(image, FINE_TUNE_IMG_SIZE)
    heatmap, edges = generate_attention_map_vitcam(model, input_tensor, resized_image)
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(resized_image, 0.4, heatmap_color, 0.8, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay[edges > 0] = [255, 255, 255]
    
    textual_explanation = generate_text_explanation(heatmap, top_label, top_conf, threshold=0.4, max_regions=3)
    
    _, buffer = cv2.imencode('.png', overlay)
    return JSONResponse(content={
        "prediction": {"part_number": top_label, "confidence": top_conf},
        "explanation": base64.b64encode(buffer).decode('utf-8'),
        "textual_explanation": textual_explanation
    })

logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)


async def save_files(dir_path, files):
    for file in files:
        filename = secure_filename(file.filename)
        file_path = f"{dir_path}/{filename}"
        contents = await file.read()
        try:
            with open(file_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            logging.error(f"Failed to save file {filename} to {dir_path}\n{e}")
        finally:
            file.file.close()


def get_dir_path(dir_name):

    dir_path = f"./uploaded_images/{dir_name}"
    try:
        if os.path.exists(dir_path):
            dirs = os.listdir(dir_path) 
            latest_folder = max(dirs, key=lambda x: os.path.getctime(f"{dir_path}/{x}"))
            ds_version = int(latest_folder[-1]) + 1
            new_dir_name = f"/dataset_v{ds_version}"
            dir_path += new_dir_name
        else:
            dir_path += "/dataset_v1"
        os.makedirs(dir_path)
    except Exception as e:
        logging.critical(e)

    return dir_path


def get_image_file_paths(dir_path):
    image_files = os.listdir(dir_path)
    image_file_paths = []
    for image_file in image_files:
        image_file_paths.append(f"{dir_path}/{image_file}")

    return image_file_paths


def insert_image_data(cursor, conn, part_number, dir_path, dataset_version):
    image_file_paths = get_image_file_paths(dir_path)
    for image_file_path in image_file_paths:
        db.insert_images_data(
            cursor, conn, part_number, dataset_version, image_file_path
        )


@app.post("/upload-files/")
async def upload_files(dir_name: str = Form(...), files: List[UploadFile] = File(...)):

    dir_path = get_dir_path(dir_name)
    await save_files(dir_path, files)

    dataset_version = dir_path[-1]

    cursor, conn = db.connect_db()
    db.create_tables_if_not_exists(cursor, conn)
    db.insert_component_data(cursor, conn, dir_name)
    db.insert_datasets_data(cursor, conn, dir_name, dataset_version)

    insert_image_data(cursor, conn, dir_name, dir_path, dataset_version)

    if cursor:
        cursor.close()
        conn.close()

    return "Files uploaded successfully"


def get_model_accuracy():
    model_acc = {}
    if os.path.exists(OUTPUT_DIR):
        folders = os.listdir(OUTPUT_DIR)
        for folder in folders:
            folder_path = os.path.join(OUTPUT_DIR, folder)
            file_pattern = os.path.join(folder_path, "val_classification_report_*")
            classification_files = glob.glob(file_pattern)
            
            if not classification_files:  # Check if any files were found
                logging.warning(f"No classification files found in {folder_path}")
                continue  # Skip to next folder
            
            try:
                classification_file = os.path.normpath(classification_files[0])
                
                with open(classification_file, newline="") as csv_file:
                    file_reader = csv.reader(csv_file)
                    rows = [row for row in file_reader]

                    acc_idx = len(rows) - 3
                    acc_str = rows[acc_idx][1]
                    acc = round(float(acc_str) * 100, 2)

                    parent_dir = os.path.dirname(classification_file)
                    file_name = os.path.basename(classification_file)
                    no_of_parts = file_name.split('_')[-1].split('.')[0]
                    model_file = f"model_fine_tune_{no_of_parts}.pth"
                    model_file_path = os.path.join(parent_dir, model_file)
                    model_file_path = os.path.normpath(model_file_path)
                    model_acc[model_file_path] = acc
                    
            except (IndexError, KeyError, ValueError) as e:
                logging.error(f"Error processing {classification_file}: {str(e)}")
                continue

        return model_acc if model_acc else None
    return None

def update_model_list():
    model_acc = get_model_accuracy()
    model_path_key = os.path.normpath(MODEL_PATH)
    model_list = {model_path_key: 0.0}

    if model_acc is not None:
        if os.path.exists(MODEL_LIST_FILE):
            with open(MODEL_LIST_FILE, "r+") as json_file:
                old_model_list = json.load(json_file)
                for path, acc in old_model_list.items():
                    if os.path.exists(path):
                        model_list[path] = acc
                model_list.update(model_acc)
                json_file.seek(0)
                json_file.truncate(0)
                json.dump(model_list, json_file, indent=4)
        else:
            model_list |= model_acc
            with open(MODEL_LIST_FILE, "w") as json_file:
                json.dump(model_list, json_file, indent=4)
    else:
        logging.error(f"No models found in {OUTPUT_DIR}")


@app.get("/model-list")
def get_model_list():
    update_model_list()
    model_list = ""
    if os.path.exists(MODEL_LIST_FILE):
        with open(MODEL_LIST_FILE, "r") as json_file:
            model_list = json.load(json_file)
    else:
        model_list = f"{MODEL_LIST_FILE} file was not found"
        logging.error(model_list)
    return model_list

from pydantic import BaseModel

class ModelRequest(BaseModel):
    model_path: str

@app.post("/load-model")
def load_selected_model(req: ModelRequest):
    try:
        global NUM_CLASSES, MODEL_PATH, model, patch_size
        MODEL_PATH = req.model_path
        parent_dir, model_file = MODEL_PATH.rsplit("\\", maxsplit=1)
        start, end = model_file.rfind("_") + 1, model_file.rfind(".")
        NUM_CLASSES = int(model_file[start:end])

        model = load_model()
        patch_size = model.patch_embed.patch_size

        CLASS_LABEL_FILE = os.path.normpath(
            os.path.join(parent_dir, f"class_labels_{NUM_CLASSES}.csv")
        )
        get_label_mapping(CLASS_LABEL_FILE)

        return {
            "success": True,
            "message": "Model changed successfully!"
        }

    except Exception as e:
        logging.error(str(e))
        return {
            "success": False,
            "message": str(e)
        }
app.include_router(auth_router)

# -------------------------
# Running FastAPI
# -------------------------
# To run: `uvicorn fastapi_app:app --host 0.0.0.0 --port 8000`