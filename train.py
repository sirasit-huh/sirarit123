import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DATA_DIR = Path("data/fish_dataset")
MODEL_DIR = Path("model")
REPORTS_DIR = Path("reports")
BATCH_SIZE = 64
RANDOM_STATE = 42

def setup_feature_extractor():
    print("Loading pretrained MobileNetV2 for robust feature extraction...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet = models.mobilenet_v2(weights=weights)
    mobilenet.eval()
    
    # Replace classification head with Identity to extract raw 1280-d features
    mobilenet.classifier = nn.Identity()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return mobilenet, transform

def extract_features(dataset, mobilenet, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features_list = []
    labels_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features using device: {device}")
    mobilenet = mobilenet.to(device)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            feats = mobilenet(images)  # Shape: (batch_size, 1280)
            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())
            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(loader):
                print(f"Processed batch {batch_idx + 1}/{len(loader)} ({min((batch_idx+1)*batch_size, len(dataset))}/{len(dataset)} images)")
                
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y

def plot_and_save_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    mobilenet, transform = setup_feature_extractor()
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Please run download_and_sample.py first.")
        
    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    class_names = dataset.classes
    print(f"Found {len(dataset)} images belonging to {len(class_names)} classes: {class_names}")
    
    start_time = time.time()
    X, y = extract_features(dataset, mobilenet, batch_size=BATCH_SIZE)
    print(f"Extracted feature matrix X shape: {X.shape}, y shape: {y.shape} in {time.time() - start_time:.2f}s")
    
    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} samples, Test size: {X_test.shape[0]} samples")
    
    # Feature Scaling (StandardScaler) fitted on Train to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define Regularized Scikit-learn Models (Combat Overfitting)
    models_dict = {
        "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_split=4, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    fitted_models = {}
    
    print("\n" + "="*60)
    print("STARTING ROBUST MULTI-BACKGROUND MODEL TRAINING")
    print("="*60)
    
    for name, clf in models_dict.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Test Accuracy : {acc * 100:.2f}%")
        print(f"Precision     : {prec * 100:.2f}%")
        print(f"Recall        : {rec * 100:.2f}%")
        print(f"F1-Score      : {f1 * 100:.2f}%")
        print(f"Training time : {train_time:.2f}s")
        
        cm = confusion_matrix(y_test, y_pred)
        cm_filename = REPORTS_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plot_and_save_confusion_matrix(cm, class_names, f"Confusion Matrix: {name}", cm_filename)
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "train_time_sec": float(train_time),
            "confusion_matrix_image": str(cm_filename)
        }
        fitted_models[name] = clf
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    print("\n" + "="*60)
    print(f"BEST PERFORMING MODEL: {best_model_name} (F1: {best_f1 * 100:.2f}%)")
    print("="*60)
    
    # Save Model Bundle with all models and formatted accuracies
    bundle = {
        "models": fitted_models,
        "accuracies": {k: f"{v['accuracy']*100:.2f}%" for k, v in results.items()},
        "scaler": scaler,
        "class_names": class_names,
        "best_model": best_model_name,
        "metrics": results
    }
    model_save_path = MODEL_DIR / "fish_classifier.joblib"
    joblib.dump(bundle, model_save_path)
    print(f"Saved complete model bundle to: {model_save_path}")
    
    # Save full metrics JSON
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "models": results,
            "best_model": best_model_name,
            "classes": class_names,
            "dataset_counts": {
                "total": len(dataset),
                "train": len(X_train),
                "test": len(X_test)
            }
        }, f, indent=4, ensure_ascii=False)
    print(f"Saved evaluation metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
