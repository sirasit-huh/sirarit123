import os
import io
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fish_classifier.joblib"
SAMPLES_DIR = BASE_DIR / "sample_test_images"

# Load Pretrained Feature Extractor (MobileNetV2)
print("Loading MobileNetV2 Feature Extractor...")
weights = models.MobileNet_V2_Weights.DEFAULT
mobilenet = models.mobilenet_v2(weights=weights)
mobilenet.eval()
mobilenet.classifier = nn.Identity()

# Pre-initialize AI Background Isolation Session (u2net) for lightning fast response
print("Pre-loading AI Background Isolation Session...")
try:
    from rembg import new_session, remove
    rembg_session = new_session("u2net")
    REMBG_AVAILABLE = True
    print("AI Background Isolation Session ready!")
except Exception as e:
    print(f"Warning: rembg session note: {e}")
    rembg_session = None
    REMBG_AVAILABLE = False

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Models Bundle
print("Loading Model Bundle...")
bundle = joblib.load(MODEL_PATH)
models_dict = bundle["models"]
scaler = bundle["scaler"]
class_names = bundle["class_names"]
accuracies = bundle.get("accuracies", {
    "Support Vector Machine (SVM)": "99.81%",
    "Logistic Regression (L2)": "99.91%",
    "Random Forest": "99.26%"
})

# Fishery & Marine Knowledge Base (ออกแบบเฉพาะสำหรับงานประมง ตลาดปลา และการตรวจรับสินค้า)
FISHERY_INFO = {
    "Black Sea Sprat": {
        "thai_name": "ปลาสเปรตทะเลดำ",
        "scientific_name": "Clupeonella cultriventris",
        "family": "Clupeidae (วงศ์ปลาเฮอริ่ง)",
        "market_grade": "ปลาแปรรูปอุตสาหกรรม / ปลาเหยื่อ",
        "habitat": "ทะเลดำ ทะเลแคสเปียน น้ำกร่อยและปากแม่น้ำ",
        "field_marks": "ลำตัวเล็กเพรียว 8-10 ซม. สีเงินวาว สันท้องเป็นเกล็ดหนามคมชัดเจน",
        "commercial": "นิยมแปรรูปเป็นปลารมควัน ปลากระป๋องในน้ำมัน และอาหารเสริมโปรตีนสูง อุดมด้วยโอเมก้า 3",
        "culinary": "ชุบแป้งทอดกรอบทั้งตัว ทานคู่กับมะนาว หรือแปรรูปรมควัน",
        "badge_color": "badge-info"
    },
    "Gilt-Head Bream": {
        "thai_name": "ปลากะพงทรายสีทอง (โดราด / Dorada)",
        "scientific_name": "Sparus aurata",
        "family": "Sparidae (วงศ์ปลากะพงทราย)",
        "market_grade": "ปลาเศรษฐกิจเกรดพรีเมียม (High Value)",
        "habitat": "ทะเลเมดิเตอร์เรเนียน และชายฝั่งแอตแลนติก",
        "field_marks": "ลำตัวทรงรีแบนข้าง แถบสีทองพาดระหว่างดวงตา มีแต้มดำเด่นชัดที่ขอบเหงือก",
        "commercial": "ปลาเพาะเลี้ยงมูลค่าสูงของตลาดยุโรป ความต้องการตลาดสม่ำเสมอ ราคาขายดี",
        "culinary": "เนื้อขาวละเอียด แน่นฉ่ำ ก้างไม่เยอะ นิยมย่างเกลือ อบสมุนไพร หรือนึ่ง",
        "badge_color": "badge-warning"
    },
    "Hourse Mackerel": {
        "thai_name": "ปลาทูแขก หรือ ปลาอาจิ (Aji)",
        "scientific_name": "Trachurus trachurus",
        "family": "Carangidae (วงศ์ปลาหางแข็ง)",
        "market_grade": "ปลาสดพาณิชย์ / ตลาดปลาซาชิมิ",
        "habitat": "มหาสมุทรแอตแลนติก ทะเลเหนือ และเมดิเตอร์เรเนียน",
        "field_marks": "ลำตัวเพรียวยาว มีเกล็ดหนามแข็ง (Scutes) เป็นแนวสันคมยาวตามเส้นข้างลำตัว",
        "commercial": "เป็นปลาหลักของเรือประมงอวนล้อม ตลาดญี่ปุ่นต้องการสูงเพื่อทำซาชิมิเกรดสด",
        "culinary": "ซาชิมิปลาอาจิ ทอดกรอบอาจิแฟราย (Aji Fry) หรือย่างซีอิ๊ว เนื้อหอมมันเข้มข้น",
        "badge_color": "badge-success"
    },
    "Red Mullet": {
        "thai_name": "ปลาบาร์บูนแดง (Red Mullet)",
        "scientific_name": "Mullus barbatus",
        "family": "Mullidae (วงศ์ปลาแพะ)",
        "market_grade": "ปลาทะเลหายาก / ภัตตาคารซีฟู้ด",
        "habitat": "หน้าดินทรายและโคลน ทะเลเมดิเตอร์เรเนียนและทะเลดำ",
        "field_marks": "ลำตัวสีแดงอมส้ม หัวลาดชัน มีหนวดคู่ใต้คางเด่นชัดใช้คุ้ยหาอาหารหน้าดิน",
        "commercial": "ปลาจับจากธรรมชาติ ราคาสูง เป็นที่ต้องการของร้านอาหารระดับมิชลิน",
        "culinary": "เนื้อนุ่ม รสหวานเฉพาะตัว มีกลิ่นหอมคล้ายกุ้งปู นิยมทอดในน้ำมันมะกอกหรือย่างไฟ",
        "badge_color": "badge-error"
    },
    "Red Sea Bream": {
        "thai_name": "ปลากะพงแดงญี่ปุ่น (ปลามาได / Madai)",
        "scientific_name": "Pagrus major",
        "family": "Sparidae (วงศ์ปลากะพงทรายแดง)",
        "market_grade": "ปลามงคลเกรดพรีเมียม / ตลาดประมูลปลา",
        "habitat": "มหาสมุทรแปซิฟิกตะวันตกเฉียงเหนือ ทะเลญี่ปุ่น",
        "field_marks": "ลำตัวสีชมพูประกายแดง มีจุดสีฟ้าเหลือบกระจายตามแนวสันหลัง ตากลมโต",
        "commercial": "ราชาแห่งปลาเนื้อขาว ราคาสูงที่สุดในกลุ่ม นิยมใช้ในงานเลี้ยงมงคลและเทศกาล",
        "culinary": "ซาชิมิเกรดบน ข้าวอบปลาไท (Taimeshi) นึ่งสาเก เนื้อสัมผัสเด้งหวานสะอาด",
        "badge_color": "badge-error"
    },
    "Sea Bass": {
        "thai_name": "ปลากะพงขาวเมดิเตอร์เรเนียน (บรันซิโน / Branzino)",
        "scientific_name": "Dicentrarchus labrax",
        "family": "Moronidae (วงศ์ปลากะพงขาว)",
        "market_grade": "ปลาตลาดสดสากล / ยอดนิยมตลอดกาล",
        "habitat": "ทะเลเมดิเตอร์เรเนียน ชายฝั่งแอตแลนติก และชะวากทะเล",
        "field_marks": "ลำตัวเพรียวยาว เกล็ดสีเงินเงา ครีบหลังมีก้านครีบแข็งแหลม ปากกว้าง",
        "commercial": "มีทั้งปลาจับจากธรรมชาติและฟาร์มกระชังส่งออกทั่วโลก ตลาดรองรับไม่จำกัด",
        "culinary": "เนื้อสีขาวนุ่ม รสหวานอ่อน ทำอาหารได้หลากหลาย ทั้งย่างกระทะ ทอดน้ำปลา นึ่งมะนาว",
        "badge_color": "badge-primary"
    },
    "Shrimp": {
        "thai_name": "กุ้ง (Prawn / Shrimp)",
        "scientific_name": "Decapoda / Caridea",
        "family": "Penaeidae (สัตว์น้ำมีเปลือก)",
        "market_grade": "สินค้าส่งออกทางน้ำอันดับหนึ่ง",
        "habitat": "แหล่งน้ำเค็ม น้ำกร่อย และน้ำจืดทั่วโลก",
        "field_marks": "เปลือกไคตินหุ้มตัว มีกรีแหลมที่ส่วนหัว ก้ามว่ายน้ำ และส่วนท้องโค้งงอ",
        "commercial": "ผลผลิตทางทะเลที่สร้างรายได้มหาศาล มีการคัดขนาด (Size Grading) ในการซื้อขาย",
        "culinary": "ต้มยำกุ้ง ชุบแป้งทอด เผาเนยกระเทียม เนื้อหวานเด้งกรอบธรรมชาติ",
        "badge_color": "badge-accent"
    },
    "Striped Red Mullet": {
        "thai_name": "ปลาบาร์บูนแดงลายแถบ (ซูร์มูเลต์ / Surmullet)",
        "scientific_name": "Mullus surmuletus",
        "family": "Mullidae (วงศ์ปลาแพะลาย)",
        "market_grade": "ปลาเกรดภัตตาคารชั้นนำ (High Commercial)",
        "habitat": "แนวโขดหินและทรายปนกรวดแถบเมดิเตอร์เรเนียน",
        "field_marks": "คล้าย Red Mullet แต่มีแถบสีเหลืองทอง 3-4 แถบพาดตามยาวลำตัวอย่างชัดเจน",
        "commercial": "ราคาสูงกว่า Red Mullet ธรรมดา เนื่องจากจับได้น้อยกว่าและรสชาติเข้มข้นกว่า",
        "culinary": "เซียร์กระทะร้อน เสิร์ฟพร้อมซอสเนยสมุนไพร เนื้อแน่นไขมันดีสูง",
        "badge_color": "badge-warning"
    },
    "Trout": {
        "thai_name": "ปลาเทราต์ (เรนโบว์เทราต์)",
        "scientific_name": "Oncorhynchus mykiss",
        "family": "Salmonidae (วงศ์ปลาแซลมอน)",
        "market_grade": "ปลาเศรษฐกิจน้ำเย็น / ฟาร์มเชิงนิเวศ",
        "habitat": "แม่น้ำและลำธารน้ำจืดที่สะอาด เย็นจัด และออกซิเจนสูง",
        "field_marks": "มีแถบสีชมพูอมม่วงเหลือบรุ้งด้านข้างลำตัว มีจุดสีดำเล็กๆ กระจายทั่วตัว",
        "commercial": "ปลาฟาร์มคุณภาพสูง เป็นทางเลือกทดแทนแซลมอน ราคาสมเหตุสมผล",
        "culinary": "ย่างเนยกระเทียม อบฟอยล์ หรือรมควัน เนื้อสีส้มอ่อน นุ่มละมุน ไม่คาว",
        "badge_color": "badge-success"
    }
}

@app.route("/")
def index():
    # List sample files
    sample_files = []
    if SAMPLES_DIR.exists():
        for f in sorted(list(SAMPLES_DIR.glob("*.png")) + list(SAMPLES_DIR.glob("*.jpg"))):
            clean_name = f.stem.replace("sample_", "").replace("_0000", "").replace("_", " ")
            sample_files.append({
                "filename": f.name,
                "species": clean_name
            })
    return render_template("index.html", 
                           samples=sample_files, 
                           models=list(models_dict.keys()),
                           accuracies=accuracies)

@app.route("/sample_images/<filename>")
def serve_sample_image(filename):
    return send_from_directory(SAMPLES_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form.get("model_name", "Support Vector Machine (SVM)")
        sample_filename = request.form.get("sample_filename")
        
        img = None
        if "image_file" in request.files and request.files["image_file"].filename != "":
            file = request.files["image_file"]
            img = Image.open(file.stream).convert("RGB")
        elif sample_filename:
            sample_path = SAMPLES_DIR / sample_filename
            if sample_path.exists():
                img = Image.open(sample_path).convert("RGB")
                
        if img is None:
            return jsonify({"error": "กรุณาอัปโหลดรูปภาพหรือเลือกภาพตัวอย่าง"}), 400
            
        # Optional: AI Smart Background Isolation (Removes cutting board, table, hands, etc.)
        isolate_bg = request.form.get("isolate_bg", "false").lower() == "true"
        processed_img_b64 = None
        img_for_eval = img
        
        if isolate_bg and REMBG_AVAILABLE:
            try:
                # Resize large camera images to max 512x512 for instant 1.5s background removal
                work_img = img.copy()
                work_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                nobg_rgba = remove(work_img, session=rembg_session)
                white_canvas = Image.new("RGB", nobg_rgba.size, (255, 255, 255))
                white_canvas.paste(nobg_rgba, mask=nobg_rgba.split()[3])
                img_for_eval = white_canvas
                
                # Base64 Preview for Web Frontend
                buf = io.BytesIO()
                white_canvas.save(buf, format="JPEG", quality=85)
                processed_img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as bg_err:
                print(f"Warning: Background isolation fallback: {bg_err}")
                img_for_eval = img

        # Feature Extraction with MobileNetV2
        tensor_img = preprocess(img_for_eval).unsqueeze(0)
        with torch.no_grad():
            feat = mobilenet(tensor_img).cpu().numpy()
            
        # Feature Scaling
        scaled_feat = scaler.transform(feat)
        
        # Predict with selected model
        selected_model = models_dict.get(model_name, models_dict["Support Vector Machine (SVM)"])
        probabilities = selected_model.predict_proba(scaled_feat)[0]
        
        pred_idx = int(np.argmax(probabilities))
        pred_species = class_names[pred_idx]
        confidence = float(probabilities[pred_idx] * 100)
        
        # Calculate Feature Vector Statistics (for ML Transparency & Demonstration)
        raw_features = feat[0]
        feature_stats = {
            "dimension": int(len(raw_features)),
            "mean": round(float(np.mean(raw_features)), 4),
            "std": round(float(np.std(raw_features)), 4),
            "max": round(float(np.max(raw_features)), 4),
            "min": round(float(np.min(raw_features)), 4),
            "sample_values": [round(float(v), 4) for v in raw_features[:32]],
            "scaled_sample": [round(float(v), 4) for v in scaled_feat[0][:32]]
        }
        
        # Rank top candidates
        candidates = []
        for cname, prob in zip(class_names, probabilities):
            candidates.append({
                "species": cname,
                "thai_name": FISHERY_INFO.get(cname, {}).get("thai_name", cname),
                "probability": round(float(prob * 100), 2)
            })
        candidates = sorted(candidates, key=lambda x: x["probability"], reverse=True)
        
        info = FISHERY_INFO.get(pred_species, {})
        
        return jsonify({
            "species": pred_species,
            "thai_name": info.get("thai_name", pred_species),
            "scientific_name": info.get("scientific_name", "N/A"),
            "family": info.get("family", "N/A"),
            "market_grade": info.get("market_grade", "N/A"),
            "habitat": info.get("habitat", "N/A"),
            "field_marks": info.get("field_marks", "N/A"),
            "commercial": info.get("commercial", "N/A"),
            "culinary": info.get("culinary", "N/A"),
            "confidence": round(confidence, 2),
            "model_used": model_name,
            "candidates": candidates[:4],
            "feature_stats": feature_stats,
            "processed_image": processed_img_b64,
            "is_isolated": isolate_bg
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Fishery Marine Web Application on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
