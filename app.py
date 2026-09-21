import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Fish Species Classification System",
    layout="wide"
)

# Species Knowledge Base
SPECIES_DATA = {
    "Black Sea Sprat": {
        "thai_name": "ปลาสเปรตทะเลดำ",
        "scientific_name": "Clupeonella cultriventris",
        "family": "Clupeidae",
        "habitat": "Black Sea, Caspian Sea, and Sea of Azov basins (freshwater, brackish, and marine).",
        "morphology": "Small elongated clupeid (8-10 cm), silvery coloration, distinct ventral keel.",
        "commercial": "Commercially significant forage fish used for smoking, canning, and marine protein products. Rich in Omega-3."
    },
    "Gilt-Head Bream": {
        "thai_name": "ปลากะพงทรายสีทอง (โดราด)",
        "scientific_name": "Sparus aurata",
        "family": "Sparidae",
        "habitat": "Mediterranean Sea and eastern coastal regions of the North Atlantic Ocean.",
        "morphology": "Deep, oval compressed body with a prominent golden band across the interorbital space and dark patch at gill cover.",
        "commercial": "High-value commercial aquaculture species across Europe. Firm white flesh with delicate sweet profile."
    },
    "Hourse Mackerel": {
        "thai_name": "ปลาทูแขก หรือ ปลาอาจิ (Aji)",
        "scientific_name": "Trachurus trachurus",
        "family": "Carangidae",
        "habitat": "Northeast Atlantic Ocean, North Sea, and Mediterranean Sea in large pelagic schools.",
        "morphology": "Slender body with prominent, heavily ossified lateral scutes and blue-green dorsal shading.",
        "commercial": "Major industrial commercial fish. Popular in Japanese culinary cuisine as Aji for sashimi and light pan-frying."
    },
    "Red Mullet": {
        "thai_name": "ปลาบาร์บูนแดง",
        "scientific_name": "Mullus barbatus",
        "family": "Mullidae",
        "habitat": "Sandy and muddy bottoms across the Mediterranean Sea and Black Sea.",
        "morphology": "Distinct pinkish-red coloration, steep profile, with prominent paired chin barbels for benthic probing.",
        "commercial": "Classic Mediterranean delicacy prized since ancient Roman times. Highly aromatic, crustacean-like flavor note."
    },
    "Red Sea Bream": {
        "thai_name": "ปลากะพงแดงญี่ปุ่น หรือ ปลามาได (Madai)",
        "scientific_name": "Pagrus major",
        "family": "Sparidae",
        "habitat": "Northwest Pacific Ocean, including coastal waters of Japan and East China Sea.",
        "morphology": "Vivid pinkish-red body with light blue iridescent speckles along dorsal line.",
        "commercial": "Regarded as the premier white fish in Japan (Madai), associated with celebration and auspicious ceremonies. Dense, pristine flesh."
    },
    "Sea Bass": {
        "thai_name": "ปลากะพงขาวเมดิเตอร์เรเนียน (Branzino)",
        "scientific_name": "Dicentrarchus labrax",
        "family": "Moronidae",
        "habitat": "Eastern Atlantic Ocean and Mediterranean Sea in coastal waters and estuaries.",
        "morphology": "Elongated silver body with sharp dorsal spines and large terminal mouth.",
        "commercial": "Global fine-dining standard (Branzino). Delicate texture, low fat content, and clean culinary versatility."
    },
    "Shrimp": {
        "thai_name": "กุ้ง (Prawn / Shrimp)",
        "scientific_name": "Decapoda / Caridea",
        "family": "Penaeidae",
        "habitat": "Marine, estuarine, and freshwater benthic zones globally.",
        "morphology": "Segmented exoskeleton, distinct cephalothorax, elongated rostrum, and five pairs of swimming pleopods.",
        "commercial": "Top globally traded seafood commodity. Natural sweet crunch, versatile high-protein staple across all global cuisines."
    },
    "Striped Red Mullet": {
        "thai_name": "ปลาบาร์บูนแดงลายแถบ (Surmullet)",
        "scientific_name": "Mullus surmuletus",
        "family": "Mullidae",
        "habitat": "Rocky and gravelly Mediterranean coastlines and eastern Atlantic shelf.",
        "morphology": "Reddish-orange hue with 3-4 longitudinal golden-yellow stripes along body flank.",
        "commercial": "Premium gastronomic fish commanding higher prices than plain Red Mullet. High unsaturated fat content and intense aroma."
    },
    "Trout": {
        "thai_name": "ปลาเทราต์ (Rainbow Trout)",
        "scientific_name": "Oncorhynchus mykiss",
        "family": "Salmonidae",
        "habitat": "Cold-water river systems, alpine lakes, and managed aquaculture facilities.",
        "morphology": "Distinct pink-reddish lateral stripe, dark spots over silvery-green olive body.",
        "commercial": "Foremost freshwater commercial sport and food fish. Tender, mild salmonid meat suitable for baking and cold smoking."
    }
}

# Feature Extractor Loader
@st.cache_resource
def get_feature_extractor():
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet = models.mobilenet_v2(weights=weights)
    mobilenet.eval()
    mobilenet.classifier = nn.Identity()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return mobilenet, transform

# Model Bundle Loader
@st.cache_resource
def get_model_bundle():
    model_path = Path("model/fish_classifier.joblib")
    if not model_path.exists():
        return None
    return joblib.load(model_path)

# Header
st.title("Fish Species Classification System")
st.caption("Automated image classification pipeline utilizing MobileNetV2 feature extraction and Scikit-learn algorithms.")
st.divider()

bundle = get_model_bundle()

if bundle is None:
    st.error("Model file not found. Please verify that 'model/fish_classifier.joblib' is available.")
    st.stop()

# Layout: 2 Columns (Left: Input & Settings, Right: Results)
col_left, col_right = st.columns([5, 7], gap="large")

with col_left:
    with st.container(border=True):
        st.subheader("Model & Input Configuration")
        
        # Model Selection Toggle
        available_models = list(bundle["models"].keys())
        selected_model_name = st.selectbox(
            "Classification Algorithm:",
            options=available_models,
            index=0
        )
        
        # Display model test accuracy if recorded
        acc_text = bundle.get("accuracies", {}).get(selected_model_name, "N/A")
        st.caption(f"Selected Model: **{selected_model_name}** | Benchmark Test Accuracy: **{acc_text}**")
        
        st.write("")
        
        # Image Source Selection
        source_mode = st.radio(
            "Image Input Source:",
            options=["Sample Dataset", "Upload File"],
            horizontal=True
        )
        
        selected_image = None
        
        if source_mode == "Sample Dataset":
            samples_dir = Path("sample_test_images")
            sample_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
            
            options_dict = {}
            for f in sample_files:
                clean_name = f.stem.replace("sample_", "").replace("_0000", "").replace("_", " ")
                options_dict[clean_name] = f
                
            chosen_name = st.selectbox("Select Test Image:", list(options_dict.keys()))
            if chosen_name:
                selected_path = options_dict[chosen_name]
                selected_image = Image.open(selected_path).convert("RGB")
                st.caption(f"Image: {chosen_name} ({selected_image.size[0]} x {selected_image.size[1]} px)")
        else:
            uploaded_file = st.file_uploader(
                "Upload Image File (JPG, PNG):",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                st.caption(f"Uploaded: {uploaded_file.name} ({selected_image.size[0]} x {selected_image.size[1]} px)")
                
        if selected_image:
            st.image(selected_image, width=380)
            predict_button = st.button("Run Classification", type="primary")
        else:
            predict_button = False

with col_right:
    if predict_button and selected_image:
        with st.spinner("Processing image and running inference..."):
            mobilenet, transform = get_feature_extractor()
            
            # Feature extraction
            tensor_img = transform(selected_image).unsqueeze(0)
            with torch.no_grad():
                features = mobilenet(tensor_img).cpu().numpy()
                
            # Scale features
            scaled_features = bundle["scaler"].transform(features)
            
            # Predict with selected model
            clf = bundle["models"][selected_model_name]
            class_names = bundle["class_names"]
            
            probabilities = clf.predict_proba(scaled_features)[0]
            pred_idx = np.argmax(probabilities)
            pred_class = class_names[pred_idx]
            confidence = probabilities[pred_idx] * 100
            
            info = SPECIES_DATA.get(pred_class, {
                "thai_name": pred_class,
                "scientific_name": "N/A",
                "family": "N/A",
                "habitat": "N/A",
                "morphology": "N/A",
                "commercial": "N/A"
            })
            
            # Primary Result Card
            with st.container(border=True):
                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    st.subheader(pred_class)
                    st.write(f"**Thai Designation:** {info['thai_name']}")
                    st.caption(f"Evaluated via **{selected_model_name}**")
                with res_col2:
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                
                st.divider()
                
                tab_info, tab_prob = st.tabs(["Species Specification", "Confidence Distribution"])
                
                with tab_info:
                    st.markdown(f"**Scientific Name & Family:** *{info['scientific_name']}* ({info['family']})")
                    st.markdown(f"**Physical Morphology:** {info['morphology']}")
                    st.markdown(f"**Natural Habitat:** {info['habitat']}")
                    st.markdown(f"**Commercial & Culinary Usage:** {info['commercial']}")
                    
                with tab_prob:
                    prob_records = []
                    for cname, pval in zip(class_names, probabilities):
                        prob_records.append({
                            "Species": cname,
                            "Probability (%)": round(pval * 100, 2)
                        })
                    prob_df = pd.DataFrame(prob_records).sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(
                        prob_df,
                        column_config={
                            "Species": st.column_config.TextColumn("Species Name"),
                            "Probability (%)": st.column_config.ProgressColumn(
                                "Confidence (%)",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100
                            )
                        },
                        hide_index=True
                    )
    else:
        with st.container(border=True):
            st.write("#### Classification Output")
            st.caption("Select an image from the left panel and click 'Run Classification' to perform inference.")
