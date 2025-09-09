# C:\Users\sivah\OneDrive\Desktop\Plant-Leaf-Disease-Detection\app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Plant Leaf Disease Detection", layout="centered")

st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a clear leaf image. Toggle Debug to see model internals.")

# --------- SETTINGS ---------
MODEL_PATH = "models/leaf_disease_model.keras"
CLASS_JSON_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

# --------- Load class names ---------
if tf.io.gfile.exists(CLASS_JSON_PATH):
    with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    # fallback (edit if your classes differ)
    class_names = [
        'Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
        'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
        'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]

# --------- Load model with safe error message ---------
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error("Model load failed. Check models/leaf_disease_model.keras is present.")
    st.exception(e)
    st.stop()

# --------- Debug toggle ---------
show_debug = st.checkbox("Show debug info (model summary, raw preds, top3)")

# Model summary (capture as text)
if show_debug:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    summary_str = buf.getvalue()
    st.text("Model summary:")
    st.code(summary_str, language="text")

# --------- File uploader ---------
uploaded_file = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess: MUST match training preprocessing (MobileNetV2)
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32")  # shape (224,224,3)
    # print some stats if debug
    if show_debug:
        st.write("Pre-normalization stats:", {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())})
    # Use MobileNetV2 preprocess_input (this matches training script)
    x = np.expand_dims(arr, axis=0)  # (1,224,224,3)
    x = preprocess_input(x)          # Important: same preprocessing as during training
    if show_debug:
        st.write("Post-preprocess stats:", {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())})

    # Predict
    preds = model.predict(x)  # shape (1, num_classes)
    if show_debug:
        st.write("Raw predictions (array):", preds.tolist())

    probs = preds[0]
    top_idx = probs.argsort()[-3:][::-1]   # top-3 indices

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"index_{pred_idx}"
    confidence = float(probs[pred_idx]) * 100

    st.success(f"Predicted: **{pred_label}**")
    st.info(f"🔍 Confidence: {confidence:.2f}%")

    # Show top-3 nicely
    cols = st.columns(3)
    for i, idx in enumerate(top_idx):
        label = class_names[idx] if idx < len(class_names) else f"index_{idx}"
        pct = probs[idx] * 100
        cols[i].metric(label, f"{pct:.2f}%")
