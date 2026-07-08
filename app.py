import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import pandas as pd
from huggingface_hub import hf_hub_download

# =====================================================

# CONFIGURATION

# =====================================================

st.set_page_config(
page_title="Disaster Damage Assessment",
page_icon="🏚️",
layout="wide"
)

DEVICE = torch.device(
"cuda" if torch.cuda.is_available()
else "cpu"
)

#MODEL_PATH = r"D:\Project Infy Image\best_unet_resnet34.pth"

NUM_CLASSES = 5

CLASS_NAMES = {
0: "Background",
1: "No Damage",
2: "Minor Damage",
3: "Major Damage",
4: "Destroyed"
}

COLOR_MAP = {
0: [0, 0, 0],
1: [0, 255, 0],
2: [255, 255, 0],
3: [255, 165, 0],
4: [255, 0, 0]
}

# =====================================================

# MODEL LOADING

# =====================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="Neel-Bokil/Only_Post_Disaster_Resilience_Model",
        filename="best_unet_resnet34_post_only.pth"
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# =====================================================

# PREPROCESSING

# =====================================================

transform = T.Compose([
T.Resize((256, 256)),
T.ToTensor(),
T.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

# =====================================================

# HELPER FUNCTIONS

# =====================================================

# colorize mask
def colorize_mask(mask):

    h, w = mask.shape

    rgb = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color

    return rgb

# create overlay
def create_overlay(image, mask_rgb):

    image = np.array(image)

    image = cv2.resize(
        image,
        (mask_rgb.shape[1], mask_rgb.shape[0])
    )

    overlay = cv2.addWeighted(
        image,
        0.7,
        mask_rgb,
        0.3,
        0
    )

    return overlay


# damage report
def damage_report(mask):

    total_pixels = mask.size

    rows = []

    for cls_id, name in CLASS_NAMES.items():

        pixels = np.sum(mask == cls_id)

        percentage = (
            pixels / total_pixels
        ) * 100

        rows.append([
            name,
            pixels,
            round(percentage, 2)
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "Class",
            "Pixel Count",
            "Percentage (%)"
        ]
    )


# damage severity
def get_damage_severity(mask):

    total_pixels = mask.size

    major_damage = np.sum(mask == 3)
    destroyed = np.sum(mask == 4)

    severe_percentage = (
        (major_damage + destroyed)
        / total_pixels
    ) * 100

    if severe_percentage < 1:
        return (
            "Low Damage",
            "🟢"
        )

    elif severe_percentage < 5:
        return (
            "Moderate Damage",
            "🟡"
        )

    elif severe_percentage < 10:
        return (
            "High Damage",
            "🟠"
        )

    else:
        return (
            "Severe Damage",
            "🔴"
        )

# predict mask
def predict_mask(post_image):

    post_img = (
        post_image
        .convert("RGB")
        .resize((256, 256))
    )

    post_tensor = transform(post_img)

    x = post_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        pred_mask = torch.argmax(
            logits,
            dim=1
        )

    pred_mask = (
        pred_mask
        .squeeze()
        .cpu()
        .numpy()
    )

    return post_img, pred_mask

# =====================================================
# UI
# =====================================================

st.title("🏚️ Disaster Damage Assessment System")

st.markdown(
    """
Upload a **Post-Disaster Satellite Image**.

The model will:

- Detect damaged buildings
- Classify damage severity
- Generate a segmentation mask
- Create a damage overlay
- Produce a damage assessment report
"""
)

post_file = st.file_uploader(
    "Upload Post-Disaster Image",
    type=["png", "jpg", "jpeg"]
)

# =====================================================

# PREDICTION

# =====================================================

if post_file:

    post_image = Image.open(post_file)

    with st.spinner("Running damage assessment..."):

        post_img, pred_mask = predict_mask(
            post_image
        )

        pred_rgb = colorize_mask(
            pred_mask
        )

        overlay = create_overlay(
            post_img,
            pred_rgb
        )

        report_df = damage_report(
            pred_mask
        )

        severity_label, severity_icon = (
            get_damage_severity(
                pred_mask
            )
        )

    st.success("Prediction Complete")

    st.subheader("Overall Damage Severity")

    st.metric(
        label="Assessment",
        value=f"{severity_icon} {severity_label}"
    )

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            post_img,
            caption="Post-Disaster Image",
            use_container_width=True
        )

    with c2:
        st.image(
            pred_rgb,
            caption="Predicted Damage Mask",
            use_container_width=True
        )

    with c3:
        st.image(
            overlay,
            caption="Damage Overlay",
            use_container_width=True
        )

    st.subheader("Damage Assessment Report")

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.subheader("Damage Legend")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.success("🟩 No Damage")

    with c2:
        st.warning("🟨 Minor Damage")

    with c3:
        st.warning("🟧 Major Damage")

    with c4:
        st.error("🟥 Destroyed")
