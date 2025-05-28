import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np
from pathlib import Path
st.set_page_config(page_title="Classificador de Pacotes", layout="centered")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # volta do src/ para a raiz do projeto
MODEL_PATH = BASE_DIR / "src/utils/dados/models/package_inspection/best.pth"


# ========== CONFIGURAÇÃO ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['damaged', 'intact']

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_model()

# ========== STREAMLIT UI ==========
st.title("📦 Classificador de Pacotes - Danificado ou Intacto")
st.markdown("Envie uma **imagem de pacote** para saber se está danificado ou intacto.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem enviada", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    st.markdown(f"### ✅ Previsão: **{pred_class.upper()}**")
    st.markdown(f"Confiança: **{confidence:.2f}%**")

    st.progress(min(int(confidence), 100))
