import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import tempfile
import numpy as np
import os
import gdown

st.set_page_config(page_title="Human Action Recognition", layout="wide", page_icon="🏃")

# 1. Define Settings and Classes
CLASSES = [
    'calling', 'clapping', 'cycling', 'dancing', 'drinking', 
    'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music', 
    'running', 'sitting', 'sleeping', 'texting', 'using_laptop'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. ResNet Scratch Definition
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=15):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 3. Model Loading Logic
def ensure_model_exists(file_id, path):
    os.makedirs('saved_models', exist_ok=True)
    if not os.path.exists(path):
        st.info(f"Downloading model to {path} from Google Drive. Please wait...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, path, quiet=False)
        st.success(f"Successfully downloaded {path}!")

@st.cache_resource
def load_resnet():
    path = 'saved_models/resnet18_scratch.pth'
    ensure_model_exists('1TFN7sy4F9sZYIPHQ7skLqOncE-vDgILz', path)
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception:
        pass # Model won't work properly if weights aren't found, handle in UI
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_vit():
    path = 'saved_models/vit_base_patch16.pth'
    ensure_model_exists('18pLbpxEKw6i0NK_k7WZXHCqdrD8fbV7M', path)
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception:
        pass
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_efficientnet():
    path = 'saved_models/efficientnet_b0.pth'
    ensure_model_exists('13DuHSzGn6f7JWxTwCrBnQDi27qjv3uIi', path)
    
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception:
        pass
    model.to(DEVICE)
    model.eval()
    return model

# Setup Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_frame(img_pil, model):
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    return CLASSES[predicted_idx.item()], confidence.item()

# 4. Build Streamlit UI
st.title("🏃 Human Action Recognition")
st.markdown("This application performs human action classification using trained deep learning models. Select a model and supply an input format to get started!")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["EfficientNet (Recommended)", "Vision Transformer (ViT)", "ResNet18 (Scratch)"])
input_type = st.sidebar.selectbox("Choose Input Stream", ["Image File", "Video File", "Live Camera", "Camera Snapshot"])

# Load Selected Model
if model_choice == "ResNet18 (Scratch)":
    selected_model = load_resnet()
elif model_choice == "Vision Transformer (ViT)":
    selected_model = load_vit()
else:
    selected_model = load_efficientnet()

st.sidebar.markdown("---")
st.sidebar.info("The models expect 15 action classes like *calling, dancing, running, using_laptop, etc.*")

# Image File Input
if input_type == "Image File":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")
        
        label, conf = predict_frame(image, selected_model)
        st.success(f"**Prediction:** {label} ({conf*100:.2f}% confidence)")

# Video File Input
elif input_type == "Video File":
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save temp video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        if st.button("Process Video Context"):
            st.write("Analyzing Keyframes...")
            cap = cv2.VideoCapture(tfile.name)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            predictions = []
            
            # Sample 1 frame per second
            for i in range(0, frame_count, int(fps)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label, conf = predict_frame(img_pil, selected_model)
                predictions.append(label)
                
            cap.release()
            
            # Simple aggregation (most frequent element)
            if predictions:
                most_common = max(set(predictions), key=predictions.count)
                st.success(f"**Overall Video Prediction:** {most_common}")
                st.write(f"Frame-by-frame snapshot: {predictions}")
            else:
                st.error("No valid frames could be extracted from this video.")

# Live Camera Feed
elif input_type == "Live Camera":
    st.header("Live Camera Feed")
    st.markdown("Ensure your browser and local Python environment have permissions to access the webcam.")
    run = st.checkbox('Start Live Webcam Stream')
    
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        # Streamlit while loop
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read frame from webcam.")
                break
                
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            label, conf = predict_frame(img_pil, selected_model)
            
            # Overlay info
            cv2.putText(img_rgb, f"{label} ({conf*100:.1f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(img_rgb)
            
            # Allow Streamlit to inject updates smoothly (fixes UI block)
            st.session_state["dummy"] = "dummy"
            
        cap.release()
    else:
        st.write("Click 'Start Live Webcam Stream' to begin.")

# Camera Snapshot (Native Streamlit)
elif input_type == "Camera Snapshot":
    st.header("Take a Snapshot")
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        image = Image.open(camera_photo).convert('RGB')
        label, conf = predict_frame(image, selected_model)
        st.success(f"**Prediction:** {label} ({conf*100:.2f}% confidence)")
