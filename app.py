import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

from models import Generator, Classifier, get_noise

# ------------------- Config -------------------
z_dim = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 40
feature_names = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
    "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]

#Loading pretrained models
@st.cache_resource
def load_models():
    gen = Generator(z_dim=z_dim).to(device)
    gen.load_state_dict(torch.load("Files/pretrained_celeba.pth", map_location=device)["gen"])
    gen.eval()

    clf = Classifier(n_classes=n_classes).to(device)
    clf.load_state_dict(torch.load("Files/pretrained_classifier.pth", map_location=device)["classifier"])
    clf.eval()
    return gen, clf

gen, classifier = load_models()

def calculate_updated_noise(noise, weight):
    return noise + (weight * noise.grad)

def get_score(current_classifications, original_classifications, target_idx, other_indices, penalty_weight):
    other_diff = current_classifications[:, other_indices] - original_classifications[:, other_indices]
    other_penalty = -penalty_weight * other_diff.norm(dim=1).mean()
    target_score = current_classifications[:, target_idx].mean()
    return target_score + other_penalty

def show_tensor_images(tensor, nrow=5):
    tensor = (tensor + 1) / 2
    grid = make_grid(tensor.detach().cpu(), nrow=nrow)
    np_img = grid.permute(1, 2, 0).numpy()
    return Image.fromarray((np_img * 255).astype(np.uint8))

st.title("🌀 Latent Space Walk")
st.markdown("Move through the GAN's latent space toward specific **face attributes**")

# Controls
target_feature = st.selectbox("🎯 Target Attribute", feature_names)
n_images = st.slider("🖼 Number of Images", 1, 8, 4)
grad_steps = st.slider("📉 Gradient Steps", 1, 50, 10)
step_strength = st.slider("📈 Step Strength", 0.1, 5.0, 1.0)
penalty_weight = st.slider("⚖️ Penalty Weight", 0.0, 1.0, 0.1)

if st.button("🚀 Start Latent Walk"):
    st.write("Generating...")

    target_idx = feature_names.index(target_feature)
    other_indices = [i for i in range(n_classes) if i != target_idx]

    noise = get_noise(n_images, z_dim, device=device).requires_grad_()
    original_classifications = classifier(gen(noise)).detach()
    fake_image_history = []

    for step in range(grad_steps):
        fake = gen(noise)
        fake_image_history.append(fake)

        scores = classifier(fake)
        score = get_score(scores, original_classifications, target_idx, other_indices, penalty_weight)

        noise.grad = None
        score.backward()
        noise.data = calculate_updated_noise(noise, step_strength / grad_steps)

    for i, imgs in enumerate(fake_image_history):
        st.image(show_tensor_images(imgs, nrow=n_images), caption=f"Step {i + 1}", use_container_width=True)
