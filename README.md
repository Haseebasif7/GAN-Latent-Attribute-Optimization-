# GAN-Based Attribute Vector Optimization App

This project implements an interactive interface for **gradient-based semantic editing of latent representations** in a pretrained GAN. The system modifies the input noise vector to enhance specific target attributes (e.g., `Young`, `Smiling`) using classifier-based feedback.

## 🧠 Overview

- A random latent code `z ∈ ℝ⁶⁴` is sampled.
- A pretrained **Generator** synthesizes images from `z`.
- A frozen **Attribute Classifier** evaluates semantic attributes.
- The latent vector `z` is iteratively updated via **gradient ascent** to increase the classifier’s score for a selected attribute.
- A penalty term preserves non-target attributes.

## 🧩 Components

- `Generator`: DCGAN-style transposed convolutional network
- `Classifier`: CNN trained on CelebA attributes (multi-label)
- `get_score`: Combines target confidence + L2 penalty on other attributes
- `calculate_updated_noise`: Gradient-based latent update step


## 🚀 Getting Started

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py


