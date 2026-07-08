# 🏚️ Disaster Damage Assessment using Post-Disaster Satellite Images

This project presents an AI-powered disaster damage assessment system that performs semantic segmentation on **post-disaster satellite imagery** to identify and classify structural damage.

Unlike traditional change-detection approaches that require both pre-disaster and post-disaster images, this model predicts damage severity using only a **single post-disaster image**, making it more practical for rapid disaster response.

The project is built using **U-Net with a ResNet34 encoder**, trained on the xBD (xView2) disaster dataset and deployed with **Streamlit**.

---

## Features

- Semantic segmentation of damaged buildings
- Uses only post-disaster satellite images
- Pixel-wise classification into five damage categories
- Interactive Streamlit web application
- Damage overlay visualization
- Damage assessment report
- Overall damage severity prediction

---

## Damage Classes

| Class | Description |
|--------|-------------|
| Background | Non-building regions |
| No Damage | Buildings without visible damage |
| Minor Damage | Slight structural damage |
| Major Damage | Significant structural damage |
| Destroyed | Completely destroyed structures |

---

## Model Architecture

- Architecture: U-Net
- Encoder: ResNet34
- Framework: PyTorch
- Input Size: 256 × 256
- Input Channels: 3 (Post-disaster RGB image)
- Output Classes: 5

---

## Dataset

The model is trained on the **xBD (xView2)** disaster dataset containing satellite imagery captured before and after natural disasters with pixel-level damage annotations.

---

## Technologies Used

- Python
- PyTorch
- Segmentation Models PyTorch
- OpenCV
- Albumentations
- NumPy
- Pandas
- Streamlit
- Hugging Face Hub

---

## Project Structure

```text
Post-Disaster-Damage-Assessment/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
    ├── 01-dataset-exploration.ipynb
    ├── 02-preprocessing-and-dataset-builder.ipynb
    ├── 03-mask-generation-and-dataset-pipeline.ipynb
    ├── 04-for-post-only-unet-training-smp.ipynb
    └── 05-for-post-only-inference-visualization.ipynb

```

The trained model is hosted separately on Hugging Face due to GitHub file size limitations.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Disaster-Damage-Assessment-Post-Disaster-Images.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Model

The trained model is hosted on Hugging Face and is downloaded automatically when the application is launched.

---

## Future Improvements

- Improve segmentation accuracy with longer training
- Experiment with DeepLabV3+ and U-Net++
- Add confidence visualization
- Support larger image resolutions
- Deploy using Docker

---

## Author

**Neel Bokil**

---

## License

This project is intended for educational and research purposes.
