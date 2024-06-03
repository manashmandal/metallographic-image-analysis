import streamlit as st
import io
import numpy as np
from preprocess import read_bytes_as_tensor, prediction_transforms, index_to_label_map
from transformers import ViTForImageClassification
import torch
import datetime

st.title("Metallographic Image Analysis")


def on_change(*args, **kwargs) -> None:
    print("args", args, kwargs)


if __name__ in ("__main__", "st_app"):
    model_path = "vit-metal"
    model = ViTForImageClassification.from_pretrained(model_path)

    uploaded_file = st.file_uploader(
        "Upload an image", on_change=on_change, type=["bmp"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.")

        image_array = read_bytes_as_tensor(uploaded_file.read())
        transformed = prediction_transforms(image_array)

        prediction = torch.argmax(model(transformed.unsqueeze(0)).logits)

        st.write(
            f"Prediction: <b>{index_to_label_map[prediction.item()]}</b> at <i>{datetime.datetime.now()}</i>",
            unsafe_allow_html=True,
        )
