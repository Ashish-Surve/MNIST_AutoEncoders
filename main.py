import time

import requests
import streamlit as st
from PIL import Image

import pandas as pd
from streamlit_drawable_canvas import st_canvas


st.title("Limit Breaker Classifier")

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="" if bg_image else bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

if st.button("Find Car!"):
    if image is not None and style is not None:
        files = {"file": image.getvalue()}
        # DEBUG
        #res = requests.post(f"http://127.0.0.1:8000/{style}", files=files)
        res = requests.post(f"http://backend:8080/{style}", files=files)
        img_path = res.json()
        # DEBUG
        # image_p = r"../backend/"+ img_path.get("name")
        # Docker
        image_p = img_path.get("name")
        image = Image.open(image_p)

        st.image(image)

        displayed_styles = [style]
        displayed = 1
        total = len(STYLES)

