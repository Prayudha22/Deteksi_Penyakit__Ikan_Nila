import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="Deteksi Penyakit Ikan Nila",
    page_icon="🐟",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

import gdown
import os
import tensorflow as tf

MODEL_PATH = "mobilenet_tilapia_disease.h5"

FILE_ID = "178Qi75NiHPHsS3F-uN4llxNibp2Hlmyg"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ==========================
# KELAS DATASET
# ==========================

CLASS_NAMES = [
    "MAS",
    "NN",
    "Prd",
    "TiLV"
]

disease_info = {

    "MAS":
    """
    Motile Aeromonas Septicemia (MAS)
    merupakan penyakit bakteri yang
    disebabkan oleh Aeromonas hydrophila.
    """,

    "NN":
    """
    Normal Nile Tilapia.
    Ikan berada dalam kondisi sehat.
    """,

    "Prd":
    """
    Parasitic Disease.
    Penyakit yang disebabkan oleh
    serangan organisme parasit.
    """,

    "TiLV":
    """
    Tilapia Lake Virus.
    Penyakit yang disebabkan oleh
    infeksi virus TiLV.
    """
}

# ==========================
# SIDEBAR
# ==========================

menu = st.sidebar.radio(
    "Menu",
    [
        "Beranda",
        "Deteksi Penyakit",
        "Informasi Model",
        "Tentang Penyakit"
    ]
)

# ==========================
# BERANDA
# ==========================

if menu == "Beranda":

    st.title(
        "🐟 Sistem Deteksi Penyakit Ikan Nila"
    )

    st.markdown("""
    ### Convolutional Neural Network (CNN)
    ### Transfer Learning MobileNetV2
    """)

    st.success(
        "Akurasi Model : 95%"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Kelas",
        "4"
    )

    col2.metric(
        "Model",
        "MobileNetV2"
    )

    col3.metric(
        "Dataset",
        "2124 Citra"
    )

# ==========================
# DETEKSI
# ==========================

elif menu == "Deteksi Penyakit":

    st.title(
        "🔍 Deteksi Penyakit Ikan Nila"
    )

    uploaded_file = st.file_uploader(
        "Upload gambar ikan nila",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Gambar Input",
                use_container_width=True
            )

        # preprocessing

        img = image.resize((224,224))

        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        pred_index = np.argmax(
            prediction
        )

        pred_class = CLASS_NAMES[
            pred_index
        ]

        confidence = (
            np.max(prediction) * 100
        )

        with col2:

            st.success(
                f"Hasil Prediksi : {pred_class}"
            )

            st.info(
                disease_info[pred_class]
            )

            st.metric(
                "Tingkat Keyakinan",
                f"{confidence:.2f}%"
            )

        st.subheader(
            "Probabilitas Setiap Kelas"
        )

        prob_df = pd.DataFrame({
            "Kelas": CLASS_NAMES,
            "Probabilitas (%)":
            prediction[0] * 100
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        st.bar_chart(
            prob_df.set_index(
                "Kelas"
            )
        )

# ==========================
# INFORMASI MODEL
# ==========================

elif menu == "Informasi Model":

    st.title(
        "📊 Informasi Model"
    )

    st.write(
        "Model yang digunakan adalah MobileNetV2 dengan Transfer Learning."
    )

    st.write(
        "Input Image : 224 x 224 piksel"
    )

    st.write(
        "Jumlah Kelas : 4"
    )

    st.write(
        "Optimizer : Adam"
    )

    st.write(
        "Learning Rate : 0.00001"
    )

    st.write(
        "Batch Size : 32"
    )

    st.write(
        "Akurasi Pengujian : 95%"
    )

# ==========================
# TENTANG PENYAKIT
# ==========================

elif menu == "Tentang Penyakit":

    st.title(
        "📖 Informasi Penyakit Ikan Nila"
    )

    for disease, info in disease_info.items():

        st.subheader(
            disease
        )

        st.write(
            info
        )