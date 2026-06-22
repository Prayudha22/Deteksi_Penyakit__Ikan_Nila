import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import gdown
import os

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Deteksi Penyakit Ikan Nila",
    page_icon="🐟",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e9ecef;
}

.stAlert {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DOWNLOAD & LOAD MODEL
# =====================================================

MODEL_PATH = "mobilenet_tilapia_disease.keras"

FILE_ID = "15KSWyeRQm0kfLU1VrEaWC3eQ_s9o93Uh"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("📥 Mengunduh model AI dari Google Drive..."):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

    with st.spinner("🧠 Memuat model AI..."):

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    return model


model = load_model()

# =====================================================
# LABEL KELAS
# =====================================================

CLASS_NAMES = [
    "MAS",
    "NN",
    "Prd",
    "TiLV"
]

# =====================================================
# INFORMASI PENYAKIT
# =====================================================

disease_info = {

    "MAS":
    """
    Motile Aeromonas Septicemia (MAS)
    merupakan penyakit bakteri yang disebabkan oleh
    Aeromonas hydrophila.
    """,

    "NN":
    """
    Normal Nile Tilapia.
    Ikan berada dalam kondisi sehat.
    """,

    "Prd":
    """
    Parasitic Disease.
    Penyakit akibat infeksi organisme parasit.
    """,

    "TiLV":
    """
    Tilapia Lake Virus (TiLV).
    Penyakit akibat infeksi virus TiLV.
    """
}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🐟 Deteksi Penyakit Ikan Nila")

menu = st.sidebar.radio(
    "Menu",
    [
        "Beranda",
        "Deteksi Penyakit",
        "Informasi Model",
        "Tentang Penyakit"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Sistem berbasis Deep Learning
menggunakan Transfer Learning
MobileNetV2.
""")

# =====================================================
# BERANDA
# =====================================================

if menu == "Beranda":

    st.title("🐟 Sistem Deteksi Penyakit Ikan Nila")

    st.markdown("""
    ### Deep Learning dengan Transfer Learning MobileNetV2
    """)

    st.info("""
    Sistem ini digunakan untuk mendeteksi penyakit ikan nila
    berdasarkan citra visual menggunakan model Deep Learning.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Kelas",
        "4"
    )

    col2.metric(
        "Dataset",
        "2124"
    )

    col3.metric(
        "Model",
        "MobileNetV2"
    )

    col4.metric(
        "Akurasi",
        "95%"
    )

    st.divider()

    st.subheader("📈 Performa Sistem")

    st.write("""
    Sistem mampu mengidentifikasi:

    - MAS (Motile Aeromonas Septicemia)
    - NN (Normal Nile Tilapia)
    - Prd (Parasitic Disease)
    - TiLV (Tilapia Lake Virus)
    """)

# =====================================================
# DETEKSI PENYAKIT
# =====================================================

elif menu == "Deteksi Penyakit":

    st.title("🔍 Deteksi Penyakit Ikan Nila")

    uploaded_file = st.file_uploader(
        "Upload gambar ikan nila",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

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

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        with st.spinner("🔍 Menganalisis gambar..."):

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

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

        st.divider()

        prob_df = pd.DataFrame({

            "Kelas": CLASS_NAMES,

            "Probabilitas (%)":
            prediction[0] * 100
        })

        st.subheader(
            "🏆 Top 3 Prediksi"
        )

        top3 = prob_df.sort_values(
            by="Probabilitas (%)",
            ascending=False
        ).head(3)

        st.dataframe(
            top3,
            use_container_width=True
        )

        st.subheader(
            "📊 Probabilitas Setiap Kelas"
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        st.bar_chart(
            prob_df.set_index(
                "Kelas"
            )
        )

# =====================================================
# INFORMASI MODEL
# =====================================================

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

# =====================================================
# TENTANG PENYAKIT
# =====================================================

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

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <b>Prayudha Ragil Musthofa</b><br>
    Teknik Informatika<br>
    Sistem Deteksi Penyakit Ikan Nila Menggunakan Deep Learning
    </center>
    """,
    unsafe_allow_html=True
)
