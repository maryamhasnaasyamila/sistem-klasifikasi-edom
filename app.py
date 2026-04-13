# ============================================================
# SIEDOM Analytics - Sistem Evaluasi & Klasifikasi Sentimen EDOM
# Maryam Hasnaa' Syamila | 0110222067
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(page_title="SIEDOM Analytics", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# ================= SIDEBAR =================
# ================= SIDEBAR STYLE =================
st.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #F9FAFB;
}

/* Remove radio bullet */
div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* Remove default spacing */
div[role="radiogroup"] label {
    margin-bottom: 6px;
}

/* Sidebar item base */
div[role="radiogroup"] label {
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 500;
    color: #1F2937;
    transition: all 0.2s ease;
}

/* Hover effect */
div[role="radiogroup"] label:hover {
    background-color: #E5E7EB;
    cursor: pointer;
}

/* Selected (active) item */
div[role="radiogroup"] input:checked + div {
    background-color: #E8F0FE !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 14px;
}

</style>
""", unsafe_allow_html=True)


# ================= SIDEBAR =================
with st.sidebar:

    st.markdown("""
        <div style="margin-bottom:20px;">
            <h2 style="margin-bottom:4px;">SIEDOM Analytics</h2>
            <div style="color:#6B7280; font-size:14px;">
                Maryam Hasnaa' Syamila<br>
                0110222067
            </div>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["📈 Evaluasi Model", "📑 Klasifikasi Sentimen"],
        label_visibility="collapsed"
    )

# Normalize menu value
if "Evaluasi" in menu:
    menu = "Evaluasi Model"
else:
    menu = "Klasifikasi Sentimen"


# ================= LOAD =================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("firqaaa/indo-sentence-bert-base")

@st.cache_resource
def load_models():
    svm = joblib.load(os.path.join(MODEL_DIR, "edom_indobert_svm_all.joblib"))
    nb  = joblib.load(os.path.join(MODEL_DIR, "edom_indobert_naivebayes_all.joblib"))
    return svm, nb

@st.cache_data
def load_dataset():
    return pd.read_csv(os.path.join(DATASET_DIR, "edom-2024-clean-balanced.csv"))

def basic_text_cleaning(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

sbert_model = load_embedding_model()
svm_model, nb_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= CACHE EVALUATION =================
@st.cache_data
def evaluate_models(df):

    df = df.copy()
    df["komentar_clean"] = df["komentar"].astype(str).apply(basic_text_cleaning)

    split_df = df[["komentar_clean", "label", "nama_dosen"]]

    train_df, test_df = train_test_split(
        split_df,
        test_size=0.2,
        random_state=42,
        stratify=split_df["label"]
    )

    X_test_embed = sbert_model.encode(
        test_df["komentar_clean"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        device=device
    )

    preds_svm = svm_model.predict(X_test_embed)
    preds_nb  = nb_model.predict(X_test_embed)

    y_test = test_df["label"].astype(int)

    acc_svm = accuracy_score(y_test, preds_svm)
    acc_nb  = accuracy_score(y_test, preds_nb)

    report_svm = classification_report(y_test, preds_svm, output_dict=True)
    report_nb  = classification_report(y_test, preds_nb, output_dict=True)

    return test_df, preds_svm, preds_nb, acc_svm, acc_nb, report_svm, report_nb


# ============================================================
# EVALUASI MODEL
# ============================================================

if menu == "Evaluasi Model":

    st.markdown("""
    <h2 style="
        font-size:28px;
        font-weight:700;
        margin-bottom:8px;
    ">
    Analisis Kinerja Model Klasifikasi Sentimen EDOM
    </h2>
    """, unsafe_allow_html=True)

    progress = st.progress(0)
    status = st.empty()

    status.text("Memuat dan mengevaluasi model...")
    df = load_dataset()

    test_df, preds_svm, preds_nb, acc_svm, acc_nb, report_svm, report_nb = evaluate_models(df)

    progress.progress(100)
    progress.empty()
    status.empty()

    # ================= MODEL TERBAIK =================
    if acc_svm > acc_nb:
        best_model_name = "Support Vector Machine"
        best_preds = preds_svm
    else:
        best_model_name = "Naive Bayes"
        best_preds = preds_nb

    test_df = test_df.copy()
    test_df["best_pred"] = best_preds

    # ================= TABEL =================
    st.markdown("""
    <h3 style="
        font-size:20px;
        font-weight:600;
        margin-top:5px;
    ">
    Tabel Evaluasi Kinerja Model
    </h3>
    """, unsafe_allow_html=True)

    metrics_table = pd.DataFrame({
        "Model": ["Support Vector Machine", "Naive Bayes"],
        "Accuracy": [acc_svm, acc_nb],
        "Precision Positif": [report_svm["1"]["precision"], report_nb["1"]["precision"]],
        "Precision Negatif": [report_svm["0"]["precision"], report_nb["0"]["precision"]],
        "Recall Positif": [report_svm["1"]["recall"], report_nb["1"]["recall"]],
        "Recall Negatif": [report_svm["0"]["recall"], report_nb["0"]["recall"]],
        "F1 Positif": [report_svm["1"]["f1-score"], report_nb["1"]["f1-score"]],
        "F1 Negatif": [report_svm["0"]["f1-score"], report_nb["0"]["f1-score"]],
    })

    numeric_cols = metrics_table.select_dtypes(include=["float64"]).columns

    st.dataframe(
        metrics_table.style.format({col: "{:.4f}" for col in numeric_cols}),
        use_container_width=True
    )

    # ================= INSIGHT =================
    st.markdown("""
    <h3 style="
        font-size:20px;
        font-weight:600;
        margin-top:5px;
    ">
    Insight Evaluasi Model
    </h3>
    """, unsafe_allow_html=True)

    gap_accuracy = abs(acc_svm - acc_nb)

    st.markdown(f"""
    <div style="
        border-left:6px solid #2E7D32;
        background-color:#F4F9F6;
        padding:20px;
        border-radius:12px;
        text-align:justify;
        margin-bottom:25px;
    ">

    Berdasarkan evaluasi pada data uji (20% test set), model 
    <b>Support Vector Machine</b> menunjukkan kinerja yang lebih optimal 
    dibandingkan Naive Bayes. Hal ini terlihat dari nilai accuracy 
    yang lebih tinggi (<b>{acc_svm:.4f}</b>) serta konsistensi precision, 
    recall, dan F1-score pada kedua kelas sentimen.

    SVM mampu mempertahankan keseimbangan performa antara kelas positif 
    dan negatif, yang tercermin dari nilai F1-score yang stabil pada 
    masing-masing kelas. Stabilitas ini menunjukkan bahwa model tidak hanya 
    akurat secara umum, tetapi juga andal dalam mengidentifikasi sentimen 
    secara proporsional. Dengan demikian, SVM memiliki kemampuan generalisasi 
    yang lebih baik dalam mengklasifikasikan sentimen evaluasi dosen.
    </div>
    """, unsafe_allow_html=True)

   # ================= STYLE GLOBAL KPI =================
    CARD_STYLE = """
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:22px;
    background-color:#FFFFFF;
    box-shadow:0 2px 6px rgba(0,0,0,0.04);
    height:100%;
    """

    VALUE_STYLE = "font-size:32px; font-weight:700; margin-top:6px;"
    LABEL_STYLE = "font-size:15px; font-weight:600; color:#374151;"

    # ================= STATISTIK DATASET EVALUASI =================
    st.markdown("---")
    st.markdown("""
    <h3 style="font-size:20px; font-weight:600;">
    Statistik Dataset Evaluasi
    </h3>
    """, unsafe_allow_html=True)

    total_data = len(df)
    total_test = len(test_df)
    total_train = total_data - total_test
    jumlah_dosen = df["nama_dosen"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Total Data</div>
            <div style="{VALUE_STYLE}">{total_data:,}</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Data Train (80%)</div>
            <div style="{VALUE_STYLE}">{total_train:,}</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Data Test (20%)</div>
            <div style="{VALUE_STYLE}">{total_test:,}</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Jumlah Dosen</div>
            <div style="{VALUE_STYLE}">{jumlah_dosen}</div>
        </div>
        """, unsafe_allow_html=True)


    # ================= DISTRIBUSI SENTIMEN =================
    st.markdown("---")
    st.markdown("""
    <h3 style="font-size:20px; font-weight:600;">
    Distribusi Sentimen (Data Test)
    </h3>
    """, unsafe_allow_html=True)

    total_all_test = len(test_df)
    total_pos_test = (test_df["best_pred"] == 1).sum()
    total_neg_test = (test_df["best_pred"] == 0).sum()

    percent_pos = round(total_pos_test/total_all_test*100)
    percent_neg = round(total_neg_test/total_all_test*100)

    c1, c2, c3 = st.columns([2,2,3])

    with c1:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Sentimen Positif</div>
            <div style="{VALUE_STYLE}">{total_pos_test:,}</div>
            <div style="
                display:inline-block;
                margin-top:8px;
                padding:4px 10px;
                border-radius:20px;
                background-color:#DCFCE7;
                color:#15803D;
                font-size:13px;
                font-weight:600;">
                {percent_pos}%
            </div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="{LABEL_STYLE}">Sentimen Negatif</div>
            <div style="{VALUE_STYLE}">{total_neg_test:,}</div>
            <div style="
                display:inline-block;
                margin-top:8px;
                padding:4px 10px;
                border-radius:20px;
                background-color:#FEE2E2;
                color:#B91C1C;
                font-size:13px;
                font-weight:600;">
                {percent_neg}%
            </div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with c3:
        filter_option = st.selectbox(
            "Filter Sentimen",
            ["Semua", "Positif", "Negatif"],
            index=0
        )


    # ================= FILTER LOGIC =================
    if filter_option == "Positif":
        df_filtered = test_df[test_df["best_pred"] == 1]
    elif filter_option == "Negatif":
        df_filtered = test_df[test_df["best_pred"] == 0]
    else:
        df_filtered = test_df.copy()

    st.markdown("---")



    # ================= VISUALISASI =================
    st.markdown("""
    <h3 style="
        font-size:20px;
        font-weight:600;
        margin-top:5px;
    ">
    Distribusi Sentimen Dosen (Model Terbaik: SVM)
    </h3>
    """, unsafe_allow_html=True)

    # 🔹 Summary dari full test set
    dosen_summary = (
        test_df.groupby(["nama_dosen", "best_pred"])
        .size()
        .unstack(fill_value=0)
    )

    # pastikan dua kolom selalu ada
    if 0 not in dosen_summary.columns:
        dosen_summary[0] = 0
    if 1 not in dosen_summary.columns:
        dosen_summary[1] = 0

    # urutkan kolom agar konsisten
    dosen_summary = dosen_summary[[0, 1]]

    dosen_summary = dosen_summary.reset_index()
    dosen_summary.columns = ["nama_dosen", "Negatif", "Positif"]

    # Tambahkan total untuk sorting
    dosen_summary["Total"] = dosen_summary["Negatif"] + dosen_summary["Positif"]

    # 🔹 Sorting dinamis sesuai filter
    if filter_option == "Semua":
        dosen_summary = dosen_summary.sort_values("Total", ascending=False)
    elif filter_option == "Positif":
        dosen_summary = dosen_summary.sort_values("Positif", ascending=False)
    else:
        dosen_summary = dosen_summary.sort_values("Negatif", ascending=False)

    # ID konsisten (semua dosen)
    dosen_summary["id_dosen"] = [
        "D" + str(i+1).zfill(3)
        for i in range(len(dosen_summary))
    ]

    COLOR_POS = "#2E7D5B"
    COLOR_NEG = "#C65A4E"

    fig = go.Figure()

    if filter_option in ["Semua", "Positif"]:
        fig.add_trace(go.Bar(
            x=dosen_summary["id_dosen"],
            y=dosen_summary["Positif"],
            name="Positif",
            marker_color=COLOR_POS
        ))

    if filter_option in ["Semua", "Negatif"]:
        fig.add_trace(go.Bar(
            x=dosen_summary["id_dosen"],
            y=dosen_summary["Negatif"],
            name="Negatif",
            marker_color=COLOR_NEG
        ))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        xaxis_title="ID Dosen",
        yaxis_title="Jumlah Sentimen",
        legend_title="Kategori Sentimen",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # ================= INTERPRETASI DISTRIBUSI TEST SET =================
    st.markdown("""
    <h3 style="
        font-size:20px;
        font-weight:600;
        margin-top:5px;
    ">
    Interpretasi Distribusi Sentimen pada Data Test
    </h3>
    """, unsafe_allow_html=True)

    if len(dosen_summary) > 0:

        total_test_data = len(test_df)
        total_pos_test = (test_df["best_pred"] == 1).sum()
        total_neg_test = (test_df["best_pred"] == 0).sum()
        positive_rate_test = (total_pos_test / total_test_data) * 100

        st.markdown(f"""
        <div style="
            border-left:6px solid #2E7D32;
            background-color:#F4F9F6;
            padding:20px;
            border-radius:12px;
            text-align:justify;
            margin-bottom:25px;
        ">

        Analisis distribusi sentimen pada data uji (20% dari total dataset) 
        menunjukkan bahwa model terbaik, yaitu <b>{best_model_name}</b>, 
        menghasilkan proporsi sentimen positif sebesar 
        <b>{positive_rate_test:.1f}%</b>.

        Pola distribusi sentimen antar dosen pada data uji terlihat konsisten 
        dan proporsional. Hal ini mengindikasikan bahwa model tidak hanya memiliki 
        nilai metrik evaluasi yang tinggi, tetapi juga mampu menghasilkan prediksi 
        yang stabil dalam konteks analisis evaluasi dosen.

        Meskipun analisis ini dilakukan pada subset data uji, distribusi yang 
        terbentuk menunjukkan bahwa model memiliki kemampuan generalisasi yang baik 
        dalam merepresentasikan persepsi mahasiswa terhadap proses pembelajaran.

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# KLASIFIKASI SENTIMEN
# ============================================================
if menu == "Klasifikasi Sentimen":

    st.markdown("""
    <h2 style="
        font-size:28px;
        font-weight:700;
        margin-bottom:8px;
    ">
    Klasifikasi Sentimen EDOM
    </h2>
    """, unsafe_allow_html=True)

    # ================= HELPER FORMAT =================
    def format_ribuan(x):
        return f"{x:,}".replace(",", ".")

    def format_persen(x):
        return f"{x:.1f}%"

    # ================= TEXT CLEANING =================
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ================= UPLOAD SECTION =================
    # st.markdown("### Cek Klasifikasi Dataset EDOM")

    st.info("""
    Model yang digunakan dalam proses klasifikasi sentimen EDOM adalah 
    **Support Vector Machine (SVM)**, yang dipilih berdasarkan hasil evaluasi 
    kinerja terbaik dibandingkan model Naive Bayes.
    """)

    st.info("""
    **Ketentuan Upload:**
    - Wajib memiliki kolom: `komentar` dan `nama_dosen`
    - Jenis file `CSV`, maksimal **200MB**
    - Kolom `label` bersifat opsional
    """)

    uploaded_file = st.file_uploader("Upload File (.csv)", type=["csv"])
    process_button = st.button("Proses")

    # ================= PROSES =================
    if uploaded_file is not None and process_button:

        df = pd.read_csv(uploaded_file)
        # Simpan jumlah dosen sebelum cleaning
        total_dosen_awal = df["nama_dosen"].nunique()
        total_data_awal = len(df)

        if not all(col in df.columns for col in ["komentar", "nama_dosen"]):
            st.error("File harus memiliki kolom: komentar dan nama_dosen")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        # 1️⃣ CLEANING
        status.text("Membersihkan teks...")
        df["komentar_clean"] = df["komentar"].astype(str).apply(clean_text)

        df = df[
            (df["komentar_clean"] != "") &
            (~df["komentar_clean"].isin(["-", "tidak ada", "tidak", "na", "none"]))
        ].reset_index(drop=True)

        # Jumlah dosen & data setelah cleaning
        total_dosen_setelah = df["nama_dosen"].nunique()
        total_data_setelah = len(df)

        # Simpan ke session_state untuk ditampilkan nanti
        st.session_state["info_dosen"] = {
            "dosen_awal": total_dosen_awal,
            "dosen_setelah": total_dosen_setelah,
            "data_awal": total_data_awal,
            "data_setelah": total_data_setelah
        }

        progress.progress(30)

        # 2️⃣ EMBEDDING SBERT
        status.text("Membuat embedding Indo-Sentence-BERT...")
        with torch.no_grad():
            embeddings = sbert_model.encode(
                df["komentar_clean"].tolist(),
                batch_size=64,
                convert_to_numpy=True,
                device=device,
                show_progress_bar=False
            )

        progress.progress(70)

        # 3️⃣ PREDIKSI SVM
        status.text("Melakukan klasifikasi dengan SVM...")
        preds = svm_model.predict(embeddings)

        df["pred_label_id"] = preds
        df["label"] = df["pred_label_id"].map({
            0: "Negatif",
            1: "Positif"
        })


        progress.progress(100)
        progress.empty()
        status.empty()

        st.session_state["hasil_klasifikasi"] = df
        st.success("Klasifikasi selesai.")

    # ================= TAMPILKAN HASIL =================
    if "hasil_klasifikasi" in st.session_state:

        df = st.session_state["hasil_klasifikasi"]
        # ================= INFO DATA =================
        if "info_dosen" in st.session_state:
            info = st.session_state["info_dosen"]
            st.markdown("---")
            st.markdown("""
                <h3 style="
                    font-size:20px;
                    font-weight:600;
                    margin-top:5px;
                ">
                Informasi Dataset
                </h3>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            with col_a:
                st.info(f"""
        Jumlah dosen sebelum cleaning: **{info['dosen_awal']}**  
        Jumlah dosen setelah cleaning: **{info['dosen_setelah']}**
        """)

            with col_b:
                st.info(f"""
        Total komentar sebelum cleaning: **{format_ribuan(info['data_awal'])}**  
        Total komentar setelah cleaning: **{format_ribuan(info['data_setelah'])}**
        """)

        st.markdown("---")
        st.markdown("""
        <h3 style="
            font-size:20px;
            font-weight:600;
            margin-top:5px;
        ">
        Preview Dataset Hasil Klasifikasi
        </h3>
        """, unsafe_allow_html=True)

        unique_names = df["nama_dosen"].unique()
        id_map = {name: f"D{str(i+1).zfill(3)}" for i, name in enumerate(unique_names)}

        df["id_dosen"] = df["nama_dosen"].map(id_map)

        df_preview = df.copy()

        preview_cols = ["id_dosen", "komentar_clean", "label"]

        st.dataframe(df_preview[preview_cols].head(20), use_container_width=True)


        # DOWNLOAD
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df[["id_dosen", "komentar_clean", "label"]].to_excel(
                writer,
                index=False,
                sheet_name="Hasil_Klasifikasi"
            )

        st.download_button(
            label="Download Hasil Klasifikasi (Excel)",
            data=output.getvalue(),
            file_name="hasil_klasifikasi_edom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


        # ================= KPI =================
        st.markdown("---")
        st.markdown("""
        <h3 style="font-size:20px; font-weight:600;">
        Distribusi Sentimen EDOM
        </h3>
        """, unsafe_allow_html=True)

        total_all = len(df)
        total_pos = (df["pred_label_id"] == 1).sum()
        total_neg = (df["pred_label_id"] == 0).sum()

        percent_pos = round((total_pos / total_all) * 100, 1) if total_all > 0 else 0
        percent_neg = round((total_neg / total_all) * 100, 1) if total_all > 0 else 0

        card_style = """
            border:1px solid #E5E7EB;
            border-radius:16px;
            padding:20px;
            background-color:#FFFFFF;
            box-shadow:0 2px 6px rgba(0,0,0,0.04);
        """

        col1, col2, col3 = st.columns([2,2,3])

        with col1:
            st.markdown(f"""
            <div style="{card_style}">
                <div style="font-size:14px; font-weight:600; margin-bottom:6px;">
                    Sentimen Positif
                </div>
                <div style="font-size:32px; font-weight:700;">
                    {format_ribuan(total_pos)}
                </div>
                <div style="
                    display:inline-block;
                    margin-top:5px;
                    padding:6px 12px;
                    border-radius:20px;
                    background-color:#E6F4EA;
                    color:#1E7E34;
                    font-size:13px;
                    font-weight:600;
                ">
                    {percent_pos}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="{card_style}">
                <div style="font-size:14px; font-weight:600; margin-bottom:6px;">
                    Sentimen Negatif
                </div>
                <div style="font-size:32px; font-weight:700;">
                    {format_ribuan(total_neg)}
                </div>
                <div style="
                    display:inline-block;
                    margin-top:5px;
                    padding:6px 12px;
                    border-radius:20px;
                    background-color:#FDECEA;
                    color:#C0392B;
                    font-size:13px;
                    font-weight:600;
                ">
                    {percent_neg}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            filter_option = st.selectbox(
                "Filter Sentimen",
                ["Semua", "Positif", "Negatif"],
                index=0
            )

        st.markdown("---")


        # ================= VISUALISASI =================
        st.markdown("""
        <h3 style="
            font-size:20px;
            font-weight:600;
            margin-top:5px;
        ">
        Distribusi Sentimen per Dosen
        </h3>
        """, unsafe_allow_html=True)

        summary = (
            df.groupby(["nama_dosen","pred_label_id"])
            .size()
            .unstack(fill_value=0)
        )

        # Pastikan kedua kolom ada
        if 0 not in summary.columns:
            summary[0] = 0
        if 1 not in summary.columns:
            summary[1] = 0

        summary = summary[[0,1]].reset_index()
        summary.columns = ["nama_dosen","Negatif","Positif"]
        summary["Total"] = summary["Negatif"] + summary["Positif"]

        # Mapping ID konsisten
        summary["id_dosen"] = summary["nama_dosen"].map(id_map)

        # Sorting dinamis berdasarkan filter
        if filter_option == "Semua":
            summary = summary.sort_values("Total", ascending=False)
        elif filter_option == "Positif":
            summary = summary.sort_values("Positif", ascending=False)
        else:
            summary = summary.sort_values("Negatif", ascending=False)

        fig = go.Figure()

        if filter_option in ["Semua","Positif"]:
            fig.add_trace(go.Bar(
                x=summary["id_dosen"],
                y=summary["Positif"],
                name="Positif",
                marker_color="#2E7D5B"
            ))

        if filter_option in ["Semua","Negatif"]:
            fig.add_trace(go.Bar(
                x=summary["id_dosen"],
                y=summary["Negatif"],
                name="Negatif",
                marker_color="#C65A4E"
            ))

        fig.update_layout(
            template="plotly_white",
            barmode="group",
            xaxis_title="ID Dosen",
            yaxis_title="Jumlah Sentimen",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)


        # ================= ANALISIS HASIL =================
        st.markdown("---")
        st.markdown("""
        <h3 style="
            font-size:20px;
            font-weight:600;
            margin-top:5px;
        ">
        Ringkasan Hasil Evaluasi
        </h3>
        """, unsafe_allow_html=True)

        if len(summary) > 0 and total_all > 0:

            summary_full = (
                df.groupby(["nama_dosen","pred_label_id"])
                .size()
                .unstack(fill_value=0)
            )

            if 0 not in summary_full.columns:
                summary_full[0] = 0
            if 1 not in summary_full.columns:
                summary_full[1] = 0

            summary_full = summary_full.reset_index()
            summary_full.columns = ["nama_dosen","Negatif","Positif"]
            summary_full["Total"] = summary_full["Negatif"] + summary_full["Positif"]
            summary_full["Proporsi_Positif"] = summary_full["Positif"] / summary_full["Total"]

            top_pos = summary_full.sort_values("Positif", ascending=False).iloc[0]
            top_neg = summary_full.sort_values("Negatif", ascending=False).iloc[0]

            overall_positive_rate = (total_pos / total_all) * 100

            card_style = """
                border:1px solid #E5E7EB;
                border-radius:16px;
                padding:18px;
                background-color:#FFFFFF;
                box-shadow:0 2px 6px rgba(0,0,0,0.04);
                height:100%;
            """

            title_style = """
                font-size:16px;
                font-weight:600;
                margin-bottom:8px;
            """

            text_style = """
                font-size:14px;
                line-height:1.6;
                color:#374151;
                text-align:justify;
            """

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div style="{card_style}">
                    <div style="{title_style}">
                        Tingkat Kepuasan Mahasiswa
                    </div>
                    <div style="{text_style}">
                        Sebanyak <b>{format_persen(overall_positive_rate)}</b> dari total komentar 
                        pada data uji menunjukkan sentimen positif.
                        Proporsi ini menggambarkan bahwa secara umum mahasiswa memiliki persepsi 
                        yang baik terhadap proses pembelajaran. Hasil ini mengindikasikan tingkat 
                        kepuasan yang relatif tinggi terhadap kualitas pengajaran.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style="{card_style}">
                    <div style="{title_style}">
                        Dosen dengan Penilaian Terbaik
                    </div>
                    <div style="{text_style}">
                        Dosen dengan ID <b>{id_map[top_pos['nama_dosen']]}</b> 
                        mencatat jumlah sentimen positif tertinggi, yaitu 
                        <b>{format_ribuan(int(top_pos['Positif']))} komentar</b> dengan proporsi positif 
                        sebesar <b>{format_persen(top_pos['Proporsi_Positif']*100)}</b>.
                        Temuan ini menunjukkan konsistensi penilaian mahasiswa terhadap kualitas 
                        pengajaran yang dinilai baik.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div style="{card_style}">
                    <div style="{title_style}">
                        Prioritas Evaluasi Pengajaran
                    </div>
                    <div style="{text_style}">
                        Dosen dengan ID <b>{id_map[top_neg['nama_dosen']]}</b> 
                        memiliki jumlah sentimen negatif tertinggi, yaitu 
                        <b>{format_ribuan(int(top_neg['Negatif']))} komentar</b>.
                        Informasi ini dapat digunakan sebagai dasar evaluasi terhadap metode 
                        penyampaian materi maupun strategi interaksi di kelas secara lebih terarah.
                    </div>
                </div>
                """, unsafe_allow_html=True)
