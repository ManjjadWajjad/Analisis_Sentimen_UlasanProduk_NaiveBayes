# Sentiment Analysis App with File Upload and Manual Input (Streamlit)
import streamlit as st
import pandas as pd
import re
import emoji
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')

# Preprocessing tools
factory = StemmerFactory()
stemmer = factory.create_stemmer()
slang_dict = {
    'yg': 'yang', 'tdk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak',
    'bgt': 'banget', 'bangettt': 'banget', 'bgs': 'bagus', 'bkn': 'bukan',
    'ok': 'oke', 'okei': 'oke', 'okelah': 'oke', 'recomended': 'recommended',
    'packing': 'kemas', 'pengiriman': 'kirim', 'seller': 'penjual',
    'brg': 'barang', 'dtg': 'datang', 'cepet': 'cepat', 'lgsg': 'langsung',
    'dgn': 'dengan', 'utk': 'untuk', 'udah': 'sudah', 'sy': 'saya',
    'jg': 'juga', 'aja': 'saja', 'sdh': 'sudah', 'kurg': 'kurang', 'tp': 'tapi'
}
negation_words = {'tidak', 'bukan', 'jangan', 'kurang', 'belum'}
stop_words = set(stopwords.words('indonesian'))
custom_stop_words = stop_words - negation_words

positive_words = [
    "bagus", "mantap", "mantep", "puas", "cepat", "baik", "recommended", "top", "keren",
    "memuaskan", "lancar", "hebat", "rapi", "rekomen", "worth", "gercep", "aman", "senang", "nyaman",
    "berfungsi", "elegan", "estetik", "modern", "awet", "terpercaya", "rekomendasi", "stabil", "mantul",
    "terjangkau", "praktis", "terbaik", "halus", "simple", "powerful"
]

negative_words = [
    "jelek", "buruk", "tidak puas", "mengecewakan", "parah", "rusak", "tidak sesuai", "cacat",
    "lambat", "mahal", "ribet", "hilang", "lelet", "tidak ori", "habis", "mati", "tidak berguna",
    "tidak berfungsi", "patah", "tidak awet", "kurang responsif", "berisik", "gagal", "menyebalkan"
]

# Refined lexicon-based labeling logic without 'netral'
def label_sentiment(text):
    words = text.split()
    pos_count = 0
    neg_count = 0
    for i, word in enumerate(words):
        if word in positive_words:
            if i > 0 and words[i-1] in negation_words:
                neg_count += 1
            else:
                pos_count += 1
        elif word in negative_words:
            if i > 0 and words[i-1] in negation_words:
                pos_count += 1
            else:
                neg_count += 1
    return 'positif' if pos_count >= neg_count else 'negatif'

# Clean function
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]+|\#', '', text)
    text = emoji.replace_emoji(text, replace='')
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stop_words]
    text = ' '.join(filtered_words)
    return stemmer.stem(text)

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("multinomialnb_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model()

# App title
st.title("📊 Aplikasi Analisis Sentimen Ulasan Produk - Naive Bayes")
st.write("Upload file CSV (dengan kolom 'ulasan' dan opsional 'sentimen') atau masukkan ulasan secara manual untuk melihat prediksi sentimennya.")

# Tabs for input options
tab1, tab2 = st.tabs(["📂 Upload File", "📝 Input Manual"])

# --- TAB 1: Upload File --- #
with tab1:
    uploaded_file = st.file_uploader("Unggah file CSV (harus ada kolom: 'ulasan')", type=['csv'])
    use_auto_label = st.checkbox("Gunakan label otomatis (Lexicon-Based) jika kolom 'sentimen' tidak tersedia", value=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'ulasan' not in df.columns:
            st.error("❌ Kolom 'ulasan' tidak ditemukan di file.")
        else:
            with st.spinner("🔄 Memproses ulasan, harap tunggu..."):
                start = time.time()
                df['clean_review'] = df['ulasan'].apply(clean_text)
                if 'sentimen' not in df.columns and use_auto_label:
                    df['sentimen'] = df['clean_review'].apply(label_sentiment)
                X = vectorizer.transform(df['clean_review'])
                df['prediksi'] = model.predict(X)
                df['label_lexicon'] = df['clean_review'].apply(label_sentiment)
                end = time.time()

            st.success(f"✅ Proses selesai dalam {end - start:.2f} detik")
            st.subheader("📋 Contoh Hasil Prediksi (5 Data Pertama)")
            display_cols = ['ulasan', 'clean_review', 'label_lexicon', 'prediksi']
            if 'sentimen' in df.columns:
                display_cols.insert(3, 'sentimen')
            def color_pred(val):
                return f'color: {"green" if val == "positif" else "red"}'
            st.dataframe(df[display_cols].head(5).style.applymap(color_pred, subset=['prediksi']))

            if 'sentimen' in df.columns:
                y_true = df['sentimen']
                y_pred = df['prediksi']

                acc = accuracy_score(y_true, y_pred)
                st.markdown(f"**🎯 Akurasi Model:** {acc*100:.2f}%")

                st.subheader("📈 Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.loc[['negatif', 'positif', 'accuracy', 'macro avg', 'weighted avg']])

                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=['positif', 'negatif'])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu", xticklabels=['positif', 'negatif'], yticklabels=['positif', 'negatif'], ax=ax)
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                st.subheader("📊 Distribusi Sentimen Prediksi")
                sentiment_counts = df['prediksi'].value_counts().sort_index()
                colors = ["green" if sentiment == "positif" else "red" for sentiment in sentiment_counts.index]
                fig3, ax3 = plt.subplots()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax3)
                ax3.set_xlabel("Sentimen")
                ax3.set_ylabel("Jumlah")
                ax3.set_title("Distribusi Sentimen")
                st.pyplot(fig3)

                df_errors = df[df['sentimen'] != df['prediksi']]
                st.subheader("🔍 Contoh Data yang Salah Klasifikasi")
                st.dataframe(df_errors[['ulasan', 'clean_review', 'sentimen', 'prediksi']].head(10))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh Data Hasil (CSV)", data=csv, file_name="hasil_sentimen.csv", mime="text/csv")

# --- TAB 2: Input Manual --- #
with tab2:
    user_input = st.text_area("Masukkan ulasan produk Anda:")
    if st.button("🔍 Prediksi Sentimen", key='manual'):
        if user_input.strip():
            clean_input = clean_text(user_input)
            X_input = vectorizer.transform([clean_input])
            prediction = model.predict(X_input)[0]
            st.markdown(f"**Ulasan Setelah Dibersihkan:** `{clean_input}`")
            if prediction == 'positif':
                st.markdown(f"<span style='color:green; font-size:20px;'>🎉 Sentimen: <b>{prediction.upper()}</b></span>", unsafe_allow_html=True)
            elif prediction == 'negatif':
                st.markdown(f"<span style='color:red; font-size:20px;'>👎 Sentimen: <b>{prediction.upper()}</b></span>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu.")
