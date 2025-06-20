import streamlit as st
import pandas as pd
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle # Diperlukan untuk memuat model yang sudah disimpan

# Inisialisasi NLTK stopwords
# NLTK stopwords akan di-download saat aplikasi berjalan untuk pertama kali
# jika belum ada. Streamlit akan mengelola ini secara otomatis.
nltk.download('stopwords', quiet=True)

# Inisialisasi Sastrawi Stemmer dan daftar stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi pembersihan teks (clean_text) dari kode Anda
def clean_text(text):
    # Pastikan input adalah string dan tangani nilai NaN
    if pd.isna(text):
        return ""
    text = str(text).lower()  # ubah ke huruf kecil
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # hapus URL
    text = re.sub(r'\@[\w]+|\#', '', text)  # hapus mention dan hashtag
    text = emoji.replace_emoji(text, replace='')  # hapus emoji
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus angka dan simbol
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = ' '.join([word for word in text.split() if word not in stop_words])  # hapus stopwords
    text = stemmer.stem(text)  # stemming (ubah ke bentuk dasar)
    return text

# Leksikon kata positif dan negatif dari kode Anda
positive_words = [
    "bagus", "mantap", "mantep", "puas", "cepat", "baik", "recommended", "top", "keren",
    "memuaskan", "lancar", "hebat", "rapi", "rekomen", "worth", "gercep"
]
negative_words = [
    "jelek", "buruk", "tidak puas", "mengecewakan", "parah", "rusak",
    "tidak sesuai", "cacat", "lambat", "mahal", "ribet", "hilang", "lelet", "tidak ori"
]

# Fungsi pelabelan sentimen (label_sentiment) dari kode Anda
# Catatan: Fungsi ini tidak digunakan untuk prediksi langsung di app.py
# melainkan asumsi bahwa model sudah dilatih dengan output dari fungsi ini.
def label_sentiment(text):
    pos = sum(1 for word in str(text).split() if word in positive_words)
    neg = sum(1 for word in str(text).split() if word in negative_words)
    return 'positif' if pos > neg else 'negatif'


# Memuat Model dan Vectorizer yang Sudah Dilatih
# Menggunakan st.cache_resource agar model hanya dimuat sekali saat aplikasi dimulai
@st.cache_resource
def load_trained_resources():
    try:
        # Asumsi file .pkl berada di root repositori GitHub
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open('multinomialnb_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_vectorizer, loaded_model
    except FileNotFoundError:
        st.error("File model atau vectorizer (.pkl) tidak ditemukan. Pastikan sudah diunggah ke repositori GitHub Anda.")
        return None, None

tfidf_vectorizer, model = load_trained_resources()

# --- Antarmuka Aplikasi Streamlit ---
st.title("Aplikasi Analisis Sentimen Ulasan Produk")
st.write("Masukkan ulasan produk untuk mengetahui sentimennya (Positif/Negatif).")

# Input teks dari pengguna
user_input = st.text_area("Tulis ulasan Anda di sini:")

# Tombol untuk memicu prediksi
if st.button("Prediksi Sentimen"):
    if user_input: # Memastikan ada input dari pengguna
        if tfidf_vectorizer is not None and model is not None: # Memastikan model berhasil dimuat
            # Lakukan pembersihan teks pada input pengguna
            clean_input = clean_text(user_input) # <--- BARIS INI YANG HILANG DAN SUDAH DITAMBAHKAN!

            if not clean_input: # Menangani jika input yang sudah dibersihkan kosong
                st.warning("Ulasan yang Anda masukkan tidak mengandung kata-kata yang relevan setelah proses pembersihan. Tidak dapat memprediksi.")
            else:
                # Transformasi input menggunakan vectorizer yang sudah dilatih
                new_text_vec = tfidf_vectorizer.transform([clean_input])
                # Lakukan prediksi menggunakan model yang sudah dilatih
                predicted_sentimen = model.predict(new_text_vec)[0]

                st.write(f"**Ulasan Bersih:** {clean_input}")
                # Menampilkan hasil prediksi dengan gaya yang berbeda
                if predicted_sentimen == 'positif':
                    st.success(f"**Prediksi Sentimen:** {predicted_sentimen.upper()} 🎉")
                elif predicted_sentimen == 'negatif':
                    st.error(f"**Prediksi Sentimen:** {predicted_sentimen.upper()} 👎")
                else:
                    # Ini jarang terjadi jika model dilatih hanya dengan 2 kelas,
                    # tetapi bisa terjadi jika ada kelas tak terduga dari model.
                    st.info(f"**Prediksi Sentimen:** {predicted_sentimen.upper()} (Sentimen tidak dikenal)")
        else:
            # Pesan error jika model atau vectorizer gagal dimuat
            st.warning("Sumber daya model (TF-IDF Vectorizer atau Model Naive Bayes) gagal dimuat. Mohon periksa file .pkl Anda.")
    else:
        # Pesan peringatan jika input teks kosong
        st.warning("Mohon masukkan ulasan terlebih dahulu.")
