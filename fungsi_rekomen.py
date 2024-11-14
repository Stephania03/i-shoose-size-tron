import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


df = pd.read_csv('static\project3.csv')

df['Prices'] = df['Prices'].str.replace('\D', '', regex=True)
df['Prices'] = df['Prices'].astype(int)
mapping = {0: "Tidak", 1: "Ya"}
# Menggunakan fungsi map() untuk mengubah nilai pada kolom "warna"
df['Available_in_Multiple_Colors'] = df['Available_in_Multiple_Colors'].map(mapping)

df_filter = df[['Brands', 'Styles', 'Prices', 'Available_in_Multiple_Colors']]

df_filter = df_filter.dropna(axis='rows')
df_filter = df_filter.fillna('unknown')

df_filter.to_csv('static/filtered_data.csv')

df['Deskripsi'] = df['Brands'] + ' - ' + df['Styles'] + ' - ' + df['Available_in_Multiple_Colors'] + ' - ' + df['Prices'].astype(str)

tfidf = TfidfVectorizer()

# Melakukan fitting dan transformasi pada kolom Deskripsi
tfidf_matrix = tfidf.fit_transform(df['Deskripsi'])

# Menghitung similarity matrix berdasarkan TF-IDF
similarity_matrix = cosine_similarity(tfidf_matrix)

def rekomendasikan_sepatu(Styles, Available_in_Multiple_Colors, Prices, df, similarity_matrix, n=5):
    # Menggabungkan fitur-fitur sepatu menjadi satu kolom teks
    deskripsi_sepatu = Styles + ' - ' + Available_in_Multiple_Colors + ' - ' + str(Prices)

    # Melakukan transformasi pada deskripsi sepatu menggunakan TF-IDF vectorizer yang sudah di-fit sebelumnya
    tfidf_vector = tfidf.transform([deskripsi_sepatu])

    # Menghitung similarity antara deskripsi sepatu input dengan semua sepatu dalam dataset
    similarity_scores = cosine_similarity(tfidf_vector, tfidf_matrix)

    # Mendapatkan indeks sepatu yang memiliki similarity tertinggi
    indeks_rekomendasi = similarity_scores.argsort()[0][::-1]

    # Menghapus indeks sepatu input dari daftar rekomendasi
    indeks_rekomendasi = indeks_rekomendasi[1:]

    # Mengembalikan merek-merek sepatu rekomendasi
    merek_rekomendasi = df.loc[indeks_rekomendasi][:n]['Brands'].values.tolist()

    return merek_rekomendasi