from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from imutils import contours
import argparse
import imutils
import cv2
import os
from sklearn.cluster import KMeans
import random as rng
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from fungsi_rekomen import rekomendasikan_sepatu

app = Flask(__name__)

def preprocess(img):
    # Kode preprocess
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img/255

    return img

def edgeDetection(clusteredImage):
    # Kode edgeDetection
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    edged = cv2.dilate(edged1, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def cropOrig(bRect, oimg):
    # Kode cropOrig
    x,y,w,h = bRect
    
    print(x,y,w,h)
    pcropedImg = oimg[y:y+h,x:x+w]

    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]

    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]

    croppedImg = oimg[iy:iy+ih,ix:ix+iw]

    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    # Kode overlayImage
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)

    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg

    return new_image

def getBoundingBox(img):
    # Kode getBoundingBox
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):
    # Kode drawCnt
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)   

    paperbb = bRect

    for i in range(len(contours)):
      color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
      cv2.drawContours(drawing, cntPoly, i, color)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
              (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    
    return drawing

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Ambil input gender dari form
    gender = request.form['gender']

    # Cek apakah file foto telah diunggah
    if 'photo' not in request.files:
        return 'File foto tidak ditemukan'

    photo = request.files['photo']

    # Cek apakah file foto memiliki nama
    if photo.filename == '':
        return 'Nama file foto kosong'

    # Simpan file foto di direktori 'temp'
    filename = secure_filename(photo.filename)
    photo_path = os.path.join('static\\images\\uploads', filename)
    photo.save(photo_path)

    # Baca foto menggunakan OpenCV
    oimg = cv2.imread(photo_path)
    preprocessedOimg = preprocess(oimg)
    
    image_2D = preprocessedOimg.reshape(preprocessedOimg.shape[0]*preprocessedOimg.shape[1], preprocessedOimg.shape[2])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(preprocessedOimg.shape[0], preprocessedOimg.shape[1], preprocessedOimg.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    
    edgedImg = edgeDetection(clusteredImg)

    boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
    pdraw = drawCnt(boundRect[1], contours, contours_poly, img)

    croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)

    newImg = overlayImage(croppedImg, pcropedImg)
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)

    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg

    fedged = edgeDetection(new_image)

    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
    fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)

    fh = y2 + fboundRect[2][3]
    fw = x2 + fboundRect[2][2]
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]

    opw = 21
    oph = 29.7

    ofs = 0.0

    if fw>fh:    
        ofs = (oph/pw)*fw
    else :
        ofs = (oph/ph)*fh


    # Tampilkan hasil ukuran kaki
    size = ""
    length = ofs

    if gender.lower() == "wanita":
        if 21.35 <= ofs <= 22.02:
            size = "36"
        elif 22.02 <= ofs <= 22.69:
            size = "37"
        elif 22.69 <= ofs <= 23.36:
            size = "38"
        elif 23.36 <= ofs <= 24.03:
            size = "39"
        elif 24.03 <= ofs <= 24.70:
            size = "40"
        elif 24.70 <= ofs <= 25.37:
            size = "41"
        elif 25.37 <= ofs <= 26.04:
            size = "42"
        elif 26.04 <= ofs <= 26.71:
            size = "43"
        elif 26.71 <= ofs <= 27.38:
            size = "44"
        # Kode lainnya untuk menghitung ukuran kaki wanita

    elif gender.lower() == "pria":
        if 21.35 <= ofs <= 22.02:
            size = "37"
        elif 22.02 <= ofs <= 22.69:
            size = "38"
        elif 22.69 <= ofs <= 23.36:
            size = "39"
        elif 23.36 <= ofs <= 24.03:
            size = "40"
        elif 24.03 <= ofs <= 24.70:
            size = "41"
        elif 24.70 <= ofs <= 25.37:
            size = "42"
        elif 25.37 <= ofs <= 26.04:
            size = "43"
        elif 26.04 <= ofs <= 26.71:
            size = "44"
        elif 26.71 <= ofs <= 27.38:
            size = "45"
        # Kode lainnya untuk menghitung ukuran kaki pria
    formatted_length = round(length, 2)
    return render_template('hasil.html', length=formatted_length, size=size)
    #return render_template('index.html', hasil_pengukuran='Hasil Pengukuran', content_hasil='Panjang Kaki : ' + str(length) + ', Ukuran Sepatu : ' + str(size) )
    #return f"Panjang kaki Anda adalah {length}. Berdasarkan panjang kaki anda, anda direkomendasikan mencari sepatu dengan ukuran {size}"


@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    # Ambil nilai dari form input
    styles = request.form['styles']
    colors = request.form['colors']
    prices = int(request.form['prices'])

    # Load model dan data yang telah disimpan
    with open('static/rekomendasi_sepatu_model.pkl', 'rb') as file:
        tfidf = pickle.load(file)
        df = pickle.load(file)
        similarity_matrix = pickle.load(file)

    # Panggil fungsi rekomendasikan_sepatu dengan input dari form
    rekomendasi = rekomendasikan_sepatu(styles, colors, prices, df, similarity_matrix, n=5)

    sepatu_unik = {}
    for merek in rekomendasi:
        sepatu_unik[merek] = True

    rekomendasi_sepatu = sepatu_unik.keys()

    # Render template hasil.html dengan merek-merek sepatu rekomendasi
    return render_template('hasil2.html', rekomendasi=rekomendasi_sepatu)


if __name__ == '__main__':
    app.run(debug=True)
