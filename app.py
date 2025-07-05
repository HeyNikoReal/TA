# Import Library 
from flask import Flask, request, jsonify
import tensorflow as tf
import time
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2

app = Flask(_name_)

# Load model 
model = tf.keras.models.load_model('16_50_0001.h5')

# Mapping kelas ke nama dan id karyawan
kelas_ke_info = [
    {'nama': 'Aggasi', 'id': '001'},
    {'nama': 'Andira', 'id': '002'},
    {'nama': 'Apri', 'id': '003'},
    {'nama': 'Ariko', 'id': '004'},
    {'nama': 'Benny', 'id': '005'},
    {'nama': 'Darimi', 'id': '006'},
    {'nama': 'Dermosis', 'id': '007'},
    {'nama': 'Dicky', 'id': '008'},
    {'nama': 'Doni', 'id': '009'},
    {'nama': 'Gilang', 'id': '010'},
    {'nama': 'Heppy', 'id': '011'},
    {'nama': 'Heri', 'id': '012'},
    {'nama': 'Holidy', 'id': '013'},
    {'nama': 'Irvan', 'id': '014'},
    {'nama': 'Muzartun', 'id': '015'},
    {'nama': 'Nofri', 'id': '016'},
    {'nama': 'Rangga', 'id': '017'},
    {'nama': 'Nuryanto', 'id': '018'},
    {'nama': 'Rafi', 'id': '019'},
    {'nama': 'Rangga', 'id': '020'},
    {'nama': 'Ricky', 'id': '021'},
    {'nama': 'Sandy', 'id': '022'},
    {'nama': 'Solihin', 'id': '023'},
    {'nama': 'Wahyu', 'id': '024'},
    {'nama': 'Yusuf', 'id': '025'}
]

detector = MTCNN()

# Fungsi crop wajah menggunakan MTCNN
def crop_face_mtcnn(img_pil, output_size=(299, 299)):
    img_np = np.array(img_pil)
    faces = detector.detect_faces(img_np)

    if not faces:
        return None  # Tidak ada wajah terdeteksi

    x, y, w, h = faces[0]['box']
    x1, y1 = abs(x), abs(y) # Menghindari nilai negatif
    x2, y2 = x1 + w, y1 + h
    face_crop = img_np[y1:y2, x1:x2]
    face_img = Image.fromarray(face_crop).resize(output_size)
    return face_img

@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        # start_time = time.time() 
        # Ambil file gambar dari request
        file = request.files.get('gambar')
        if not file:
            return jsonify({'status': 'gagal', 'error': 'File gambar tidak ditemukan'}), 400

        img_pil = Image.open(file.stream).convert('RGB')
        wajah = crop_face_mtcnn(img_pil)

        if wajah is None:
            return jsonify({'status': 'gagal', 'error': 'Wajah tidak terdeteksi'}), 400

        wajah_array = np.array(wajah) / 255.0
        wajah_normalized = np.expand_dims(wajah_array, axis=0)

        # Prediksi
        pred = model.predict(wajah_normalized)
        kelas = int(np.argmax(pred))
        confidence = float(np.max(pred))

        if kelas >= len(kelas_ke_info):
            return jsonify({'status': 'gagal', 'error': 'Kelas tidak dikenali'}), 400

        info = kelas_ke_info[kelas]

        # end_time = time.time() 
        # processing_time = round(end_time - start_time, 4)

        return jsonify({
            'status': 'sukses',
            'kelas': kelas,
            'nama': info['nama'],
            'id': info['id'],
            'confidence': round(confidence, 4),
            # 'processing_time': processing_time
        })

    except Exception as e:
        return jsonify({'status': 'gagal', 'error': str(e)}), 500

if _name_ == '_main_':
    # Untuk akses publik via IP VM
    app.run(host='0.0.0.0', port=5000)