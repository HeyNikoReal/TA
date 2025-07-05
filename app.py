# # Import Library 
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import time
# import numpy as np
# from PIL import Image
# from mtcnn import MTCNN
# import cv2

# app = Flask(__name__)

# # Load model 
# model = tf.keras.models.load_model('16_50_0001.h5')

# # Mapping kelas ke nama dan id karyawan
# kelas_ke_info = [
#     {'nama': 'Aggasi', 'id': '001'},
#     {'nama': 'Andira', 'id': '002'},
#     {'nama': 'Apri', 'id': '003'},
#     {'nama': 'Ariko', 'id': '004'},
#     {'nama': 'Benny', 'id': '005'},
#     {'nama': 'Darimi', 'id': '006'},
#     {'nama': 'Dermosis', 'id': '007'},
#     {'nama': 'Dicky', 'id': '008'},
#     {'nama': 'Doni', 'id': '009'},
#     {'nama': 'Gilang', 'id': '010'},
#     {'nama': 'Heppy', 'id': '011'},
#     {'nama': 'Heri', 'id': '012'},
#     {'nama': 'Holidy', 'id': '013'},
#     {'nama': 'Irvan', 'id': '014'},
#     {'nama': 'Muzartun', 'id': '015'},
#     {'nama': 'Nofri', 'id': '016'},
#     {'nama': 'Rangga', 'id': '017'},
#     {'nama': 'Nuryanto', 'id': '018'},
#     {'nama': 'Rafi', 'id': '019'},
#     {'nama': 'Rangga', 'id': '020'},
#     {'nama': 'Ricky', 'id': '021'},
#     {'nama': 'Sandy', 'id': '022'},
#     {'nama': 'Solihin', 'id': '023'},
#     {'nama': 'Wahyu', 'id': '024'},
#     {'nama': 'Yusuf', 'id': '025'}
# ]

# detector = MTCNN()

# # Fungsi crop wajah menggunakan MTCNN
# def crop_face_mtcnn(img_pil, output_size=(299, 299)):
#     img_np = np.array(img_pil)
#     faces = detector.detect_faces(img_np)

#     if not faces:
#         return None  # Tidak ada wajah terdeteksi

#     x, y, w, h = faces[0]['box']
#     x1, y1 = abs(x), abs(y) # Menghindari nilai negatif
#     x2, y2 = x1 + w, y1 + h
#     face_crop = img_np[y1:y2, x1:x2]
#     face_img = Image.fromarray(face_crop).resize(output_size)
#     return face_img

# @app.route('/prediksi', methods=['POST'])
# def prediksi():
#     try:
#         # start_time = time.time() 
#         # Ambil file gambar dari request
#         file = request.files.get('gambar')
#         if not file:
#             return jsonify({'status': 'gagal', 'error': 'File gambar tidak ditemukan'}), 400

#         img_pil = Image.open(file.stream).convert('RGB')
#         wajah = crop_face_mtcnn(img_pil)

#         if wajah is None:
#             return jsonify({'status': 'gagal', 'error': 'Wajah tidak terdeteksi'}), 400

#         wajah_array = np.array(wajah) / 255.0
#         wajah_normalized = np.expand_dims(wajah_array, axis=0)

#         # Prediksi
#         pred = model.predict(wajah_normalized)
#         kelas = int(np.argmax(pred))
#         confidence = float(np.max(pred))

#         if kelas >= len(kelas_ke_info):
#             return jsonify({'status': 'gagal', 'error': 'Kelas tidak dikenali'}), 400

#         info = kelas_ke_info[kelas]

#         # end_time = time.time() 
#         # processing_time = round(end_time - start_time, 4)

#         return jsonify({
#             'status': 'sukses',
#             'kelas': kelas,
#             'nama': info['nama'],
#             'id': info['id'],
#             'confidence': round(confidence, 4),
#             # 'processing_time': processing_time
#         })

#     except Exception as e:
#         return jsonify({'status': 'gagal', 'error': str(e)}), 500

# if __name__ == '__main__':
#     # Untuk akses publik via IP VM
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import onnxruntime
import cv2

app = Flask(__name__)

# === Load Models ===
model_recog = tf.keras.models.load_model('16_50_0001.h5')  # face recognition
sess_fas = onnxruntime.InferenceSession('modelrgb.onnx', providers=['CPUExecutionProvider'])  # anti-spoofing

# === Mapping nama karyawan ===
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

# === Preprocessing untuk model anti-spoofing (112x112 RGB NCHW) ===
def preprocess_antispoof(img_rgb):
    img = cv2.resize(img_rgb, (112, 112)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

# === Fungsi anti-spoofing ===
def predict_real_or_fake(face_np):
    inp = preprocess_antispoof(face_np)
    out = sess_fas.run(None, {sess_fas.get_inputs()[0].name: inp})[0]
    return float(out[0][1])

# === Fungsi deteksi wajah & cropping ===
def crop_face(img_pil):
    img_np = np.array(img_pil)
    faces = detector.detect_faces(img_np)
    if not faces:
        return None, None

    x, y, w, h = faces[0]['box']
    x1, y1 = abs(x), abs(y)
    cropped_np = img_np[y1:y1+h, x1:x1+w]
    
    face_for_spoofing = cropped_np
    face_for_recog = Image.fromarray(cropped_np).resize((299, 299))
    
    return face_for_recog, face_for_spoofing

@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        file = request.files.get('gambar')
        if not file:
            return jsonify({'status': 'gagal', 'error': 'Gambar tidak ditemukan'}), 400

        img_pil = Image.open(file.stream).convert('RGB')
        face_pil, face_np = crop_face(img_pil)

        if face_pil is None:
            return jsonify({'status': 'gagal', 'error': 'Wajah tidak terdeteksi'}), 400

        # === 1. Deteksi apakah real atau fake ===
        score = predict_real_or_fake(face_np)
        if score <= 0.5:
            return jsonify({'status': 'gagal', 'foto': 'fake', 'score': round(score, 4)}), 200

        # === 2. Jika real, klasifikasi wajah ===
        face_arr = np.array(face_pil) / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)
        pred = model_recog.predict(face_arr)
        kelas = int(np.argmax(pred))
        confidence = float(np.max(pred))

        if kelas >= len(kelas_ke_info):
            return jsonify({'status': 'gagal', 'error': 'Kelas tidak dikenali'}), 400

        info = kelas_ke_info[kelas]

        return jsonify({
            'status': 'sukses',
            'foto': 'real',
            'score': round(score, 4),
            'kelas': kelas,
            'nama': info['nama'],
            'id': info['id'],
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({'status': 'gagal', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
