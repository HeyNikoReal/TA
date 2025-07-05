from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2
import onnxruntime

app = Flask(__name__)

# === Load Model ===
model_recog = tf.keras.models.load_model('16_50_0001.h5')  # Model face recognition
sess_fas = onnxruntime.InferenceSession('modelrgb.onnx', providers=['CPUExecutionProvider'])  # Model anti-spoofing

# === Data Karyawan ===
kelas_ke_info = [
    {'nama': 'Aggasi', 'id': '001'}, {'nama': 'Andira', 'id': '002'},
    {'nama': 'Apri', 'id': '003'}, {'nama': 'Ariko', 'id': '004'},
    {'nama': 'Benny', 'id': '005'}, {'nama': 'Darimi', 'id': '006'},
    {'nama': 'Dermosis', 'id': '007'}, {'nama': 'Dicky', 'id': '008'},
    {'nama': 'Doni', 'id': '009'}, {'nama': 'Gilang', 'id': '010'},
    {'nama': 'Heppy', 'id': '011'}, {'nama': 'Heri', 'id': '012'},
    {'nama': 'Holidy', 'id': '013'}, {'nama': 'Irvan', 'id': '014'},
    {'nama': 'Muzartun', 'id': '015'}, {'nama': 'Nofri', 'id': '016'},
    {'nama': 'Rangga', 'id': '017'}, {'nama': 'Nuryanto', 'id': '018'},
    {'nama': 'Rafi', 'id': '019'}, {'nama': 'Rangga', 'id': '020'},
    {'nama': 'Ricky', 'id': '021'}, {'nama': 'Sandy', 'id': '022'},
    {'nama': 'Solihin', 'id': '023'}, {'nama': 'Wahyu', 'id': '024'},
    {'nama': 'Yusuf', 'id': '025'}
]

detector = MTCNN()

# === Fungsi Preprocessing Anti-Spoofing ===
def preprocess_antispoof(face_rgb):
    img = cv2.resize(face_rgb, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img

# === Prediksi Real atau Fake ===
def predict_real_or_fake(face_rgb):
    inp = preprocess_antispoof(face_rgb)
    out = sess_fas.run(None, {sess_fas.get_inputs()[0].name: inp})[0]
    score = float(out[0][1])  # Ambil nilai probabilitas real
    return score

# === Fungsi Deteksi & Crop Wajah ===
def crop_face_mtcnn(img_pil, out_size=(299, 299)):
    img_np = np.array(img_pil)  # RGB
    faces = detector.detect_faces(img_np)

    if not faces:
        return None, None

    x, y, w, h = faces[0]['box']
    x1, y1 = abs(x), abs(y)
    face_np = img_np[y1:y1+h, x1:x1+w]
    face_pil = Image.fromarray(face_np).resize(out_size)
    return face_pil, face_np  # PIL + array RGB

# === Endpoint Prediksi ===
@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        file = request.files.get('gambar')
        if not file:
            return jsonify({'status': 'gagal', 'error': 'File gambar tidak ditemukan'}), 400

        img_pil = Image.open(file.stream).convert('RGB')
        face_pil, face_np = crop_face_mtcnn(img_pil)

        if face_pil is None:
            return jsonify({'status': 'gagal', 'error': 'Wajah tidak terdeteksi'}), 200

        # === Cek apakah wajah real/fake ===
        score = predict_real_or_fake(face_np)
        if score <= 0.5:
            return jsonify({
                'status': 'gagal',
                'foto': 'fake',
                'score': round(score, 4)
            }), 200

        # === Jika real, lanjut ke klasifikasi wajah ===
        face_arr = np.array(face_pil) / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)
        pred = model_recog.predict(face_arr)
        kelas = int(np.argmax(pred))
        confidence = float(np.max(pred))

        if kelas >= len(kelas_ke_info):
            return jsonify({'status': 'gagal', 'error': 'Kelas tidak dikenali'}), 200

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

# === Run Aplikasi ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
