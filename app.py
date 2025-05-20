# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load model
# model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')

# @app.route('/prediksi', methods=['POST'])
# def prediksi():
#     file = request.files['gambar']
#     img = Image.open(file.stream).resize((299, 299))
#     img = np.expand_dims(np.array(img) / 255.0, axis=0)
#     pred = model.predict(img)
#     kelas = int(np.argmax(pred))  
#     return jsonify({'nama': f'User ID: {kelas}'})

# # Fungsi utama untuk menjalankan Flask
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# app = Flask(__name__)

# # Load model
# model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')

# # Mapping kelas ke nama dan kode karyawan
# kelas_ke_info = [
#     {'nama': 'Aggasi',   'kode': 'K001'},
#     {'nama': 'Akbar',    'kode': 'K002'},
#     {'nama': 'Apri',     'kode': 'K003'},
#     {'nama': 'Ari',      'kode': 'K004'},
#     {'nama': 'Bellza',   'kode': 'K005'},
#     {'nama': 'Darimi',   'kode': 'K006'},
#     {'nama': 'Doni',     'kode': 'K007'},
#     {'nama': 'Gilang',   'kode': 'K008'},
#     {'nama': 'Heppy',    'kode': 'K009'},
#     {'nama': 'Heri',     'kode': 'K010'},
#     {'nama': 'Holidy',   'kode': 'K011'},
#     {'nama': 'Irvan',    'kode': 'K012'},
#     {'nama': 'Julia',    'kode': 'K013'},
#     {'nama': 'Meylin',   'kode': 'K014'},
#     {'nama': 'Muzartun', 'kode': 'K015'},
#     {'nama': 'Nofri',    'kode': 'K016'},
#     {'nama': 'Rangga',   'kode': 'K017'},
#     {'nama': 'Rinaldi',  'kode': 'K018'},
#     {'nama': 'Riska',    'kode': 'K019'},
#     {'nama': 'Sandy',    'kode': 'K020'},
#     {'nama': 'Solihin',  'kode': 'K021'},
#     {'nama': 'Sonia',    'kode': 'K022'},
#     {'nama': 'Wahyu',    'kode': 'K023'},
#     {'nama': 'Yudi',     'kode': 'K024'},
#     {'nama': 'Yusuf',    'kode': 'K025'}
# ]

# @app.route('/prediksi', methods=['POST'])
# def prediksi():
#     file = request.files['gambar']
#     img = Image.open(file.stream).resize((299, 299))
#     img = np.expand_dims(np.array(img) / 255.0, axis=0)

#     pred = model.predict(img)
#     kelas = int(np.argmax(pred))
#     confidence = float(np.max(pred))

#     if kelas < len(kelas_ke_info):
#         nama = kelas_ke_info[kelas]['nama']
#         kode = kelas_ke_info[kelas]['kode']
#     else:
#         nama = 'Tidak diketahui'
#         kode = 'N/A'

#     return jsonify({
#         'kelas': kelas,
#         'nama': nama,
#         'kode_karyawan': kode,
#         'confidence': confidence
#     })

# # Fungsi utama
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# 2. Saat ini
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from mtcnn import MTCNN
# import cv2

# app = Flask(__name__)

# # Load model
# model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')
# detector = MTCNN()

# # Mapping kelas ke nama dan kode karyawan
# kelas_ke_info = [
#     {'nama': 'Aggasi',   'kode': 'K001'},
#     {'nama': 'Akbar',    'kode': 'K002'},
#     {'nama': 'Apri',     'kode': 'K003'},
#     {'nama': 'Ari',      'kode': 'K004'},
#     {'nama': 'Bellza',   'kode': 'K005'},
#     {'nama': 'Darimi',   'kode': 'K006'},
#     {'nama': 'Doni',     'kode': 'K007'},
#     {'nama': 'Gilang',   'kode': 'K008'},
#     {'nama': 'Heppy',    'kode': 'K009'},
#     {'nama': 'Heri',     'kode': 'K010'},
#     {'nama': 'Holidy',   'kode': 'K011'},
#     {'nama': 'Irvan',    'kode': 'K012'},
#     {'nama': 'Julia',    'kode': 'K013'},
#     {'nama': 'Meylin',   'kode': 'K014'},
#     {'nama': 'Muzartun', 'kode': 'K015'},
#     {'nama': 'Nofri',    'kode': 'K016'},
#     {'nama': 'Rangga',   'kode': 'K017'},
#     {'nama': 'Rinaldi',  'kode': 'K018'},
#     {'nama': 'Riska',    'kode': 'K019'},
#     {'nama': 'Sandy',    'kode': 'K020'},
#     {'nama': 'Solihin',  'kode': 'K021'},
#     {'nama': 'Sonia',    'kode': 'K022'},
#     {'nama': 'Wahyu',    'kode': 'K023'},
#     {'nama': 'Yudi',     'kode': 'K024'},
#     {'nama': 'Yusuf',    'kode': 'K025'}
# ]

# @app.route('/prediksi', methods=['POST'])
# def prediksi():
#     file = request.files['gambar']
#     img_pil = Image.open(file.stream).convert('RGB')
#     img_np = np.array(img_pil)

#     # Deteksi wajah
#     faces = detector.detect_faces(img_np)
#     if not faces:
#         return jsonify({'error': 'Wajah tidak terdeteksi'}), 400

#     # Ambil wajah pertama
#     x, y, w, h = faces[0]['box']
#     face_crop = img_np[y:y+h, x:x+w]

#     # Resize ke 299x299
#     face_resized = cv2.resize(face_crop, (299, 299))
#     face_normalized = np.expand_dims(face_resized / 255.0, axis=0)

#     # Prediksi
#     pred = model.predict(face_normalized)
#     kelas = int(np.argmax(pred))
#     confidence = float(np.max(pred))

#     if kelas < len(kelas_ke_info):
#         nama = kelas_ke_info[kelas]['nama']
#         kode = kelas_ke_info[kelas]['kode']
#     else:
#         nama = 'Tidak diketahui'
#         kode = 'N/A'

#     return jsonify({
#         'kelas': kelas,
#         'nama': nama,
#         'kode_karyawan': kode,
#         'confidence': confidence
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# from flask import Flask, request, jsonify
# from PIL import Image
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from mtcnn.mtcnn import MTCNN
# import os

# # Inisialisasi Flask & model
# app = Flask(__name__)
# model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')  # Path ke model kamu
# detector = MTCNN()

# # Mapping kelas ke info karyawan
# kelas_ke_info = [
#     {'nama': 'Aggasi',   'kode': 'K001'},
#     {'nama': 'Akbar',    'kode': 'K002'},
#     {'nama': 'Apri',     'kode': 'K003'},
#     {'nama': 'Ari',      'kode': 'K004'},
#     {'nama': 'Bellza',   'kode': 'K005'},
#     {'nama': 'Darimi',   'kode': 'K006'},
#     {'nama': 'Doni',     'kode': 'K007'},
#     {'nama': 'Gilang',   'kode': 'K008'},
#     {'nama': 'Heppy',    'kode': 'K009'},
#     {'nama': 'Heri',     'kode': 'K010'},
#     {'nama': 'Holidy',   'kode': 'K011'},
#     {'nama': 'Irvan',    'kode': 'K012'},
#     {'nama': 'Julia',    'kode': 'K013'},
#     {'nama': 'Meylin',   'kode': 'K014'},
#     {'nama': 'Muzartun', 'kode': 'K015'},
#     {'nama': 'Nofri',    'kode': 'K016'},
#     {'nama': 'Rangga',   'kode': 'K017'},
#     {'nama': 'Rinaldi',  'kode': 'K018'},
#     {'nama': 'Riska',    'kode': 'K019'},
#     {'nama': 'Sandy',    'kode': 'K020'},
#     {'nama': 'Solihin',  'kode': 'K021'},
#     {'nama': 'Sonia',    'kode': 'K022'},
#     {'nama': 'Wahyu',    'kode': 'K023'},
#     {'nama': 'Yudi',     'kode': 'K024'},
#     {'nama': 'Yusuf',    'kode': 'K025'}
# ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'gambar' not in request.files:
#         return jsonify({'error': 'File gambar tidak ditemukan'}), 400

#     file = request.files['gambar']
#     img_pil = Image.open(file.stream).convert('RGB')
#     img_np = np.array(img_pil)

#     # Deteksi wajah
#     faces = detector.detect_faces(img_np)
#     if not faces:
#         return jsonify({'error': 'Wajah tidak terdeteksi'}), 400

#     x, y, w, h = faces[0]['box']
#     x, y = max(0, x), max(0, y)
#     face_crop = img_np[y:y+h, x:x+w]

#     # Resize dan preprocessing
#     face_resized = cv2.resize(face_crop, (299, 299))
#     img_array = keras_image.img_to_array(face_resized)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Prediksi
#     prediction = model.predict(img_array)
#     predicted_class_idx = int(np.argmax(prediction))
#     confidence = float(np.max(prediction))

#     if predicted_class_idx < len(kelas_ke_info):
#         nama = kelas_ke_info[predicted_class_idx]['nama']
#         kode = kelas_ke_info[predicted_class_idx]['kode']
#     else:
#         nama = 'Tidak diketahui'
#         kode = '-'

#     return jsonify({
#         'kelas': predicted_class_idx,
#         'nama': nama,
#         'kode_karyawan': kode,
#         'confidence': round(confidence, 4)
#     })

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2
import datetime

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')
detector = MTCNN()

# Mapping kelas ke nama dan kode karyawan
kelas_ke_info = [
    {'nama': 'Aggasi',   'kode': 'K001'},
    {'nama': 'Akbar',    'kode': 'K002'},
    {'nama': 'Apri',     'kode': 'K003'},
    {'nama': 'Ari',      'kode': 'K004'},
    {'nama': 'Bellza',   'kode': 'K005'},
    {'nama': 'Darimi',   'kode': 'K006'},
    {'nama': 'Doni',     'kode': 'K007'},
    {'nama': 'Gilang',   'kode': 'K008'},
    {'nama': 'Heppy',    'kode': 'K009'},
    {'nama': 'Heri',     'kode': 'K010'},
    {'nama': 'Holidy',   'kode': 'K011'},
    {'nama': 'Irvan',    'kode': 'K012'},
    {'nama': 'Julia',    'kode': 'K013'},
    {'nama': 'Meylin',   'kode': 'K014'},
    {'nama': 'Muzartun', 'kode': 'K015'},
    {'nama': 'Nofri',    'kode': 'K016'},
    {'nama': 'Rangga',   'kode': 'K017'},
    {'nama': 'Rinaldi',  'kode': 'K018'},
    {'nama': 'Riska',    'kode': 'K019'},
    {'nama': 'Sandy',    'kode': 'K020'},
    {'nama': 'Solihin',  'kode': 'K021'},
    {'nama': 'Sonia',    'kode': 'K022'},
    {'nama': 'Wahyu',    'kode': 'K023'},
    {'nama': 'Yudi',     'kode': 'K024'},
    {'nama': 'Yusuf',    'kode': 'K025'}
]

@app.route('/prediksi', methods=['POST'])
def prediksi():
        file = request.files['gambar']
        img_pil = Image.open(file.stream).convert('RGB')
        img_np = np.array(img_pil)

        # Deteksi wajah
        faces = detector.detect_faces(img_np)
        if not faces:
            return jsonify({'status': 'gagal', 'error': 'Wajah tidak terdeteksi'}), 400

        # Ambil wajah pertama
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)  # Hindari koordinat negatif
        face_crop = img_np[y:y+h, x:x+w]

        # Resize ke ukuran input model
        face_resized = cv2.resize(face_crop, (299, 299))
        face_normalized = np.expand_dims(face_resized / 255.0, axis=0)

        # Prediksi
        pred = model.predict(face_normalized)
        kelas = int(np.argmax(pred))
        confidence = float(np.max(pred))
        threshold = 0.80

        if kelas >= len(kelas_ke_info):
            return jsonify({'status': 'gagal', 'error': 'Kelas tidak dikenali oleh sistem'}), 400

        # if confidence < threshold:
        #     return jsonify({'status': 'gagal', 'error': 'Prediksi tidak meyakinkan', 'confidence': confidence}), 400

        # Ambil data karyawan
        nama = kelas_ke_info[kelas]['nama']
        kode = kelas_ke_info[kelas]['kode']

        # (Opsional) Catat absensi ke file log
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # with open("log_absensi.txt", "a") as f:
        #     f.write(f"{timestamp} | {kode} | {nama} | hadir\n")

        return jsonify({
            'status': 'sukses',
            'kelas': kelas,
            'nama': nama,
            'kode_karyawan': kode,
            'confidence': round(confidence, 4)
        })

    # except Exception as e:
    #     return jsonify({'status': 'gagal', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
