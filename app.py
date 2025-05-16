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

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('inceptionv3_face_model_32_50_0001.h5')

# Mapping index kelas ke nama asli
kelas_ke_nama = [
    'Aggasi', 'Akbar', 'Apri', 'Ari', 'Bellza', 'Darimi', 'Doni', 'Gilang',
    'Heppy', 'Heri', 'Holidy', 'Irvan', 'Julia', 'Meylin', 'Muzartun',
    'Nofri', 'Rangga', 'Rinaldi', 'Riska', 'Sandy', 'Solihin', 'Sonia',
    'Wahyu', 'Yudi', 'Yusuf'
]

@app.route('/prediksi', methods=['POST'])
def prediksi():
    file = request.files['gambar']
    img = Image.open(file.stream).resize((299, 299))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(img)
    kelas = int(np.argmax(pred))
    nama = kelas_ke_nama[kelas] if kelas < len(kelas_ke_nama) else "Tidak diketahui"

    return jsonify({
        'kelas': kelas,
        'nama': nama,
        'confidence': float(np.max(pred))  # optional: tambahkan kepercayaan
    })

# Fungsi utama untuk menjalankan Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
