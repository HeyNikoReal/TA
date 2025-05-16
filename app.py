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
    img = Image.open(file.stream).resize((299, 299))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(img)
    kelas = int(np.argmax(pred))
    confidence = float(np.max(pred))

    if kelas < len(kelas_ke_info):
        nama = kelas_ke_info[kelas]['nama']
        kode = kelas_ke_info[kelas]['kode']
    else:
        nama = 'Tidak diketahui'
        kode = 'N/A'

    return jsonify({
        'kelas': kelas,
        'nama': nama,
        'kode_karyawan': kode,
        'confidence': confidence
    })

# Fungsi utama
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
