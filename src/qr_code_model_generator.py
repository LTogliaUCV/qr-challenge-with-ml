import os
import qrcode
import cv2
import numpy as np
from zxing import *
from keras.preprocessing import image
import tensorflow as tf

# Crea un directorio para guardar los códigos QR
if not os.path.exists('qr_codes'):
    os.makedirs('qr_codes')

# Genera 1000 códigos QR aleatorios y los guarda en el directorio
for i in range(1000):
    data = 'QR code ' + str(i)
    filename = 'qr_codes/qr_code_' + str(i) + '.png'
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)

# Carga las imágenes de los códigos QR
X = []
for i in range(1000):
    filename = 'qr_codes/qr_code_' + str(i) + '.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Carga la imagen en escala de grises
    img = cv2.resize(img, (100, 100))  # Redimensiona la imagen

    # Agrega la imagen preprocesada a la lista de características
    X.append(img)

# Convierte la lista de características en un ndarray
X = np.array(X)

# Carga las etiquetas de los códigos QR
y = []
for i in range(1000):
    data = i
    y.append(data)

# Convierte la lista de etiquetas en un ndarray de enteros
y = np.array(y, dtype=int)

# Agrega una dimensión adicional para el número de canales
X = np.expand_dims(X, axis=-1)

# Construye el modelo CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1000)
])

# Compila el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrena el modelo con los datos preprocesados
model.fit(X, y, epochs=10, batch_size=32)
# Guarda el modelo en un archivo HDF5
model.save('qr-challenge-with-ml/qr_model.h5')