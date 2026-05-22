import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split # Tambahkan ini
import joblib

IMG_SIZE = 224

def train_cnn():
    # Gabungkan data agar bisa dibagi ulang 85:15
    df_train = pd.read_csv('fruit_train.csv')
    df_test = pd.read_csv('fruit_test.csv')
    # Mengambil hanya sebagian data jika 13rb data terlalu berat
    df_total = pd.concat([df_train, df_test]).sample(frac=0.3, random_state=42)

    images, labels = [], []

    print("[STEP 1] Preprocessing Gambar...")
    for idx, row in df_total.iterrows():
        img = cv2.imread(row['file'])
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = (img.astype(np.float32) / 127.5) - 1.0 
            images.append(img_norm)
            labels.append(row['label'])

    X = np.array(images)
    y = np.array(labels)

    # PEMBAGIAN DATA 85% Training & 15% Testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"Data Training: {len(X_train)} | Data Testing: {len(X_val)}")

    print("[STEP 2] Building MobileNetV2...")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("[STEP 3] Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    joblib.dump(model, 'model_fruit_cnn.pkl')
    print("✅ Model CNN 85:15 Berhasil Melatih!")

if __name__ == "__main__":
    train_cnn()