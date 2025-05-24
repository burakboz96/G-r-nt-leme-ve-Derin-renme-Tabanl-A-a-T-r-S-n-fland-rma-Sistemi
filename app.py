import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# 📁 1. AYARLAR
data_dir = "/Users/burakbozoglu/Desktop/Agac_Gorselleri"
model_path = "mobilenet_agac_modeli.h5"
metadata_path = "dataset_metadata.txt"
batch_size = 32
img_size = (224, 224)
num_epochs = 25

# 📌 2. Veri Zaman Kontrolü
def get_latest_modification_time(directory):
    latest_time = 0
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                latest_time = max(latest_time, file_time)
    return latest_time

if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        saved_mod_time = float(f.read().strip())
else:
    saved_mod_time = 0

current_mod_time = get_latest_modification_time(data_dir)

# 📦 3. Veri Artırma ve Sınıflar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode="nearest"
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(train_data.class_indices.keys())

# ⚖️ 4. class_weight Hesaplama
y_train = train_data.classes
class_weights_array = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

# ✅ 5. Model Eğitimi Gerekli mi?
if os.path.exists(model_path) and saved_mod_time == current_mod_time:
    print("✅ Eğitimli model bulundu. Model yükleniyor...")
    model = keras.models.load_model(model_path)
else:
    print("🚀 Yeni veri tespit edildi. Model eğitiliyor...")

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=num_epochs,
                        class_weight=class_weights)

    model.save(model_path)
    with open(metadata_path, "w") as f:
        f.write(str(current_mod_time))

    print("✅ Eğitim tamamlandı ve model kaydedildi.")

    # 🎯 Performans Göstergeleri
    val_data.reset()
    y_true = val_data.classes
    y_pred = np.argmax(model.predict(val_data), axis=1)

    print("🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 📈 Grafikler
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Eğitim Doğruluğu")
    plt.plot(val_acc, label="Doğrulama Doğruluğu")
    plt.title("Doğruluk")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Eğitim Kaybı")
    plt.plot(val_loss, label="Doğrulama Kaybı")
    plt.title("Kayıp")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("model_performans.png")
    plt.show()

    # 🎯 Ek Metrikler
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"F1 Score: {round(f1, 4)} - Precision: {round(precision, 4)} - Recall: {round(recall, 4)}")

    # 🔍 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.show()

# 🌐 6. Flask API
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        return jsonify({
            "class": class_names[idx],
            "confidence": round(float(preds[idx]), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 7. Sunucu Başlat
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
