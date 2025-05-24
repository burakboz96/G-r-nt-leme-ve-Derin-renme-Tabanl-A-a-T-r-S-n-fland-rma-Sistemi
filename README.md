# 🌳 Görüntü İşleme ve Derin Öğrenme Tabanlı Ağaç Türü Sınıflandırma Sistemi



## 👤 Burak Bozoğlu  


---

## 📌 1. Giriş

Doğal ekosistemlerin sürdürülebilirliği ve orman kaynaklarının etkin yönetimi, biyolojik çeşitliliğin izlenmesi ve korunmasıyla doğrudan ilişkilidir. Bu çerçevede, **ağaç türlerinin doğru ve hızlı bir şekilde sınıflandırılması**, orman envanteri çalışmaları, habitat izleme sistemleri ve çevresel karar destek mekanizmaları açısından büyük önem taşır.

Bu çalışmada, çeşitli ağaç türlerine ait görseller kullanılarak eğitilen bir **derin öğrenme modeli** aracılığıyla, bilinmeyen bir görüntünün ait olduğu ağaç türünün otomatik olarak tahmin edilmesi amaçlanmıştır.

<img width="1580" alt="Ekran Resmi 2025-05-24 22 31 46" src="https://github.com/user-attachments/assets/4759b1e2-d119-4f61-9ddf-addd7e56964d" />
---

## 🧠 Kullanılan Yöntemler ve Yaklaşımlar

- **Model Mimarisi:**  
  - [MobileNetV2](https://arxiv.org/abs/1801.04381) (Transfer Learning yöntemiyle)
  - Hafif ve hızlı yapısı sayesinde mobil ve gömülü sistemlerde kullanıma uygundur.

- **Eğitim Stratejisi:**  
  - İlk aşamada modelin alt katmanları dondurularak eğitim yapılmış, ardından belirli katmanlar yeniden eğitilerek **fine-tuning** uygulanmıştır.

- **Veri Ön İşleme:**  
  - Kapsamlı **data augmentation** teknikleri kullanılmıştır (çevrim, döndürme, yakınlaştırma vb.).
  - **class_weight** yöntemiyle sınıf dengesizlikleri giderilmiştir.


---

## 📊 Performans Metrikleri

Model başarısı sadece doğrulukla değil, şu gelişmiş metriklerle değerlendirilmiştir:

- ✅ Accuracy (Doğruluk)
- 📈 Precision (Hassasiyet)
- 🔁 Recall (Duyarlılık)
- 📊 F1 Score
- 🧩 Confusion Matrix

---

## 🚀 Nasıl Kullanılır?

```bash
# Gerekli kütüphaneleri yükleyin
pip install tensorflow numpy matplotlib scikit-learn

# Modeli eğitin veya mevcut modeli yükleyin
python train_model.py

# Test verisi üzerinde tahmin yapın
python predict.py --image path_to_image.jpg
