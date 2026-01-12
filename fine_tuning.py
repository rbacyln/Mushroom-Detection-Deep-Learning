import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import train_generator, test_generator

# 1. Modeli "Derlemeden" Yükle (Hata almamak için en güvenli yol)
print("--- Model Yükleniyor (Sadece Mimari ve Ağırlıklar) ---")
# compile=False diyerek o "float 3.0" hatasını bypass ediyoruz
model = load_model('mantar_modeli_v1.h5', compile=False)

# 2. Fine-Tuning İçin Katman Ayarları
# Modelin içindeki MobileNetV3 katmanını buluyoruz (Genelde ilk katmandır)
base_model = model.layers[0] 
base_model.trainable = True

# Son 20 katman hariç hepsini dondur
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 3. Modeli Manuel Olarak Yeniden Derle
print("--- Model Yeniden Derleniyor ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. İnce Ayar Eğitimini Başlat
print("--- Fine-Tuning Başlıyor (Hedef: %80+ Başarı) ---")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 5. Final Modelini Yeni Formatla Kaydet
model.save('mantar_modeli_final.keras')
print("\nFinal modeli başarıyla kaydedildi!")