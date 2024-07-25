import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Veri seti yolunun tanımlanması
data_dir = r'C:\Users\Betul\Desktop\Pistachio_Image_Dataset'

# Görüntü dosyalarının ve etiketlerin yüklenmesi
def load_images_and_labels(data_dir):
    images, labels = [], []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(img_path)
                    labels.append(class_name)
    return images, labels

image_paths, labels = load_images_and_labels(data_dir)

# Pandas DataFrame'e görüntü dosya yolları ve etiketlerin eklenmesi
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# Görüntülerin yüklenmesi ve işlenmesi
def process_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye dönüştür
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalizasyon
    return image

df['processed_image'] = df['image_path'].map(lambda x: process_image(x, size=(128, 128)))

# Veri setinin eğitim, doğrulama ve test setlerine 70/15/15 oranında bölünmesi
images = np.stack(df['processed_image'].values)
labels = pd.factorize(df['label'])[0]

# Eğitim seti için %70
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)

# Kalan %30'u doğrulama ve test setlerine %15/%15 olarak bölme
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Eğitim, doğrulama ve test setlerinin boyutlarını yazdırma
print(f"Eğitim veri seti boyutu: {X_train.shape}")
print(f"Doğrulama veri seti boyutu: {X_val.shape}")
print(f"Test veri seti boyutu: {X_test.shape}")

# Data Augmentation tekniklerinin uygulanması
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1
)

train_generator = datagen.flow(X_train, y_train, batch_size=16)
validation_generator = datagen.flow(X_val, y_val, batch_size=16)

# VGG16 modelini yükleyin, son fully connected katmanları dahil olmadan
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Modelin katmanlarını dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modelin üzerine ekleme yapın
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Modeli derleyin
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli eğitin
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 16,
    validation_steps=len(X_val) // 16
)

# Modeli fine-tuning için hazırlayın
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Modeli yeniden derleyin
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli fine-tuning ile yeniden eğitin
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 16,
    validation_steps=len(X_val) // 16
)
# Modeli kaydet
model.save('vgg16_finetuned.h5')

# Test verilerini kaydet
np.savez('test_data.npz', X_test=X_test, y_test=y_test)


# Modeli değerlendirin
test_generator = datagen.flow(X_test, y_test, batch_size=16, shuffle=False)
test_loss, test_acc = model.evaluate(test_generator, steps=len(X_test) // 16)
print(f"Test Doğruluğu: {test_acc:.4f}")

# Tahminler
y_pred_probs = model.predict(test_generator, steps=len(X_test) // 16)
y_pred = (y_pred_probs > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test[:len(y_pred)], y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

# Fine-tuning sonrası modeli değerlendirin
test_loss_fine, test_acc_fine = model.evaluate(test_generator, steps=len(X_test) // 16)
print(f"İnce Ayar Sonrası Test Doğruluğu: {test_acc_fine:.4f}")

# Fine-tuning sonrası tahminler
y_pred_probs_fine = model.predict(test_generator, steps=len(X_test) // 16)
y_pred_fine = (y_pred_probs_fine > 0.5).astype(int)

# Fine-tuning sonrası Confusion Matrix
cm_fine = confusion_matrix(y_test[:len(y_pred_fine)], y_pred_fine)
disp_fine = ConfusionMatrixDisplay(confusion_matrix=cm_fine, display_labels=['Class 0', 'Class 1'])

# Her sınıftan belirli sayıda görüntü gösterme fonksiyonu
def show_sample_images(df, n_per_class=1):
    classes = df['label'].unique()
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2, len(classes) * 2))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        class_samples = df[df['label'] == cls].sample(n_per_class)
        for j, (index, row) in enumerate(class_samples.iterrows()):
            ax = axes[i * n_per_class + j]
            image = row['processed_image']
            label = row['label']
            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Tüm sonuçları ve grafikleri aynı anda göster
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Eğitim ve doğrulama doğruluğu grafiği
axes[0, 0].plot(history.history['accuracy'], label='Eğitim Doğruluğu')
axes[0, 0].plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Doğruluk')
axes[0, 0].legend(loc='lower right')
axes[0, 0].set_title('Eğitim ve Doğrulama Doğruluğu')

# Eğitim ve doğrulama kaybı grafiği
axes[0, 1].plot(history.history['loss'], label='Eğitim Kaybı')
axes[0, 1].plot(history.history['val_loss'], label='Doğrulama Kaybı')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Kayıp')
axes[0, 1].legend(loc='upper right')
axes[0, 1].set_title('Eğitim ve Doğrulama Kaybı')

# Fine-tuning sonrası eğitim ve doğrulama doğruluğu grafiği
axes[1, 0].plot(history_fine.history['accuracy'], label='Eğitim Doğruluğu (Fine-Tuning)')
axes[1, 0].plot(history_fine.history['val_accuracy'], label='Doğrulama Doğruluğu (Fine-Tuning)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Doğruluk')
axes[1, 0].legend(loc='lower right')
axes[1, 0].set_title('Fine-Tuning: Eğitim ve Doğrulama Doğruluğu')

# Fine-tuning sonrası eğitim ve doğrulama kaybı grafiği
axes[1, 1].plot(history_fine.history['loss'], label='Eğitim Kaybı (Fine-Tuning)')
axes[1, 1].plot(history_fine.history['val_loss'], label='Doğrulama Kaybı (Fine-Tuning)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Kayıp')
axes[1, 1].legend(loc='upper right')
axes[1, 1].set_title('Fine-Tuning: Eğitim ve Doğrulama Kaybı')

plt.tight_layout()
plt.show()

# Confusion Matrix ve Classification Report
disp.plot()
plt.title('Initial Model Confusion Matrix')
plt.show()
print("Classification Report:\n", classification_report(y_test[:len(y_pred)], y_pred, target_names=['Class 0', 'Class 1']))

disp_fine.plot()
plt.title('Fine-Tuned Model Confusion Matrix')
plt.show()
print("Fine-Tuning Sonrası Classification Report:\n", classification_report(y_test[:len(y_pred_fine)], y_pred_fine, target_names=['Class 0', 'Class 1']))

# Her sınıftan 2 görüntü göster
show_sample_images(df, n_per_class=2)

