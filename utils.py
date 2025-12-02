# utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, color=None, random_color=False):
    """Segmentasyon maskesini görüntü üzerine görselleştirmek için yardımcı fonksiyon."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is None:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def read_image(image_path):
    """Görüntüyü OpenCV ile okur ve RGB formatına dönüştürür."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def display_results(image, classified_objects, title="Tespit Edilen Nesneler ve Etiketler"):
    """Görüntüyü, üzerindeki maskeleri ve etiketleri gösterir."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    if classified_objects:
        for obj in classified_objects:
            mask = obj['mask']
            label = obj['label']
            x, y, w, h = obj['bbox']

            show_mask(mask, ax, random_color=True)

            # Etiketi (label) sınırlayıcı kutunun üstüne yaz
            ax.text(int(x), int(y) - 10, label,
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

    plt.title(title)
    plt.axis('off')
    plt.show()