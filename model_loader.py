# model_loader.py
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import numpy as np
from classifier import classify_cropped_object

# --------------------------------------------------------------------------
# SAM MODEL YÜKLEME AYARLARI
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sam_vit_l_0b3195.pth")
MODEL_TYPE = "vit_l"


def load_sam_model():
    """SAM modelini yükler."""
    print(f"SAM modeli yükleniyor... ({MODEL_TYPE})")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        print("SAM modeli başarıyla yüklendi.")
        return sam
    except FileNotFoundError:
        print(f"HATA: SAM modeli ağırlık dosyası bulunamadı: {MODEL_PATH}")
        return None
    except Exception as e:
        print(f"SAM modeli yüklenirken beklenmedik hata oluştu: {e}")
        return None


# --------------------------------------------------------------------------
# SEGMENTASYON VE SINIFLANDIRMA İŞLEMİ
# --------------------------------------------------------------------------

def get_segmentation_masks(image, sam_model):
    """
    Görüntü için otomatik maske oluşturmayı başlatır ve nesneleri ViT ile sınıflandırır.
    """
    if sam_model is None:
        print("SAM modeli yüklenemedi. Segmentasyon iptal edildi.")
        return None

    # SAM Parametreleri (Segmentasyon kalitesi için yüksek eşikler)
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=16,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
        min_mask_region_area=2000
    )

    print("Otomatik segmentasyon başlatılıyor...")
    results = mask_generator.generate(image)

    H, W, _ = image.shape
    TOTAL_IMAGE_AREA = H * W  # Toplam alan hesaplandı

    print(f"Toplam {len(results)} nesne adayı tespit edildi.")

    classified_objects = []

    for result in results:
        mask = result['segmentation']
        area = result['area']
        x, y, w, h = result['bbox']

        x = int(x);
        y = int(y);
        w = int(w);
        h = int(h)
        x_max, y_max = image.shape[1], image.shape[0]
        x_end = min(x + w, x_max);
        y_end = min(y + h, y_max)

        if x_end <= x or y_end <= y:
            continue

        cropped_image = image[y:y_end, x:x_end]

        # total_image_area parametresi eklendi
        object_label = classify_cropped_object(cropped_image, area, TOTAL_IMAGE_AREA)

        classified_objects.append({
            'mask': mask,
            'bbox': result['bbox'],
            'label': object_label
        })

    return classified_objects