# generator.py (OPTİMİZASYONLU VE DÜZELTİLMİŞ VERSİYON)
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
import re
from classifier import TRANSLATION_DICT
import cv2  # Maske Dilasyonu için OpenCV kütüphanesini import ettik!

# --------------------------------------------------------------------------
# GLOBAL TANIMLAMALAR
# --------------------------------------------------------------------------

MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Inpainting optimizasyon hiperparametreleri
CFG_SCALE = 9.5  # Prompt'a sadakat seviyesi (7-8'den 9.5'e yükseltildi)
MASK_DILATION_SIZE = 15  # Yapısal maskeyi kaç piksel genişleteceğimiz (10-20 arası ideal)

# Cihaz belirleme
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

inpainting_pipeline = None


def load_generator_pipeline():
    """Stable Diffusion Inpainting Pipeline'ını yükler."""
    global inpainting_pipeline
    print(f"\nGenerative AI (Inpainting) modeli yükleniyor... ({MODEL_ID})")

    try:
        dtype = torch.float32

        inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        ).to(DEVICE)
        print(f"Generative AI modeli başarıyla yüklendi. (Kullanılan Cihaz: {DEVICE.upper()})")
        return inpainting_pipeline
    except Exception as e:
        print(f"HATA: Generative AI modeli yüklenirken hata oluştu: {e}")
        inpainting_pipeline = None
        return None


def create_redesign_prompt(classified_objects, user_prompt):
    """
    Kullanıcı prompt'u ve tespit edilen nesneleri birleştirerek Stable Diffusion için
    detaylı bir Inpainting prompt'u oluşturur.
    """
    present_objects = set()

    for obj in classified_objects:
        if obj['label'] not in ['Sınıflandırılamadı', 'Çok Küçük Nesne', 'Model Hatası']:
            present_objects.add(obj['label'])

    style_part = f"interior design, {user_prompt}, cinematic light, photorealistic"

    object_list = ", ".join(list(present_objects))
    context_part = f"interior scene with {object_list}"

    final_prompt = f"{style_part}. A highly detailed {context_part}."

    final_prompt = re.sub(r'\s+', ' ', final_prompt).strip()

    return final_prompt


def find_largest_structural_mask(classified_objects):
    """
    Duvar ve zemin etiketleri güvenilir değilse, en büyük iki maskeyi bulur.
    """
    structural_labels = set(TRANSLATION_DICT.values()).intersection({'duvar', 'zemin', 'tavan'})

    large_masks = []
    for obj in classified_objects:
        if obj['label'] in ['Sınıflandırılamadı', 'boş alan'] or obj['label'] in structural_labels:
            large_masks.append((obj['area'], obj['mask'], obj))

    large_masks.sort(key=lambda x: x[0], reverse=True)

    return large_masks[:3]


def generate_redesign_image(original_image_np, classified_objects, user_prompt):
    """
    Orijinal görüntüyü ve maskeyi kullanarak Stable Diffusion ile yeni görsel üretir.
    """
    global inpainting_pipeline
    if inpainting_pipeline is None:
        return None

    # --- 1. Maskeleri Birleştirme ve Kullanıcı İsteği Kontrolü ---

    H, W, _ = original_image_np.shape
    combined_mask_np = np.zeros((H, W), dtype=bool)
    objects_to_change = set()

    CHANGEABLE_OBJECTS = ['koltuk', 'kanepe', 'sandalye', 'masa', 'halı', 'lamba', 'dolap', 'puf', 'yatak', 'komodin']
    prompt_lower = user_prompt.lower()

    is_wall_requested = any(
        word in prompt_lower for word in ['duvar', 'wall', 'siyah duvar', 'beyaz duvar', 'duvarları'])
    is_floor_requested = any(word in prompt_lower for word in ['zemin', 'floor', 'yer', 'fayans'])

    # Eğer Duvar veya Zemin istenmişse, en büyük maskeleri bul ve maskele
    if is_wall_requested or is_floor_requested:
        largest_masks_info = find_largest_structural_mask(classified_objects)

        for area, mask, obj in largest_masks_info:
            current_label = obj['label']

            if (is_wall_requested and current_label in ['duvar', 'Sınıflandırılamadı', 'boş alan']) or \
                    (is_floor_requested and current_label in ['zemin', 'Sınıflandırılamadı', 'boş alan']):

                # MASK DİLASYONU (Genişletme) UYGULANIYOR!
                kernel = np.ones((MASK_DILATION_SIZE, MASK_DILATION_SIZE), np.uint8)
                # bool maskeyi uint8'e çevir
                mask_uint8 = mask.astype(np.uint8) * 255
                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
                dilated_mask_bool = dilated_mask.astype(bool)

                # 1. Maskeyi birleştir (genişletilmiş maske ile)
                combined_mask_np = np.logical_or(combined_mask_np, dilated_mask_bool)

                # 2. Etiketi objects_to_change setine ekle
                if is_wall_requested and 'duvar' not in objects_to_change:
                    objects_to_change.add('duvar')
                if is_floor_requested and 'zemin' not in objects_to_change:
                    objects_to_change.add('zemin')

                # Sadece en büyük maskeyi aldıktan sonra diğer yapısal maskeleri almayı durdur
                if is_wall_requested or is_floor_requested:
                    break  # En büyük (muhtemelen duvar) maskeyi aldıktan sonra döngüden çık

    # Mobilya maskelerini birleştirme (genişletme yapılmaz)
    for obj in classified_objects:
        if obj['label'] in CHANGEABLE_OBJECTS:
            combined_mask_np = np.logical_or(combined_mask_np, obj['mask'])
            objects_to_change.add(obj['label'])

    if not objects_to_change:
        print("UYARI: Yeniden tasarlanacak anlamlı nesne (mobilya, duvar, zemin) bulunamadı.")
        return Image.fromarray(original_image_np)

    # NumPy dizilerini PIL Image formatına dönüştürme
    image = Image.fromarray(original_image_np).convert("RGB")
    mask_image = Image.fromarray((combined_mask_np.astype(np.uint8) * 255)).convert("L")

    # --- 2. Prompt'u Oluşturma ---
    full_prompt = create_redesign_prompt(classified_objects, user_prompt)

    # --- 3. Görüntü Oluşturma (Inpainting) ---

    print(f"\n--- Generative AI İşlemi Başlatılıyor (CFG: {CFG_SCALE}, Dilasyon: {MASK_DILATION_SIZE}px) ---")
    print(f"Maskelenenler: {', '.join(objects_to_change)}")
    print(f"Nihai Prompt: {full_prompt[:120]}...")

    try:
        generator = torch.Generator(DEVICE).manual_seed(42)

        output = inpainting_pipeline(
            prompt=full_prompt,
            image=image,
            mask_image=mask_image,
            # CFG_SCALE Kullanılıyor
            guidance_scale=CFG_SCALE,
            negative_prompt="bad quality, blurry, noise, distortions, disfigured, monochrome, cartoon, painting, changed perspective, changed furniture location",
            num_inference_steps=50,
            generator=generator
        )
        print("Generative AI yeniden tasarımı tamamlandı.")
        return output.images[0]

    except Exception as e:
        print(f"Generative AI üretimi sırasında hata oluştu: {e}")
        return None