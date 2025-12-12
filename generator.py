# generator.py
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
import re
from classifier import TRANSLATION_DICT  # ViT etiket sözlüğünden faydalanmak için import edildi

# --------------------------------------------------------------------------
# GLOBAL TANIMLAMALAR
# --------------------------------------------------------------------------

MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Cihaz belirleme: CUDA yoksa MPS (Apple Silicon GPU) kullan, o da yoksa CPU kullan.
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
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

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
    detaylı ve 77 token sınırına uygun bir Inpainting prompt'u oluşturur.
    """

    present_objects = set()

    for obj in classified_objects:
        # Anlamsız etiketleri prompt'a katmamak için filtreleme
        if obj['label'] not in ['Sınıflandırılamadı', 'Çok Küçük Nesne', 'Model Hatası']:
            present_objects.add(obj['label'])

            # --- 1. Prompt'u Oluşturma (Kısa ve Öz) ---

    style_part = f"interior design, {user_prompt}, cinematic light, photorealistic"

    # 2. Odanın Mevcut İçeriği (Modelin bağlamı korumasına yardımcı olur)
    object_list = ", ".join(list(present_objects))
    context_part = f"interior scene with {object_list}"

    final_prompt = f"{style_part}. A highly detailed {context_part}."

    final_prompt = re.sub(r'\s+', ' ', final_prompt).strip()

    return final_prompt


def find_largest_structural_mask(classified_objects):
    """
    Duvar ve zemin etiketleri güvenilir değilse, en büyük iki maskeyi bulur
    ve bu maskelerin 'duvar' veya 'zemin' olma olasılığı en yüksektir.
    """
    structural_labels = set(TRANSLATION_DICT.values()).intersection({'duvar', 'zemin', 'tavan'})

    # Sadece büyük maskeleri tut
    large_masks = []
    for obj in classified_objects:
        # Sadece sınıflandırılamamış veya yapısal (duvar, zemin, tavan) olarak etiketlenmiş büyük maskelere odaklan
        if obj['label'] in ['Sınıflandırılamadı', 'Çok Küçük Nesne'] or obj['label'] in structural_labels:
            large_masks.append((obj['area'], obj['mask'], obj))

    # Alan büyüklüğüne göre sırala (en büyüğü en başta)
    large_masks.sort(key=lambda x: x[0], reverse=True)

    # En büyük 3 maskenin (Duvar/Zemin/Tavan olma ihtimali yüksek) maske ve orijinal objelerini döndür
    return large_masks[:3]  # En büyük 3 maskeyi al


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

    # Varsayılan değiştirilebilir nesneler (Mobilyalar)
    CHANGEABLE_OBJECTS = ['koltuk', 'kanepe', 'sandalye', 'masa', 'halı', 'lamba', 'dolap', 'puf', 'yatak', 'komodin']

    # YENİ VE KRİTİK MANTIK: Duvar/Zemin İsteklerini Zorlama
    prompt_lower = user_prompt.lower()

    is_wall_requested = any(
        word in prompt_lower for word in ['duvar', 'wall', 'siyah duvar', 'beyaz duvar', 'duvarları'])
    is_floor_requested = any(word in prompt_lower for word in ['zemin', 'floor', 'yer', 'fayans'])

    # Eğer Duvar veya Zemin istenmişse, en büyük maskeleri bul ve maskele
    if is_wall_requested or is_floor_requested:
        largest_masks_info = find_largest_structural_mask(classified_objects)

        for area, mask, obj in largest_masks_info:
            current_label = obj['label']

            # Duvar veya Zemin istenmişse ve maske henüz maskelenmemişse
            if (is_wall_requested and current_label in ['duvar', 'Sınıflandırılamadı']) or \
                    (is_floor_requested and current_label in ['zemin', 'Sınıflandırılamadı']):

                # Maskeyi birleştir ve etiketi zorla ekle
                combined_mask_np = np.logical_or(combined_mask_np, mask)

                if is_wall_requested:
                    objects_to_change.add('duvar')
                elif is_floor_requested:
                    objects_to_change.add('zemin')

                # Bu maske zaten zorla eklendiği için mobilya döngüsüne girmeye gerek yok

    # Mobilya maskelerini birleştirme
    for obj in classified_objects:
        if obj['label'] in CHANGEABLE_OBJECTS:
            # Maske zaten yapısal override ile eklenmiş olabilir, tekrar ekleme zararsızdır
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

    print(f"\n--- Generative AI İşlemi Başlatılıyor ---")
    print(f"Maskelenenler: {', '.join(objects_to_change)}")
    print(f"Nihai Prompt: {full_prompt[:120]}...")

    try:
        generator = torch.Generator(DEVICE).manual_seed(42)

        output = inpainting_pipeline(
            prompt=full_prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt="bad quality, blurry, noise, distortions, disfigured, monochrome, cartoon, painting",
            num_inference_steps=50,
            generator=generator
        )
        print("Generative AI yeniden tasarımı tamamlandı.")
        return output.images[0]

    except Exception as e:
        print(f"Generative AI üretimi sırasında hata oluştu: {e}")
        return None