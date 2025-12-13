# classifier.py
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np

# --------------------------------------------------------------------------
# GLOBAL TANIMLAMALAR
# --------------------------------------------------------------------------

INTERIOR_WHITELIST = [
    'bed', 'sofa', 'chair', 'table', 'lamp', 'wardrobe', 'closet',
    'nightstand', 'mirror', 'rug', 'cabinet', 'shelf', 'ottoman',
    'wall', 'ceiling', 'floor', 'tile', 'plaster', 'wallpaper', 'room', 'space',
    'quilt', 'cushion', 'pillow', 'curtain', 'window', 'shade', 'frame', 'pouf',
    'couch', 'sectional', 'daybed', 'dresser', 'comforter', 'press', 'console',
    'art', 'picture', 'painting', 'basket', 'plant', 'vase', 'clock', 'sconce',
    'stove', 'sink', 'washbasin', 'lavabo', 'desk'
]

TRANSLATION_DICT = {
    'bed': 'yatak', 'sofa': 'koltuk', 'chair': 'sandalye', 'table': 'masa',
    'lamp': 'lamba', 'wardrobe': 'gardırop', 'closet': 'dolap',
    'nightstand': 'komodin', 'mirror': 'ayna', 'rug': 'halı',
    'cabinet': 'dolap', 'shelf': 'raf', 'ottoman': 'puf', 'pouf': 'puf',
    'couch': 'kanepe', 'desk': 'çalışma masası', 'window': 'pencere',
    'wall': 'duvar', 'ceiling': 'tavan', 'floor': 'zemin', 'tile': 'fayans',
    'curtain': 'perde', 'quilt': 'yorgan', 'cushion': 'yastık',
    'pillow': 'yastık', 'frame': 'çerçeve', 'art': 'sanat eseri',
    'picture': 'resim', 'painting': 'tablo', 'basket': 'sepet',
    'plant': 'bitki', 'vase': 'vazo', 'stove': 'ocak', 'sink': 'lavabo',
    'space': 'boş alan', 'windsor': 'yastık',
    'tobacco': 'dolap', 'shop': 'dolap', 'vest': 'yastık', 'tie': 'yastık',
    'abaya': 'örtü', 'daybed': 'yatak', 'plate': 'raf', 'rack': 'raf'
}

# Yüzdelik Alan Eşiği ve Listesi
LARGE_AREA_PERCENT_THRESHOLD = 0.05
STRUCTURAL_WHITELIST = ['wall', 'floor', 'ceiling', 'room', 'space', 'plaster', 'tile']
NEGATIVE_FILTER_LIST = ['tobacco', 'bulletproof', 'abaya', 'bula', 'bola', 'tie', 'windsor',
                        'terrier', 'retriever', 'bulldog', 'schnauzer', 'beagle', 'swab', 'mop']

# SINIFLANDIRMA İÇİN MİNİMUM GÜVEN EŞİĞİ (ADE20K tarzı optimizasyon)
MIN_CONFIDENCE_THRESHOLD = 0.15

# Kullanılacak ViT modelinin adı (Strateji 1)
MODEL_NAME = "google/vit-large-patch16-224"

# Modeli ve işlemciyi global olarak tanımla
feature_extractor = None
classification_model = None
DEVICE = "cpu"

try:
    print(f"ViT Sınıflandırma modeli yükleniyor... ({MODEL_NAME})")
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    classification_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    classification_model.to(DEVICE)
    print("ViT modeli başarıyla yüklendi.")
except Exception as e:
    print(f"ViT modeli yüklenirken hata oluştu: {e}")
    feature_extractor = None
    classification_model = None


# --------------------------------------------------------------------------
# FONKSİYONLAR
#sfdsfasa
# --------------------------------------------------------------------------

def get_label_from_id(idx):
    """Sınıf ID'sini temiz etikete dönüştürür."""
    label = classification_model.config.id2label[idx]
    if ":" in label:
        return label.split(":")[-1].strip().lower()
    return label.lower()


def get_base_label(full_label):
    """Çok kelimeli bir etiketten Whitelist'e uyan ilk temel nesne adını döndürür."""
    parts = full_label.replace(',', ' ').replace('-', ' ').split()
    for part in parts:
        if part in INTERIOR_WHITELIST:
            return part
    return full_label


def classify_cropped_object(cropped_np_image, area_size, total_image_area, top_k=5):
    """
    Kırpılmış nesneyi sınıflandırır, alan yüzdesine dayalı yapısal kontrol yapar
    ve Whitelist kullanarak en uygun etiketi seçer.
    """
    if classification_model is None:
        return "Model Hatası"
    if cropped_np_image.size == 0 or cropped_np_image.shape[0] < 16 or cropped_np_image.shape[1] < 16:
        return "Çok Küçük Nesne"

    # Yüzdelik eşiği piksel cinsine çevirme
    required_area_pixels = total_image_area * LARGE_AREA_PERCENT_THRESHOLD

    try:
        image = Image.fromarray(cropped_np_image)
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = classification_model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

        top_1_confidence = top_k_probs.squeeze()[0].item()
        top_1_label = get_label_from_id(top_k_indices.squeeze()[0].item())

        final_english_label = None

        # GÜVEN KONTROLÜ
        if top_1_confidence < MIN_CONFIDENCE_THRESHOLD:
            # Yapısal nesneler (duvar/zemin) büyükse, düşük güvene rağmen devam et
            if not (area_size > required_area_pixels):
                return "Sınıflandırılamadı"

        # 1. BÜYÜK ALAN KONTROLÜ (Yüzdelik Eşiğe Göre Kontrol)
        if area_size > required_area_pixels:
            base_label = get_base_label(top_1_label)
            if base_label in STRUCTURAL_WHITELIST:
                final_english_label = base_label

        # 2. NORMAL AKIŞ (Negatif Filtre ve Whitelist Eşleşmesi)
        if final_english_label is None:
            for idx in top_k_indices.squeeze().tolist():
                label = get_label_from_id(idx)

                # Blacklist Kontrolü
                if any(neg_word in label for neg_word in NEGATIVE_FILTER_LIST):
                    continue

                base_label = get_base_label(label)

                if base_label in INTERIOR_WHITELIST:
                    final_english_label = base_label
                    break

        # 3. FİLTRESİZ SON ÇARE (Blacklist'e takılmamışsa Top-1'i al)
        if final_english_label is None:
            if any(neg_word in top_1_label for neg_word in NEGATIVE_FILTER_LIST):
                return "Sınıflandırılamadı"

            final_english_label = get_base_label(top_1_label)

        # 4. TÜRKÇELEŞTİRME
        return TRANSLATION_DICT.get(final_english_label, final_english_label)

    except Exception as e:
        return "Sınıflandırma Hatası"