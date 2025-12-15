# app.py
from model_loader import load_sam_model, get_segmentation_masks
from utils import read_image, display_results
from generator import load_generator_pipeline, generate_redesign_image
import os


# --------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------------------------------

def get_clean_labels(classified_objects):
    """
    Sadece anlamlı, Generative AI için uygun etiketleri içeren yeni bir liste oluşturur.
    """
    clean_objects = []

    # Tüm TRANSLATION_DICT çıktılarını içeren anlamlı etiketler
    ANLAMLI_ETIKETLER = {
        'yatak', 'koltuk', 'sandalye', 'masa', 'lamba', 'gardırop', 'dolap',
        'komodin', 'ayna', 'halı', 'raf', 'puf', 'kanepe', 'çalışma masası',
        'pencere', 'duvar', 'tavan', 'zemin', 'perde', 'yorgan', 'yastık',
        'çerçeve', 'sanat eseri', 'bitki', 'vazo', 'ocak', 'lavabo', 'örtü', 'boş alan', 'space'
    }

    for obj in classified_objects:
        label = obj['label']

        if label in ANLAMLI_ETIKETLER and label not in ['Sınıflandırılamadı', 'Çok Küçük Nesne', 'Model Hatası']:
            clean_objects.append(obj)

    return clean_objects


def initial_analysis_and_suggestion(classified_objects):
    """Sınıflandırılmış nesneleri kullanarak ilk analiz ve önerileri oluşturur."""
    print("\n--- İlk Analiz ve Öneri Oluşturma ---")
    if classified_objects is None or not classified_objects:
        print("Görüntüde nesne tespit edilemedi.")
        return

    labels = [obj['label'] for obj in classified_objects if
              obj['label'] not in ['Sınıflandırılamadı', 'Çok Küçük Nesne', 'Model Hatası']]
    label_counts = {item: labels.count(item) for item in set(labels)}

    print(f"Analiz: Odada {len(labels)} anlamlı nesne tespit edildi.")
    print("Tespit Edilen Nesneler (Etiketler):")
    for label, count in label_counts.items():
        print(f"- **{label}**: {count} adet")

    print("\nGenel Standartlara Göre Öneriler:")
    if 'koltuk' in label_counts or 'kanepe' in label_counts:
        print("-> Oturma alanınıza uygun aydınlatma ile mekanı daha geniş gösterebilirsiniz.")
    if 'duvar' in label_counts:
        print("-> Duvar renginiz nötr. Daha dinamik bir tasarım için bir Vurgu Duvarı (Accent Wall) oluşturulabilir.")

    print("--------------------------------------\n")


def redesign_and_search_items(classified_objects, original_image_np):
    """Kullanıcının prompt'una göre yeniden tasarımı yapar ve eşya arama linklerini sunar."""
    print("\n--- Yeniden Tasarım ve Eşya Arama ---")

    # 1. Kullanıcıdan prompt alımı
    user_prompt = input(
        "Lütfen yeni tasarım için promptunuzu girin (Örn: modern minimalist koltuklar, beyaz duvarlar): ")

    # 2. Generative AI ile yeniden tasarım
    print("\nAdım 2: Yeni Tasarım Görseli Oluşturuluyor (Stable Diffusion)...")

    # generate_redesign_image'a temizlenmiş nesne listesi ve orijinal görüntü gönderiliyor
    redesigned_image_pil = generate_redesign_image(original_image_np, classified_objects, user_prompt)

    if redesigned_image_pil:
        redesigned_image_pil.show()
        print("Yeniden tasarlanmış görüntü gösterildi.")
    else:
        print("Yeniden tasarım başarısız oldu.")

    # 3. Eşya Arama (Placeholder)
    print("\nAdım 3: Yeni Tasarımda Kullanılan Ürünler İçin Yönlendirme Linkleri (Simülasyon):")

    # Örnek Arama (Kullanıcının girdiği prompt'a göre)
    if 'koltuk' in [obj['label'] for obj in classified_objects]:
        print(f"* **{user_prompt.split(',')[0]} Mobilyalar:** Arama linki: [E-Ticaret/Arama Motoru URL'si]")

    print("-----------------------------------\n")


# --------------------------------------------------------------------------

def main():
    """Uygulamanın ana akışını çalıştırır."""
    test_image_path = "test_oda_fotografi.jpg"

    try:
        input_image = read_image(test_image_path)
        print(f"Görüntü okundu: {test_image_path}")

        # Generative AI Pipeline'ını yükle (Ağırlık indirilecektir)
        load_generator_pipeline()

        sam_model = load_sam_model()
        if sam_model is None:
            return

        classified_objects = get_segmentation_masks(input_image, sam_model)

        final_clean_objects = get_clean_labels(classified_objects)

        # Görselleştirme (Sınıflandırılamadı etiketlerinin de görünmesi için tüm listeyi kullanıyoruz)
        display_results(input_image, classified_objects, "Tespit Edilen Nesneler ve Etiketler (Temizlenmiş)")

        initial_analysis_and_suggestion(final_clean_objects)

        # Orijinal görüntüyü yeniden tasarım fonksiyonuna gönderiyoruz
        redesign_and_search_items(final_clean_objects, input_image)

    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
        print(
            "Lütfen 'test_oda_fotografi.jpg' dosyasını koyduğunuzdan ve model ağırlıklarının bulunduğundan emin olun.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    main()