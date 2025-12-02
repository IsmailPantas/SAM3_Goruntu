# app.py
from model_loader import load_sam_model, get_segmentation_masks
from utils import read_image, display_results
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

        # Kontrol: Etiket, bizim anlamlı kabul ettiğimiz Türkçe listede var mı?
        if label in ANLAMLI_ETIKETLER:
            clean_objects.append(obj)

    return clean_objects


def initial_analysis_and_suggestion(classified_objects):
    """Sınıflandırılmış nesneleri kullanarak ilk analiz ve önerileri oluşturur."""
    print("\n--- İlk Analiz ve Öneri Oluşturma ---")
    if classified_objects is None or not classified_objects:
        print("Görüntüde nesne tespit edilemedi.")
        return

    labels = [obj['label'] for obj in classified_objects]
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


def redesign_and_search_items(classified_objects):
    """Kullanıcının prompt'una göre yeniden tasarımı simüle eder ve eşya arama linklerini sunar."""
    print("\n--- Yeniden Tasarım ve Eşya Arama ---")

    print("Adım 1: Kullanıcı Prompt'u Alındı (Örn: 'Duvar rengini mavi yap, halıyı değiştir.')")
    print("Adım 2: Yeni Tasarım Görseli Oluşturuluyor (Generative AI)...")
    # BURADA GENERATIVE AI ENTEGRASYONU YAPILACAKTIR.

    # 3. Eşya Arama (Uygulamanın en büyük farkı)
    print("\nAdım 3: Yeni Tasarımda Kullanılan Ürünler İçin Yönlendirme Linkleri:")

    if 'lamba' in [obj['label'] for obj in classified_objects]:
        print("* **Modern Sarkıt Lamba:** 'Modern Sarkıt Lamba' için arama [Link E-Ticaret/Arama Motoru URL'si]")

    print("* **Turkuaz Mavi Duvar Boyası:** 'Turkuaz iç cephe boyası' için arama [Link Yapı Market URL'si]")
    print("-----------------------------------\n")


# --------------------------------------------------------------------------

def main():
    """Uygulamanın ana akışını çalıştırır."""
    test_image_path = "test_oda_fotografi2.jpg"

    try:
        input_image = read_image(test_image_path)
        print(f"Görüntü okundu: {test_image_path}")

        sam_model = load_sam_model()
        if sam_model is None:
            return

        classified_objects = get_segmentation_masks(input_image, sam_model)

        # Temizlik ve Filtreleme
        final_clean_objects = get_clean_labels(classified_objects)

        display_results(input_image, final_clean_objects, "Tespit Edilen Nesneler ve Etiketler (Temizlenmiş)")

        initial_analysis_and_suggestion(final_clean_objects)
        redesign_and_search_items(final_clean_objects)

    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
        print(
            "Lütfen 'test_oda_fotografi.jpg' dosyasını koyduğunuzdan ve model ağırlıklarının bulunduğundan emin olun.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    main()