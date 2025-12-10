SAM3 Görüntü Segmentasyon ve Nesne Sınıflandırma
SAM tabanlı segmentasyon ile oda fotoğraflarındaki nesneleri otomatik tespit eden ve ViT ile sınıflandırıp Türkçe etiketleyen bir demo uygulama. Çıktıda maskeler, etiketler ve basit tasarım önerileri gösterilir.
Özellikler
Segmentasyon: Meta Segment Anything (SAM) ile otomatik maske üretimi.
Sınıflandırma: ViT (google/vit-large-patch16-224) ile kırpılmış nesnelerin sınıflandırılması.
Türkçe etiketler: Whitelist + çeviri sözlüğü ile anlamlı Türkçe etiketler.
Görselleştirme: Rastgele renkli maskeler ve bbox üstü etiketler.
Öneriler: Basit oda analizi ve yeniden tasarım adımlarının iskeleti.
Gereksinimler
Python 3.9+ (önerilir)
CUDA destekli GPU (opsiyonel; yoksa otomatik CPU kullanılır)
Sistem paketleri: libgl1 / macOS’ta opencv bağımlılıkları
Kurulum
txt
git clone <repo-url> SAM3_Goruntucd SAM3_Goruntupython -m venv .venvsource .venv/bin/activate  # Windows: .venv\Scripts\activatepip install --upgrade pippip install -r requirements.txt
Model Ağırlıkları
models/sam_vit_l_0b3195.pth dosyasını models/ klasörüne yerleştirin.
Ağırlık yoksa SAM yüklenemez ve segmentasyon yapılmaz.
Doğru model tipi: vit_l.
Kullanım
python app.py
Varsayılan giriş görseli: test_oda_fotografi2.jpg. Çıktılar:
Konsolda: model yükleme logları, etiketler, basit analiz ve öneriler.
Pencerede: maskeler ve etiketlerle görselleştirilmiş görüntü.
Kendi görselinizi kullanmak için app.py içindeki test_image_path değişkenini güncelleyin.
Yapı
app.py: Ana akış; görsel okuma, SAM yükleme, maske üretimi, sınıflandırma, etiket temizleme, görselleştirme, öneriler.
model_loader.py: SAM yükleme ve otomatik maske üretimi.
classifier.py: ViT ile sınıflandırma, whitelist/blacklist filtreleri ve Türkçe çeviri.
utils.py: Görsel okuma, maske/etiket çizimi yardımcıları.
models/: SAM ağırlıkları.
test_oda_fotografi*.jpg: Örnek görseller.
Ana Akış (Kısa)
Görsel okunur (utils.read_image).
SAM yüklenir (model_loader.load_sam_model).
Otomatik maskeler üretilir (SamAutomaticMaskGenerator).
Her maske kırpılır ve ViT ile sınıflandırılır (classifier.classify_cropped_object).
Anlamlı Türkçe etiketler filtrelenir (app.get_clean_labels).
Maskeler ve etiketler çizilir (utils.display_results).
Basit analiz ve öneri çıktıları üretilir.
Özelleştirme
Giriş görseli: app.py → test_image_path.
Anlamlı etiket seti: app.py → ANLAMLI_ETIKETLER.
Whitelist/çeviri: classifier.py → INTERIOR_WHITELIST, TRANSLATION_DICT.
SAM ayarları: model_loader.py → SamAutomaticMaskGenerator parametreleri.
Güven eşiği: classifier.py → MIN_CONFIDENCE_THRESHOLD, LARGE_AREA_PERCENT_THRESHOLD.
Bilinen Sınırlamalar
SAM ağırlığı olmadan çalışmaz.
CPU’da yavaş olabilir; GPU önerilir.
ViT modeli genel amaçlıdır, iç mekân özelinde hatalar verebilir.
Görselleştirme, masaüstü ortamı gerektirir (matplotlib GUI).
Geliştirme Fikirleri
Generative AI ile yeniden tasarım görseli üretimi (placeholder mevcut).
E-ticaret arama linklerinin gerçek entegrasyonu.
CLI argümanları ile dinamik görsel ve eşik seçimi.
Test eklenmesi ve hafif model seçenekleri.
