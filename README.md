# SAM3 Görüntü Segmentasyon ve Nesne Sınıflandırma

SAM tabanlı segmentasyon ile oda fotoğraflarındaki nesneleri otomatik tespit eden ve ViT ile sınıflandırıp Türkçe etiketleyen bir demo uygulama.  
Çıktıda maskeler, etiketler ve basit tasarım önerileri gösterilir.

---

## Özellikler

- **Segmentasyon:** Meta Segment Anything (SAM) ile otomatik maske üretimi.
- **Sınıflandırma:** ViT (`google/vit-large-patch16-224`) ile kırpılmış nesnelerin sınıflandırılması.
- **Türkçe etiketler:** Whitelist + çeviri sözlüğü ile anlamlı Türkçe etiketleme.
- **Görselleştirme:** Rastgele renkli maskeler ve bbox üzeri etiket gösterimi.
- **Öneriler:** Basit oda analizi ve yeniden tasarım adımlarının iskeleti.

---

## Gereksinimler

- Python 3.9+ (önerilir)  
- CUDA destekli GPU (opsiyonel; yoksa CPU kullanılır)  
- Sistem paketleri:
  - Linux: `libgl1`
  - macOS: OpenCV bağımlılıkları

---

## Kurulum

```bash
git clone <repo-url>
cd SAM3_Goruntu

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Model Ağırlıkları

`models/` klasörüne aşağıdaki dosyayı ekleyin:

```bash
models/sam_vit_l_0b3195.pth
```

Ağırlık yoksa SAM yüklenemez ve segmentasyon yapılamaz.  
Doğru model tipi: **vit_l**

---

## Kullanım

```bash
python app.py
```

Varsayılan giriş görseli: **test_oda_fotografi2.jpg**

Çıktılar:

- Konsolda: model yükleme logları, etiketler, basit analiz ve öneriler  
- Pencerede: maskeler ve etiketlerle görselleştirilmiş görüntü

Kendi görselinizi kullanmak için `app.py` içindeki `test_image_path` değişkenini güncelleyin.

---

## Proje Yapısı

```
SAM3_Goruntu/
│
├── app.py                     # Ana akış
├── model_loader.py            # SAM yükleme + maske üretimi
├── classifier.py              # ViT sınıflandırma + whitelist/çeviri
├── utils.py                   # Yardımcı fonksiyonlar, çizimler
│
├── models/
│   └── sam_vit_l_0b3195.pth   # SAM ağırlıkları (elle eklenmeli)
│
├── requirements.txt
├── README.md
├── test_oda_fotografi1.jpg
└── test_oda_fotografi2.jpg
```

---

## Ana Akış (Kısa)

1. Görsel okunur (`utils.read_image`)  
2. SAM yüklenir (`model_loader.load_sam_model`)  
3. Otomatik maskeler üretilir (`SamAutomaticMaskGenerator`)  
4. Her maske kırpılır ve ViT ile sınıflandırılır (`classifier.classify_cropped_object`)  
5. Anlamlı Türkçe etiketler filtrelenir (`app.get_clean_labels`)  
6. Maskeler ve etiketler çizilir (`utils.display_results`)  
7. Basit analiz ve öneri çıktıları üretilir

---

## Özelleştirme

- **Giriş görseli:** `app.py` → `test_image_path`  
- **Anlamlı etiket seti:** `app.py` → `ANLAMLI_ETIKETLER`  
- **Whitelist / çeviri:** `classifier.py` → `INTERIOR_WHITELIST`, `TRANSLATION_DICT`  
- **SAM ayarları:** `model_loader.py` → `SamAutomaticMaskGenerator` parametreleri  
- **Güven eşiği:** `classifier.py` → `MIN_CONFIDENCE_THRESHOLD`, `LARGE_AREA_PERCENT_THRESHOLD`

---

## Bilinen Sınırlamalar

- SAM ağırlığı olmadan çalışmaz.  
- CPU’da yavaş olabilir; GPU önerilir.  
- ViT modeli genel amaçlıdır; iç mekân özelinde hatalar verebilir.  
- Görselleştirme masaüstü GUI gerektirir (matplotlib).

---

## Geliştirme Fikirleri

- Generative AI ile yeniden tasarım görseli üretimi (placeholder mevcut)  
- E-ticaret arama linklerinin gerçek entegrasyonu  
- CLI argümanları ile dinamik görsel ve eşik seçimi  
- Test eklenmesi ve hafif model seçenekleri

---

## Örnek Çalıştırma Çıktısı (Kısa)

- Konsol: `Loaded SAM model: vit_l`, `Generated 12 masks`, `Detected labels: sandalye (0.92), masa (0.89), halı (0.85)`  
- Görsel penceresi: Orijinal görsel üzerinde renkli maskeler ve bbox etiketleri

---

## Lisans & Katkı

- MIT Lisansı önerilir — dilediğiniz lisansı ekleyin.  
- Katkı için `CONTRIBUTING.md` oluşturabilirsiniz.

---

## İletişim

Herhangi bir problem veya geliştirme önerisi için issue açın veya PR gönderin.
