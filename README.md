# CableRouteFinder

Kabelübersichtsplan (KÜP) içindeki kabloların Kabellageplan (KLP) üzerindeki yolunu
**otomatik** bulur ve çizer. Şu ana kadar CablePlan'de elle çizilen kablo yollarının yerini alması amaçlanıyor.

## Ne yapar?

```
KÜP (PDF) ──► kablo listesi: numara, başlangıç, bitiş, uzunluk, kesit
                    │
KLP (PDF) ──► trase ağı (kırmızı kesikli çizgiler → graf)
          ──► kablo numarası etiketleri + trase üzerindeki bağlantı noktaları
          ──► eleman isimleri (W22/G4004, 3045Y, KS1307000, SK1307110 …)
                    │
                    ▼
      her kablo için trase grafında en kısa bağlantılı yol
      (etiketlerin bağlandığı tüm noktalardan + başlangıç/bitiş elemanından geçen)
                    │
                    ▼
  • işaretli PDF (her kablo ayrı katman/layer)
  • CablePlan JSON'u (data/<Plan>.json, doğrudan CablePlan'de açılır)
  • rapor (CSV/JSON): KÜP uzunluğu ↔ çizilen yol uzunluğu, güven derecesi, uyarılar
```

### Algoritma kısaca

1. **Vektör okuma** – PDF'ler CAD çıktısı; yazılar bile font değil çizgi. Tüm çizim nesneleri
   renk/kalınlık bilgisiyle okunur (`pdfvector.py`; döndürülmüş sayfalar görüntü koordinatına çevrilir).
2. **Katman tanıma** (`klp.detect_style`) – kırmızı nesneler çizgi kalınlığına göre ayrılır:
   trase (kesikli, ~0,96), kablo kanalı (çift kesikli, ~0,51), kablo numaraları + kılavuz çizgileri (~0,74).
   Kalınlıklar istatistikle otomatik bulunur, sabit değildir.
3. **Trase grafı** (`trassegraph.py`) – her kesikli çizgi parçası kendi yönünde uzatılır (boşluklar kapanır),
   rasterleştirilir, iskeletlenir (skeletonize) ve graf çıkarılır. Açık uçlar yakındaki traseye bağlanır.
4. **Yazı okuma** (`textocr.py`) – çizgi yazılar satırlara gruplanır, eğimi (PCA) bulunur, yataya
   döndürülüp tek bir toplu görüntüde Tesseract ile okunur; düşük güvenli satırlar tek tek tekrar okunur.
5. **KLP etiketleri** – "S1307010 …" yığınları ve kılavuz çizgisinin trase üzerindeki ucu (anker) bulunur.
   Anlamı: *bu kablolar bu trase noktasından geçiyor*.
6. **KÜP ayrıştırma** (`kuep.py`) – her kablo numarasının altındaki yatay çizgi takip edilir; çizginin
   ucundaki kutu (KS/SK/KV) veya yanındaki yazı (13W22, 13P8 …) başlangıç/bitiş olur.
7. **İsim eşleştirme** – KÜP'teki `13W22/13G4004` KLP'de `W22/G4004`, `13L3045Y` → `3045Y`,
   `KS 1307000` → `KS 1307000` olarak aranır.
8. **Rota** (`router.py`) – kablonun tüm ankerleri + başlangıç/bitiş noktaları terminal kabul edilir,
   trase grafında Steiner ağacı (yaklaşık) hesaplanır → normalde tek bir çizgi. Eleman birden fazla
   traseye yakınsa, ankerlerden en az dolambaçla ulaşılan aday seçilir.
9. **Kontrol** – çizilen yol uzunluğu (ölçek 1:500) KÜP uzunluğu ile karşılaştırılır; sapma varsa uyarı.

## Kurulum

```bash
# Tesseract OCR gerekli
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki  (PATH'e ekleyin)
#   Linux:   sudo apt install tesseract-ocr
pip install -r requirements.txt      # veya: pip install .
```

## Kullanım

```bash
python -m cableroutefinder KLP.pdf --kuep KUEP.pdf -o ausgabe/
```

Örnek dosyalarla:

```bash
python -m cableroutefinder samples/10_15_SKL_04-500_KLP.pdf --kuep samples/10_16_SKL_A02_KUEP.pdf -o ausgabe/
```

Seçenekler:

| Parametre | Açıklama |
|---|---|
| `--kuep KUEP.pdf` | Kablo listesi ve başlangıç/bitiş buradan alınır. Verilmezse KLP'de etiketli tüm kablolar çizilir. |
| `--cables S1307010,S1307505` | Sadece bu kablolar |
| `--scale 500` | KLP ölçeği (uzunluk karşılaştırması için) |
| `--cableplan-json <workspace>/data/<Plan>.json` | Mevcut CablePlan dosyasına ekle (elle çizilmiş kablolar korunur) |
| `--overwrite` | Aynı isimli mevcut kabloların üzerine yaz |

Çıktılar (`ausgabe/`):

* `<KLP>_kabelwege.pdf` – kablo yolları çizili plan. Her kablo ayrı PDF katmanı ("Kabel S1307505"),
  Acrobat'ta tek tek açılıp kapatılabilir. Algılanan trase ağı da ayrı (kapalı) bir katman.
  Düz çizgi = yüksek güven, kesikli çizgi = kontrol edilmeli.
* `<KLP>.json` – CablePlan plan dosyası. Workspace'teki `data/` klasörüne PDF ile aynı isimle koyulur.
* `<KLP>_bericht.csv` – Excel'de açılabilen rapor (Kabel; Von; Nach; Länge KÜP; Länge Weg; Sicherheit; Hinweise).

## Örnek sonuç (samples/)

KÜP 10/16'daki 53 kablonun 48'i için yol çizildi. Çoğunda yol uzunluğu KÜP uzunluğuna çok yakın
(örn. S1307502: KÜP 130 m ↔ yol 129,6 m; S1307514: 170 m ↔ 160 m). Çizilemeyenler KLP'de hiç etiketi
olmayan, birkaç metrelik kısa bağlantı kabloları (S1307011, S1307111 …).

## Sınırlar / sonraki adımlar

* Sayfadan çıkan kablolar (ESTW-A, KS 1306000 …) sadece bu sayfadaki etiketlere kadar çizilir.
* Bazı eleman isimleri KLP'de yok ya da OCR okuyamıyor (ör. sinyal "P8"); raporda uyarı olarak görünür.
* KÜP uzunluğuyla uyuşmayan kablolar raporda işaretlenir – bunlar elle kontrol edilmeli
  (bazen KÜP'teki uzunluk da hatalı olabilir).
* CablePlan içine bir "Otomatik çiz" düğmesi eklenerek bu araç doğrudan çağrılabilir
  (ya da algoritma C#'a taşınabilir).

## Testler

```bash
pytest
```
