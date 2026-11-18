# Elma ve kiraz: 2025 zirai donu ve tedarik zinciri — veri ve kodlar

Bu klasördeki tüm tablolar ve şekiller, `ham_veri/` içindeki resmî verilerden aşağıdaki betiklerle yeniden üretilir. Kodlarda elle girilmiş sonuç ya da sentetik veri yoktur. Tek simülasyon, sınır testi kritik değerleri içindir (`ortak.pss_simulasyon`, sabit tohum 20261011); doğruluğu PSS (2001) tablolarıyla karşılaştırılarak kontrol edilmiştir (`tablolar_ana_analiz.xlsx`, `Ek_PSS_dogrulama` sayfası).

## 1. Kurulum

Python 3.13.16 ile test edilmiştir. Gerekli paketler:

```
pip install numpy==2.5.3 pandas==3.0.5 scipy==1.18.1 statsmodels==0.15.0 matplotlib==3.11.2 openpyxl==3.1.5
```

Daha eski sürümlerle de büyük olasılıkla çalışır, ancak sonuçlar yalnızca bu sürümlerle doğrulanmıştır.

## 2. Ham veriler (`ham_veri/`)

| Dosya | Kaynak | İçerik | SHA-256 (ilk 12 karakter) |
|---|---|---|---|
| `HKS_2017_bugun_TAM.csv` | Ticaret Bakanlığı Hal Kayıt Sistemi | Günlük ürün, çeşit ve tür bazında ortalama fiyat (TL/kg) ve işlem hacmi, 01.01.2017–03.10.2026 | 8d2a9635c19a |
| `EVDS_USD_alis_gunluk.xlsx` | TCMB EVDS, TP.DK.USD.A.YTL | Günlük ABD doları döviz alış kuru | e54ddc0e092b |
| `TUIK_ihracat_080810_2018_2022.xlsx`, `..._2023_2026.xlsx` | TÜİK Dış Ticaret Veri Tabanı, Özel Ticaret Sistemi | Elma (GTİP 0808.10, 12'li kodlar), aylık ihracat USD ve kg | 94af01e6f5b9 / 66cb10fc8f41 |
| `TUIK_ihracat_0805_HS6_2018_2022.xlsx`, `..._2023_2026.xlsx` | Aynı | Narenciye (HS6: 080510, 080521, 080522, 080529, 080540, 080550, 080590) | 5069b387c0d0 / bbbcf44dcc32 |
| `TUIK_ihracat_kiraz_kavun_karpuz_HS6_2018_2022.xlsx`, `..._2023_2026.xlsx` | Aynı | Kiraz 080929, vişne 080921, karpuz 080711, kavun 080719 | 97ca9782532f / a4ba0bdf9cc4 |
| `TUIK_ihracat_patates_070190_HS6_*.xlsx`, `TUIK_ihracat_sogan_070310_HS6_*.xlsx` | Aynı | Patates 070190, kuru soğan 070310 (2018–2022 ve 2023–2026) | `sha256sum` ile |

TÜİK dosyaları rapor tarihi 11.10.2026 olan, indirildiği gibi bırakılmış dosyalardır. HKS dosyası büyük olduğu (95 MB) için pakete eklenmemiştir; araştırmacıda mevcuttur ve `ham_veri/` klasörüne konmalıdır. Tam SHA-256 değerleri için: `sha256sum ham_veri/*`.

## 3. Çalıştırma sırası

Proje kök klasöründen sırayla çalıştırın. Toplam süre 1 dakikanın altındadır.

```
python kod/01_veri_hazirlama.py
python kod/02_ekonometrik_analiz.py
python kod/03_don_kirilma.py
python kod/04_kontrol_grubu_did.py
python kod/05_ihracat_did.py
python kod/06_kiraz_on_kayitli_analiz.py
python kod/07_patates_sogan_kontrol.py
python kod/08_limonsuz_narenciye.py
python kod/09_makale_tablolari.py
```

`kod/ortak.py` bir yardımcı modüldür; doğrudan çalıştırılmaz.

## 4. Betikler ve ürettikleri

| Betik | Ne yapar | Çıktılar |
|---|---|---|
| `01_veri_hazirlama.py` | HKS elma kayıtlarını temizler (sıfır hacim ve organik hariç), aylık hacim ağırlıklı fiyat hesaplar; kuru aylık ortalamaya çevirir; TÜİK elma ihracatını okur ve birim değer hesaplar; hepsini birleştirir | `veri_hal_aylik.csv`, `veri_ihracat_aylik.csv`, `veri_analiz.csv` |
| `02_ekonometrik_analiz.py` | Don öncesi (2018–2024) fiyat aktarımı: birim kök, ARDL/NARDL sınır testi, uzun/kısa dönem katsayılar, asimetri, depolama etkisi, tanı testleri, Huber/HAC sağlamlık, çeşit düzeyi, alternatif tanım (T10) | `tablolar_ana_analiz.xlsx` (T1–T10), `ozet_*.txt`, `rapor_02_ana_analiz.txt`, Şekil 2–3 |
| `03_don_kirilma.py` | Don etkisi regresyonları (kontrol grubu yok), kestirim Chow testi, don etkileşimleri, tüm örneklem CUSUM | `tablolar_don_kirilma.xlsx` (K1–K5), Şekil 4–5 |
| `04_kontrol_grubu_did.py` | Hal verisiyle farkların farkı: elma ve kontrol ürünleri (narenciye; patates ve soğan); olay çalışması; zamanda ve ürünlerde plasebo; doğrulama ürünleri (kiraz, şeftali, erik) | `tablolar_did.xlsx` (D1–D4), `veri_did_panel.csv`, Şekil 6 |
| `05_ihracat_did.py` | İhracatta farkların farkı (elma ve narenciye) ve üçlü fark (ihracat/hal) | `tablolar_ihracat_did.xlsx` (E1–E4), `veri_ihracat_did.csv`, Şekil 7 |
| `06_kiraz_on_kayitli_analiz.py` | `ON_KAYIT_kiraz_analiz_plani.md` dosyasındaki ön kayıtlı planın birebir uygulaması | `tablolar_kiraz_on_kayitli.xlsx`, `veri_kiraz.csv`, Şekil 8 |
| `08_limonsuz_narenciye.py` | **Sonradan eklenen analiz (20.10.2026).** Tebliğ (İhracat: 2025/6; RG 11.09.2025, sayı 33014) elma ve limon ihracatını kayda bağladığı için 04–06'daki tahminleri limonsuz narenciye kontrolüyle tekrarlar | `tablolar_limonsuz_narenciye.xlsx` |
| `09_makale_tablolari.py` | Makaledeki İngilizce Tablo 1–5 ve Şekil 1–3'ü sonuç dosyalarından üretir (elle sayı girilmez); `tabulate` paketi gerekir | `cikti/makale/` |
| `07_patates_sogan_kontrol.py` | Patates ve soğan ihracatının ek kontrol olarak uygunluk testi (aylık DiD için uygun bulunmadı) ve pazarlama yılı bazında betimsel karşılaştırma | `tablolar_patates_sogan.xlsx` |

## 5. Önemli tanımlar ve kararlar

- **Hal fiyatı (fiyat aktarımı analizi):** Starking, Golden ve Granny Smith çeşitlerinin aylık hacim ağırlıklı ortalaması, aylık ortalama kurla USD'ye çevrilmiş. Farkların farkı analizlerinde tüm elma ürün adları kullanılır ve fiyatlar TL cinsindendir; ortak şokları zaman farkı giderir.
- **Örneklemler:** HKS kapsamı 2017'de düşük olduğu için analizler 2018-01'de başlar. TÜİK ihracat verisi 2026-08'de, HKS verisi 2026-09'da biter.
- **Don sonrası tanımları:** POST1, Nisan 2025 ve sonrası; POST2, Ağustos 2025 ve sonrası (yeni hasat). İki tanım da raporlanır.
- **Pazarlama yılı:** Ağustos–Temmuz. 2026/27 yalnızca Ağustos–Eylül 2026'yı içerir.
- **Standart hatalar:** Farkların farkında Newey-West HAC (aylıkta 6, kiraz sezonunda 2 gecikme).
- **Sınır testi:** statsmodels `UECM.bounds_test` kullanılmamıştır; mevsim kuklası içermeyen ayrı bir model tahmin ediyor ve regresör sayısını (k) bağımlı değişken dahil sayıyor. Bunun yerine F testi doğrudan hesaplanır, kritik değerler aynı n ve k için simülasyonla üretilir.
- **Sonradan eklenen tanım:** 02'deki T10 (kurun yalnızca kısa dönemde yer aldığı tanım) ana tanımdan sonra denenmiştir; iki tanım birlikte raporlanmalıdır.
- **Kiraz analizi:** Plan, ihracat verisi görülmeden önce yazılmış ve değiştirilmeden uygulanmıştır (bkz. ön kayıt dosyası).

## 6. Bilinen sınırlar

- Don sonrası dönem tek bir hasat sezonunu kapsar.
- Her üründe tek bir tedavi birimi vardır; çıkarım zaman serisi HAC standart hatalarına ve plasebo testlerine dayanır.
- İhracat birim değerleri bileşim etkilerini içerir; düşük hacimli aylarda uç değerler görülür.
- Hal verisi, hal dışından geçen satışları kapsamaz.

## 7. Ön kayıt dosyaları

- `ON_KAYIT_kiraz_analiz_plani_ORIJINAL_2026-10-11.md`: Kiraz ihracat verisi indirilmeden önce yazılan planın değiştirilmemiş hâli (11.10.2026, 18:22). SHA-256: `51264e38632ab55902c4fb0003b96f276bb935ef73fe6075aafa657b580fe2c1`.
- `ON_KAYIT_kiraz_analiz_plani.md`: Aynı plan ve sonuna eklenen tarihli uygulama notları. Planın kendisi değiştirilmemiştir; ilk bölüm orijinal dosyayla birebir aynıdır.
