# Ön kayıt: Kiraz analizi planı

**Yazılma zamanı:** 11 Ekim 2026, 18:22 (UTC+3), kiraz ihracat verisi indirilmeden ve incelenmeden önce.

**Bilinenler:** Kiraz için yalnızca HKS hal verisi daha önce görüldü (04 betiğindeki doğrulama: 2025'te hal hacmi yaklaşık −%85; farkların farkı β = −0,60, p = 0,10). İhracat verisi görülmedi.

## Amaç
Elmada bulunan örüntünün (ihracatın iç piyasaya göre orantısız daralması) ikinci bir üründe tekrarlanıp tekrarlanmadığını görmek ve depolamanın şok etkisini geciktirip geciktirmediğini test etmek.

## Hipotezler (yön önceden belirlenmiştir)
- **K1:** Kiraz ihracat miktarı, 2025 sezonunda (Mayıs–Temmuz 2025) kontrol grubuna göre düşer (β < 0).
- **K2 (üçlü fark):** Kirazda ihracat, iç piyasaya (hal) göre ve kontrole kıyasla 2025 sezonunda daha çok daralır (β < 0).
- **K3 (depolama/zamanlama):** Depolanamayan kirazda etki don yılının kendi sezonunda (2025) görülür. Depolanabilir elmada Nisan–Temmuz 2025'te etki yoktur, etki 2025/26 hasadıyla başlar. (Elma tarafı zaten gözlendi; ön kayıt kiraz tarafı içindir.)
- **K4 (toparlanma):** Don yalnızca 2025'i etkilediği için 2026 kiraz sezonunda (Mayıs–Temmuz 2026) etki belirgin biçimde küçülür veya kaybolur. Etki 2026'da da aynı büyüklükte sürerse, bu dondan başka bir nedene işaret eder ve öyle raporlanır.

## Veri
- TÜİK, Özel Ticaret Sistemi, HS6, aylık, USD ve kg, 2018-01 .. son ay:
  - 080921 vişne (ekşi kiraz), 080929 kiraz (diğer). Ana analiz: 080929. Sağlamlık: 080921 + 080929.
  - Yaz kontrolü (isteğe bağlı): 080711 karpuz, 080719 kavun.
- HKS: KİRAZ (tüm alt ürün adları).
- Kontrol (ana): narenciye ihracatı (05 betiğindeki tanım). Kontrol (alternatif): karpuz + kavun ihracatı, indirilirse.

## Tasarım
- Kiraz sezonu: **Mayıs, Haziran, Temmuz** (yalnızca bu aylar kullanılır). Her yıl 3 ay; 2018–2026 için 27 gözlem.
- Model: D_t = ln y_KİRAZ,t − ln y_KONTROL,t = a + ay kuklaları + β₂₀₂₅·S2025_t + β₂₀₂₆·S2026_t + ε, Newey-West HAC (2 gecikme).
  - S2025 = 2025 Mayıs–Temmuz, S2026 = 2026 Mayıs–Temmuz. Referans: 2018–2024 sezonları.
- Sonuç değişkenleri: ln ihracat miktarı, ln ihracat birim değeri, üçlü fark ln(İhr/Hal).
- Plasebo: 2019..2024'ün her sezonu sırayla "sahte tedavi" olarak (gerçek don sonrası veriler hariç).
- Gözlem sayısı az olduğu için p-değerlerine ek olarak gerçek β'nın plasebo dağılımındaki sırası raporlanır.
- Model tanımı bu plandan sonra değiştirilirse değişiklik ve gerekçesi bu dosyaya eklenir; iki sonuç birlikte raporlanır.
## Uygulama notu (11.10.2026)
Plan değiştirilmeden uygulanmıştır (kod/06_kiraz_on_kayitli_analiz.py). TÜİK'te kaydı olmayan aylar 0 ihracat kabul edildi; sezon aylarında (Mayıs-Temmuz) sıfır değer yoktur.

## Sonradan eklenen not (20.10.2026) — plan değiştirilmemiştir
Ön kayıtlı analizden sonra iki bilgi ortaya çıktı: (i) Tebliğ (İhracat: 2025/6) ile elma ve limon ihracatı kayda bağlandı; limon narenciye kontrol grubunun içindedir. (ii) Mayıs–Temmuz aylarında limon dışı narenciye ihracatı çok düşüktür (çoğu ay birkaç yüz ton) ve 2025 yazında da düşmüştür. Bu nedenle narenciyenin kiraz sezonu için zayıf bir kontrol olduğu sonradan değerlendirilmiştir. Ön kayıtlı sonuçlar (06 betiği) değiştirilmeden raporlanır; limonsuz narenciye ile yapılan ek tahminler 08 betiğinde (08_limonsuz_narenciye.py), "sonradan eklenen analiz" olarak ayrıca raporlanır.
