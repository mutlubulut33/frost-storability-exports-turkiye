# -*- coding: utf-8 -*-
"""
01_veri_hazirlama.py
Ham verilerden aylık analiz veri setini üretir.

Girdiler (ham_veri/):
  - HKS_2017_bugun_TAM.csv              : Hal Kayıt Sistemi günlük ürün fiyat/hacim verisi
  - EVDS_USD_alis_gunluk.xlsx           : TCMB EVDS, TP.DK.USD.A.YTL (USD döviz alış, günlük)
  - TUIK_ihracat_080810_2018_2022.xlsx  : TÜİK Dış Ticaret, Özel Ticaret Sistemi, GTİP 0808.10 aylık ihracat
  - TUIK_ihracat_080810_2023_2026.xlsx  : aynı, 2023-2026

Çıktılar (cikti/):
  - veri_hal_aylik.csv          : hal aylık fiyat/hacim (2017-01 .. 2026-09)
  - veri_ihracat_aylik.csv      : ihracat aylık değer/miktar/birim değer (2018-01 .. 2026-08)
  - veri_analiz.csv             : birleştirilmiş analiz veri seti (2018-01 .. 2026-08)

Çalıştırma:  python kod/01_veri_hazirlama.py   (proje kök klasöründen)
"""
from pathlib import Path
import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HAM = KOK / "ham_veri"
CIKTI = KOK / "cikti"
CIKTI.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1) HAL (HKS) VERİSİ
# ---------------------------------------------------------------------------
hks = pd.read_csv(HAM / "HKS_2017_bugun_TAM.csv", encoding="utf-8-sig")
hks["Tarih"] = pd.to_datetime(hks["Tarih"], format="%d.%m.%Y")

# Elma kayıtları: ürün adı "ELMA" ile başlayanlar; Trabzon hurması (CENNET ELMASI)
# ve yer elması ("ELMASI" içerenler) hariç.
elma = hks[hks["Ürün Adı"].str.startswith("ELMA")
           & ~hks["Ürün Adı"].str.contains("ELMASI")].copy()
n0 = len(elma)
# İşlem hacmi sıfır olan kayıtlar çıkarılır (fiyat bilgisi işleme dayanmıyor).
elma = elma[elma["İşlem Hacmi"] > 0]
# Organik ürünler çıkarılır (farklı fiyat segmenti, düşük hacim).
elma = elma[elma["Ürün Türü"] != "Organik Tarım"]
print(f"HKS elma kayıtları: {n0} -> temizlik sonrası {len(elma)}")

elma["deger"] = elma["Ortalama Fiyat"] * elma["İşlem Hacmi"]
CESIT = {"ELMA, STARKİNG": "Starking", "ELMA GOLDEN": "Golden",
         "ELMA GRANNY SMITH": "GrannySmith"}

hal = pd.DataFrame()
for kod, ad in CESIT.items():
    s = (elma[elma["Ürün Cinsi"] == kod]
         .groupby(pd.Grouper(key="Tarih", freq="MS"))[["deger", "İşlem Hacmi"]].sum())
    hal[f"hal_fiyat_{ad}_TL"] = s["deger"] / s["İşlem Hacmi"]      # hacim ağırlıklı TL/kg
    hal[f"hal_hacim_{ad}_ton"] = s["İşlem Hacmi"] / 1e3
uc = (elma[elma["Ürün Cinsi"].isin(CESIT)]
      .groupby(pd.Grouper(key="Tarih", freq="MS"))[["deger", "İşlem Hacmi"]].sum())
hal["hal_fiyat_3cesit_TL"] = uc["deger"] / uc["İşlem Hacmi"]
hal["hal_hacim_tum_ton"] = (elma.groupby(pd.Grouper(key="Tarih", freq="MS"))
                            ["İşlem Hacmi"].sum() / 1e3)
hal = hal.loc["2017-01":"2026-09"]          # Ekim 2026 eksik ay, dışarıda

# ---------------------------------------------------------------------------
# 2) DÖVİZ KURU (EVDS)
# ---------------------------------------------------------------------------
fx = pd.read_excel(HAM / "EVDS_USD_alis_gunluk.xlsx", skiprows=1, header=None,
                   usecols=[0, 1], names=["Tarih", "usdtry"])
fx["Tarih"] = pd.to_datetime(fx["Tarih"], format="%d-%m-%Y", errors="coerce")
fx["usdtry"] = pd.to_numeric(fx["usdtry"], errors="coerce")
fx = fx.dropna().set_index("Tarih")["usdtry"]
usd_aylik = fx.resample("MS").mean()        # aylık ortalama (akım verilerle uyumlu)
hal["usdtry"] = usd_aylik.reindex(hal.index)
for c in list(CESIT.values()) + ["3cesit"]:
    hal[f"hal_fiyat_{c}_USD"] = hal[f"hal_fiyat_{c}_TL"] / hal["usdtry"]

# ---------------------------------------------------------------------------
# 3) İHRACAT (TÜİK)
# ---------------------------------------------------------------------------
def tuik_oku(yol):
    d = pd.read_excel(yol, header=None, skiprows=10).iloc[:, :7]
    d.columns = ["yil", "ay", "gtip", "ad", "usd", "tl", "kg"]
    # Yıl ve ay yalnızca ilk satırda yazılı; aşağı doldur.
    d["yil"] = d["yil"].where(d["yil"].astype(str).str.fullmatch(r"\d{4}")).ffill()
    d["ay"] = d["ay"].where(d["ay"].astype(str).str.fullmatch(r"\d{1,2}")).ffill()
    # Sadece 12 haneli GTİP satırları (toplam satırları hariç)
    d = d[d["gtip"].astype(str).str.fullmatch(r"\d+")].copy()
    d["gtip"] = d["gtip"].astype(str).str.zfill(12)
    d[["yil", "ay"]] = d[["yil", "ay"]].astype(int)
    return d

ihr = pd.concat([tuik_oku(HAM / "TUIK_ihracat_080810_2018_2022.xlsx"),
                 tuik_oku(HAM / "TUIK_ihracat_080810_2023_2026.xlsx")])
assert not ihr.duplicated(["yil", "ay", "gtip"]).any(), "Yinelenen kayıt var"
GRUP = {"080810800011": "Golden",
        "080810800013": "Starking",   # Starking
        "080810800014": "Starking",   # Starkrimson -> Red Delicious grubu
        "080810800019": "Diger",
        "080810100000": "Diger"}      # dökme sanayilik elma (çok küçük)
bilinmeyen = set(ihr["gtip"]) - set(GRUP)
assert not bilinmeyen, f"Tanımsız GTİP: {bilinmeyen}"
ihr["grup"] = ihr["gtip"].map(GRUP)
ihr["Tarih"] = pd.to_datetime(dict(year=ihr.yil, month=ihr.ay, day=1))

top = ihr.groupby("Tarih")[["usd", "kg"]].sum()
gr = ihr.pivot_table(index="Tarih", columns="grup", values=["usd", "kg"], aggfunc="sum")
ix = pd.DataFrame(index=top.index)
ix["ihr_deger_usd"] = top["usd"]
ix["ihr_miktar_ton"] = top["kg"] / 1e3
ix["ihr_bd_USD"] = top["usd"] / top["kg"]                     # birim değer USD/kg
for g in ["Starking", "Golden", "Diger"]:
    ix[f"ihr_miktar_{g}_ton"] = gr[("kg", g)] / 1e3
    ix[f"ihr_bd_{g}_USD"] = gr[("usd", g)] / gr[("kg", g)]
    ix[f"ihr_pay_{g}"] = gr[("kg", g)] / top["kg"]
print(f"İhracat: {ix.index.min():%Y-%m} .. {ix.index.max():%Y-%m}, {len(ix)} ay")

# ---------------------------------------------------------------------------
# 4) BİRLEŞTİRME VE TÜRETİLEN DEĞİŞKENLER
# ---------------------------------------------------------------------------
d = hal.join(ix, how="inner")
d["oran_hal_ihr"] = d["hal_fiyat_3cesit_USD"] / d["ihr_bd_USD"]
d["lnH"] = np.log(d["hal_fiyat_3cesit_USD"])      # bağımlı değişken
d["lnX"] = np.log(d["ihr_bd_USD"])                # açıklayıcı: ihracat birim değeri
d["lnER"] = np.log(d["usdtry"])                    # kontrol: kur
d["lnXvol"] = np.log(d["ihr_miktar_ton"])
# Depolama dönemi: Kasım-Mayıs = 1 ; hasat/yaz (Haziran-Ekim) = 0
d["depolama"] = d.index.month.isin([11, 12, 1, 2, 3, 4, 5]).astype(int)
# Don dönemi kuklası: Nisan 2025 ve sonrası
d["don"] = (d.index >= "2025-04-01").astype(int)
assert d[["lnH", "lnX", "lnER"]].notna().all().all(), "Eksik değer var"

def yaz(df, ad):
    o = df.copy(); o.index = o.index.strftime("%Y-%m"); o.index.name = "Ay"
    o.to_csv(CIKTI / ad, encoding="utf-8-sig", float_format="%.6f")

yaz(hal, "veri_hal_aylik.csv")
yaz(ix, "veri_ihracat_aylik.csv")
yaz(d, "veri_analiz.csv")
print(f"Analiz veri seti: {d.index.min():%Y-%m} .. {d.index.max():%Y-%m}, {len(d)} ay, {d.shape[1]} değişken")
