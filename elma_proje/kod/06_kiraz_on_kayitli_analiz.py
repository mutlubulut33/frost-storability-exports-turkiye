# -*- coding: utf-8 -*-
"""
06_kiraz_on_kayitli_analiz.py
ON_KAYIT_kiraz_analiz_plani.md dosyasındaki planın birebir uygulaması (11.10.2026 18:22'de sabitlendi).

Sezon: Mayıs, Haziran, Temmuz (2018-2026, 27 gözlem)
Model: D_t = ln y_KİRAZ,t − ln y_KONTROL,t = a + ay kuklaları + β2025·S2025 + β2026·S2026 + ε ; HAC(2)
Sonuçlar: ln ihracat miktarı, ln ihracat birim değeri, üçlü fark ln(İhr/Hal)
Kiraz: ana = 080929 ; sağlamlık = 080921 + 080929
Kontrol: ana = narenciye ihracatı (05 betiği tanımı) ; alternatif = karpuz + kavun ihracatı (080711+080719)
Üçlü fark yalnızca narenciye kontrolüyle (hal verisi narenciye için mevcut).
Plasebo: 2019..2024 sezonları sırayla sahte tedavi (yalnızca 2018-2024 verisi).
Çalıştırma: python kod/06_kiraz_on_kayitli_analiz.py  (01, 04, 05 önce çalıştırılmış olmalı)
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortak import KOK, TABLO, SEKIL
warnings.filterwarnings("ignore")
HAM, CIKTI = KOK / "ham_veri", KOK / "cikti"
SEZON_AY = [5, 6, 7]

def tuik_hs6(yol):
    d = pd.read_excel(yol, header=None, skiprows=10).iloc[:, :7]
    d.columns = ["yil", "ay", "kod", "ad", "usd", "tl", "kg"]
    d["yil"] = d["yil"].where(d["yil"].astype(str).str.fullmatch(r"\d{4}")).ffill()
    d["ay"] = d["ay"].where(d["ay"].astype(str).str.fullmatch(r"\d{1,2}")).ffill()
    d = d[d["kod"].astype(str).str.fullmatch(r"\d+")].copy()
    d["kod"] = d["kod"].astype(str).str.zfill(6)
    d[["yil", "ay"]] = d[["yil", "ay"]].astype(int)
    d["Tarih"] = pd.to_datetime(dict(year=d.yil, month=d.ay, day=1))
    return d

k = pd.concat([tuik_hs6(HAM / "TUIK_ihracat_kiraz_kavun_karpuz_HS6_2018_2022.xlsx"),
               tuik_hs6(HAM / "TUIK_ihracat_kiraz_kavun_karpuz_HS6_2023_2026.xlsx")])
assert not k.duplicated(["yil", "ay", "kod"]).any()
p = k.pivot_table(index="Tarih", columns="kod", values=["usd", "kg"], aggfunc="sum")
idx = pd.date_range("2018-01-01", "2026-08-01", freq="MS")
p = p.reindex(idx).fillna(0.0)          # TÜİK'te kaydı olmayan ay = 0 ihracat

d = pd.DataFrame(index=idx)
d["kiraz_ton"] = p[("kg", "080929")] / 1e3
d["kiraz_bd"] = p[("usd", "080929")] / p[("kg", "080929")]
d["kiraz_visne_ton"] = (p[("kg", "080929")] + p[("kg", "080921")]) / 1e3
d["kiraz_visne_bd"] = (p[("usd", "080929")] + p[("usd", "080921")]) / (p[("kg", "080929")] + p[("kg", "080921")])
d["kavun_karpuz_ton"] = (p[("kg", "080711")] + p[("kg", "080719")]) / 1e3
d["kavun_karpuz_bd"] = (p[("usd", "080711")] + p[("usd", "080719")]) / (p[("kg", "080711")] + p[("kg", "080719")])
nar = pd.read_csv(CIKTI / "veri_ihracat_did.csv", index_col=0, encoding="utf-8-sig")
nar.index = pd.PeriodIndex(nar.index, freq="M").to_timestamp()
d = d.join(nar[["nar_ihr_ton", "nar_bd", "nar_hal_ton"]])
hal = pd.read_csv(CIKTI / "veri_did_panel.csv", index_col=0, encoding="utf-8-sig")
hal.index = pd.PeriodIndex(hal.index, freq="M").to_timestamp()
d["kiraz_hal_ton"] = hal["hacim_ton_KİRAZ"]

s = d[d.index.month.isin(SEZON_AY)].copy()
assert len(s) == 27, len(s)
assert (s.filter(like="_ton") > 0).all().all(), "Sezon aylarında sıfır değer var"
print("Kiraz sezonu ihracatı (ton):")
print(s["kiraz_ton"].groupby(s.index.year).sum().round(0).to_string())

FARK = {
    ("ln İhracat miktarı", "Kiraz (080929)", "Narenciye"): np.log(s.kiraz_ton) - np.log(s.nar_ihr_ton),
    ("ln İhracat miktarı", "Kiraz (080929)", "Kavun+karpuz"): np.log(s.kiraz_ton) - np.log(s.kavun_karpuz_ton),
    ("ln İhracat miktarı", "Kiraz+vişne", "Narenciye"): np.log(s.kiraz_visne_ton) - np.log(s.nar_ihr_ton),
    ("ln İhracat birim değeri", "Kiraz (080929)", "Narenciye"): np.log(s.kiraz_bd) - np.log(s.nar_bd),
    ("ln İhracat birim değeri", "Kiraz (080929)", "Kavun+karpuz"): np.log(s.kiraz_bd) - np.log(s.kavun_karpuz_bd),
    ("Üçlü fark ln(İhr/Hal)", "Kiraz (080929)", "Narenciye"):
        (np.log(s.kiraz_ton) - np.log(s.kiraz_hal_ton)) - (np.log(s.nar_ihr_ton) - np.log(s.nar_hal_ton)),
}

def tasarim(idx, kuklalar):
    X = pd.DataFrame({"const": 1.0}, index=idx)
    X["ay06"] = (idx.month == 6).astype(float); X["ay07"] = (idx.month == 7).astype(float)
    for ad, yil in kuklalar.items():
        X[ad] = (idx.year == yil).astype(float)
    return X

sat = []
for (sonuc, tedavi, kontrol), f in FARK.items():
    X = tasarim(f.index, {"S2025": 2025, "S2026": 2026})
    r = sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    # Plasebo: yalnızca 2018-2024, her sezon sırayla sahte tedavi
    f0 = f[f.index.year <= 2024]
    plas = []
    for y in range(2019, 2025):
        rp = sm.OLS(f0, tasarim(f0.index, {"S": y})).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        plas.append(rp.params["S"])
    plas = np.array(plas)
    for c in ["S2025", "S2026"]:
        b = r.params[c]
        sat.append({"Sonuç": sonuc, "Tedavi": tedavi, "Kontrol": kontrol, "Sezon": c[1:],
                    "N": int(r.nobs), "β": b, "Std.hata (HAC)": r.bse[c], "p": r.pvalues[c],
                    "%95 GA alt": r.conf_int().loc[c, 0], "%95 GA üst": r.conf_int().loc[c, 1],
                    "% etki": (np.exp(b) - 1) * 100,
                    "Plasebo β aralığı (2019-24)": f"{plas.min():.3f} .. {plas.max():.3f}",
                    "|plasebo| ≥ |β| sayısı (6 içinde)": int((np.abs(plas) >= abs(b)).sum())})
K = pd.DataFrame(sat)
pd.set_option("display.width", 250)
print("\nSonuçlar:"); print(K.round(3).to_string(index=False))

# Sezon-yılı farkları (görsel)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for a, key in zip(axs, [("ln İhracat miktarı", "Kiraz (080929)", "Narenciye"),
                         ("ln İhracat birim değeri", "Kiraz (080929)", "Narenciye"),
                         ("Üçlü fark ln(İhr/Hal)", "Kiraz (080929)", "Narenciye")]):
    f = FARK[key]
    # mevsim etkisini ay ortalamalarından arındırıp yıllık ortalama (2018-2024 ort. = 0)
    on = f[f.index.year <= 2024]
    ay_ort = on.groupby(on.index.month).mean()
    dm = f - ay_ort.reindex(f.index.month).values
    yil = dm.groupby(dm.index.year).mean()
    renk = ["#c05621" if y == 2025 else ("#2b6cb0" if y == 2026 else "#a0aec0") for y in yil.index]
    a.bar(yil.index.astype(str), yil.values, color=renk)
    a.axhline(0, color="black", lw=.6); a.set_title(f"Kiraz − narenciye: {key[0]}", fontsize=9)
    a.tick_params(axis="x", labelsize=8, rotation=45); a.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Mayıs-Temmuz ortalaması; 2018-2024 ay ortalamalarından sapma. Turuncu: 2025 (don), mavi: 2026.", fontsize=7)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(SEKIL / "sekil8_kiraz_sezon_farklari.png", dpi=200); plt.close()

o = d.copy(); o.index = o.index.strftime("%Y-%m"); o.index.name = "Ay"
o.to_csv(CIKTI / "veri_kiraz.csv", encoding="utf-8-sig", float_format="%.6f")
with pd.ExcelWriter(TABLO / "tablolar_kiraz_on_kayitli.xlsx") as xw:
    K.to_excel(xw, sheet_name="kiraz_sonuclar", index=False)
print("\nKaydedildi.")
