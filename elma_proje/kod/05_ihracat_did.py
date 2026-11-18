# -*- coding: utf-8 -*-
"""
05_ihracat_did.py
İhracat tarafında farkların farkı (DiD) ve "ihracat tampon mu?" üçlü fark (DDD) testi.

Tedavi  : Elma ihracatı (GTİP 0808.10, 12'li kodların toplamı; 01 betiğinin çıktısı)
Kontrol : Narenciye ihracatı (HS6: 080510 portakal, 080521/080522/080529 mandalina grubu,
          080540 greyfurt, 080550 limon; 080590 dahil edilmez)
Dönem   : 2018-01 .. 2026-08

E1  DiD:  D_t = ln y_ELMA,t − ln y_NARENCİYE,t = a + ay kuklaları + β·POST_t + ε  (HAC 6)
    y: ihracat miktarı (ton) ve ihracat birim değeri (USD/kg)
    Kontrol tanımları: (i) narenciye toplamı, (ii) dört grubun log ortalaması
E2  Olay çalışması: pazarlama yılı (Ağustos-Temmuz) kuklaları, referans 2023/24
E3  Zamanda plasebo: sahte tedavi 2019..2024 Nisan
E4  Üçlü fark (DDD): ihracat kanalının iç piyasaya göre ek daralması
    DDD_t = [ln İhr_ELMA − ln Hal_ELMA] − [ln İhr_NAR − ln Hal_NAR]
    β < 0  => dondan sonra elmada ihracat, iç piyasaya (hal) göre narenciyeye kıyasla daha çok daraldı
Çalıştırma: python kod/05_ihracat_did.py   (01 ve 04 betikleri önce çalıştırılmış olmalı)
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
POST = {"POST1 (≥2025-04)": "2025-04-01", "POST2 (≥2025-08)": "2025-08-01"}

# ---------------------------------------------------------------------------
# Narenciye ihracatı
# ---------------------------------------------------------------------------
def tuik_hs6(yol):
    d = pd.read_excel(yol, header=None, skiprows=10).iloc[:, :7]
    d.columns = ["yil", "ay", "kod", "ad", "usd", "tl", "kg"]
    d["yil"] = d["yil"].where(d["yil"].astype(str).str.fullmatch(r"\d{4}")).ffill()
    d["ay"] = d["ay"].where(d["ay"].astype(str).str.fullmatch(r"\d{1,2}")).ffill()
    d = d[d["kod"].astype(str).str.fullmatch(r"\d+")].copy()
    d["kod"] = d["kod"].astype(str).str.zfill(6)
    d[["yil", "ay"]] = d[["yil", "ay"]].astype(int)
    return d
nar = pd.concat([tuik_hs6(HAM / "TUIK_ihracat_0805_HS6_2018_2022.xlsx"),
                 tuik_hs6(HAM / "TUIK_ihracat_0805_HS6_2023_2026.xlsx")])
assert not nar.duplicated(["yil", "ay", "kod"]).any()
GRUP = {"080510": "Portakal", "080521": "Mandalina", "080522": "Mandalina",
        "080529": "Mandalina", "080540": "Greyfurt", "080550": "Limon", "080590": None}
assert set(nar["kod"]) <= set(GRUP), set(nar["kod"]) - set(GRUP)
nar["grup"] = nar["kod"].map(GRUP)
nar = nar[nar["grup"].notna()]
nar["Tarih"] = pd.to_datetime(dict(year=nar.yil, month=nar.ay, day=1))
ng = nar.pivot_table(index="Tarih", columns="grup", values=["usd", "kg"], aggfunc="sum")
nt = nar.groupby("Tarih")[["usd", "kg"]].sum()
print(f"Narenciye ihracatı: {nt.index.min():%Y-%m} .. {nt.index.max():%Y-%m}, {len(nt)} ay")

# Elma ihracatı ve hal hacimleri
el = pd.read_csv(CIKTI / "veri_ihracat_aylik.csv", index_col=0, encoding="utf-8-sig")
el.index = pd.PeriodIndex(el.index, freq="M").to_timestamp()
hal = pd.read_csv(CIKTI / "veri_did_panel.csv", index_col=0, encoding="utf-8-sig")
hal.index = pd.PeriodIndex(hal.index, freq="M").to_timestamp()

d = pd.DataFrame(index=el.index)
d["elma_ihr_ton"] = el["ihr_miktar_ton"]
d["elma_bd"] = el["ihr_bd_USD"]
d["nar_ihr_ton"] = nt["kg"] / 1e3
d["nar_bd"] = nt["usd"] / nt["kg"]
for g in ["Portakal", "Mandalina", "Greyfurt", "Limon"]:
    d[f"{g}_ihr_ton"] = ng[("kg", g)] / 1e3
    d[f"{g}_bd"] = ng[("usd", g)] / ng[("kg", g)]
d["elma_hal_ton"] = hal["hacim_ton_ELMA"]
d["nar_hal_ton"] = hal[[f"hacim_ton_{u}" for u in ["PORTAKAL", "MANDALİNA", "LİMON", "GREYFURT"]]].sum(axis=1)
# Not: mandalinanın 2018-06 ve 2018-07'de ihracatı yok (TÜİK'te kayıt yok). Bu iki ay yalnızca
# "4 grubun log ortalaması" tanımından düşer; toplam narenciye tanımında tüm aylar kullanılır.
ana = ["elma_ihr_ton", "elma_bd", "nar_ihr_ton", "nar_bd", "elma_hal_ton", "nar_hal_ton"]
assert d[ana].notna().all().all() and (d[ana] > 0).all().all(), "Ana serilerde eksik/sıfır var"
print("Eksik değerli aylar (alt gruplar):", d.index[d.isna().any(axis=1)].strftime("%Y-%m").tolist())
print(f"Ortak örneklem: {d.index.min():%Y-%m} .. {d.index.max():%Y-%m}, {len(d)} ay")

GR4 = ["Portakal", "Mandalina", "Greyfurt", "Limon"]
FARK0 = {
    ("ln İhracat miktarı", "Narenciye toplamı"): np.log(d.elma_ihr_ton) - np.log(d.nar_ihr_ton),
    ("ln İhracat miktarı", "4 grubun log ortalaması"): np.log(d.elma_ihr_ton) - np.log(d[[f"{g}_ihr_ton" for g in GR4]]).mean(axis=1),
    ("ln İhracat birim değeri", "Narenciye toplamı"): np.log(d.elma_bd) - np.log(d.nar_bd),
    ("ln İhracat birim değeri", "4 grubun log ortalaması"): np.log(d.elma_bd) - np.log(d[[f"{g}_bd" for g in GR4]]).mean(axis=1),
    ("DDD: ln(İhr/Hal) elma − narenciye", "Narenciye toplamı"):
        (np.log(d.elma_ihr_ton) - np.log(d.elma_hal_ton)) - (np.log(d.nar_ihr_ton) - np.log(d.nar_hal_ton)),
}

FARK = {k: v.dropna() for k, v in FARK0.items()}

def ay_kukla(idx):
    X = pd.DataFrame({"const": 1.0}, index=idx)
    for m in range(2, 13):
        X[f"ay{m:02d}"] = (idx.month == m).astype(float)
    return X
def did(f, pb):
    X = ay_kukla(f.index); X["POST"] = (f.index >= pb).astype(float)
    return sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

# E1 / E4
sat = []
for (sonuc, kontrol), f in FARK.items():
    for pad, pb in POST.items():
        r = did(f, pb); b = r.params["POST"]
        sat.append({"Sonuç": sonuc, "Kontrol": kontrol, "Post": pad, "N": int(r.nobs), "β": b,
                    "Std.hata (HAC)": r.bse["POST"], "p": r.pvalues["POST"],
                    "%95 GA alt": r.conf_int().loc["POST", 0], "%95 GA üst": r.conf_int().loc["POST", 1],
                    "% etki": (np.exp(b) - 1) * 100})
E1 = pd.DataFrame(sat)
print("\nE1/E4 DiD ve DDD:"); print(E1.round(3).to_string(index=False))

# E2 Olay çalışması
def py(idx):
    y = np.where(idx.month >= 8, idx.year, idx.year - 1)
    return pd.Series([f"{a}/{str(a+1)[-2:]}" for a in y], index=idx)
REF = "2023/24"; ol = []
for key in [("ln İhracat miktarı", "Narenciye toplamı"), ("ln İhracat birim değeri", "Narenciye toplamı"),
            ("DDD: ln(İhr/Hal) elma − narenciye", "Narenciye toplamı")]:
    f = FARK[key]; p_ = py(f.index); X = ay_kukla(f.index)
    for y in sorted(p_.unique()):
        if y != REF: X[f"PY_{y}"] = (p_ == y).astype(float)
    r = sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    for c in [c for c in X if c.startswith("PY_")] + ["REF"]:
        if c == "REF":
            ol.append({"Sonuç": key[0], "Pazarlama yılı": REF, "γ": 0, "GA alt": 0, "GA üst": 0, "p": np.nan}); continue
        ol.append({"Sonuç": key[0], "Pazarlama yılı": c[3:], "γ": r.params[c], "p": r.pvalues[c],
                   "GA alt": r.conf_int().loc[c, 0], "GA üst": r.conf_int().loc[c, 1]})
E2 = pd.DataFrame(ol); E2["s"] = E2["Pazarlama yılı"].str[:4].astype(int)
E2 = E2.sort_values(["Sonuç", "s"]).drop(columns="s")
print("\nE2 Olay çalışması:"); print(E2.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for a, s in zip(axs, ["ln İhracat miktarı", "ln İhracat birim değeri", "DDD: ln(İhr/Hal) elma − narenciye"]):
    g = E2[E2["Sonuç"] == s]; x = np.arange(len(g))
    a.errorbar(x, g["γ"], yerr=[g["γ"] - g["GA alt"], g["GA üst"] - g["γ"]], fmt="o", color="#c05621", capsize=3, ms=4)
    a.set_xticks(x); a.set_xticklabels([e if e != "2026/27" else "2026/27*" for e in g["Pazarlama yılı"]], rotation=45, fontsize=8)
    a.axhline(0, color="black", lw=.6); a.axvline(len(g) - 2.5, color="red", ls=":", lw=.9)
    a.set_title(f"Elma − narenciye: {s}", fontsize=9); a.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Referans: 2023/24. %95 GA (HAC). * 2026/27 yalnızca Ağustos 2026.", fontsize=7)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(SEKIL / "sekil7_ihracat_did_olay.png", dpi=200); plt.close()

# E3 Zamanda plasebo
pl = []
for key in [("ln İhracat miktarı", "Narenciye toplamı"), ("DDD: ln(İhr/Hal) elma − narenciye", "Narenciye toplamı")]:
    f_t = FARK[key]
    gercek = did(f_t, POST["POST1 (≥2025-04)"]).params["POST"]
    for yil in range(2019, 2025):
        sahte = pd.Timestamp(f"{yil}-04-01")
        son = min(sahte + pd.DateOffset(months=16), pd.Timestamp("2025-03-01"))
        r = did(f_t.loc[:son], sahte)
        pl.append({"Sonuç": key[0], "Sahte tedavi": f"{yil}-04", "β": r.params["POST"], "p": r.pvalues["POST"],
                   "Gerçek β (2025-04)": gercek})
E3 = pd.DataFrame(pl)
print("\nE3 Zamanda plasebo:"); print(E3.round(3).to_string(index=False))

d_out = d.copy(); d_out.index = d_out.index.strftime("%Y-%m"); d_out.index.name = "Ay"
d_out.to_csv(CIKTI / "veri_ihracat_did.csv", encoding="utf-8-sig", float_format="%.6f")
with pd.ExcelWriter(TABLO / "tablolar_ihracat_did.xlsx") as xw:
    E1.to_excel(xw, sheet_name="E1_E4_DiD_DDD", index=False)
    E2.to_excel(xw, sheet_name="E2_olay_calismasi", index=False)
    E3.to_excel(xw, sheet_name="E3_plasebo", index=False)
print("\nKaydedildi.")
