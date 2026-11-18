# -*- coding: utf-8 -*-
"""
08_limonsuz_narenciye.py
SONRADAN EKLENEN ANALİZ (20.10.2026). Gerekçe: Ticaret Bakanlığı'nın "İhracı Kayda Bağlı Mallara İlişkin
Tebliğ (İhracat 2006/7)'de Değişiklik Yapılmasına Dair Tebliğ (İhracat: 2025/6)" ile elma (0808.10) ve limon
(0805.50) ihracatı aynı anda kayda bağlanmıştır. Limon, 05 ve 06 betiklerindeki narenciye kontrol grubunun
içindedir; bu nedenle kontrol grubu da aynı politikadan etkilenmiştir. Bu betik, limonu çıkararak
(portakal 080510 + mandalina grubu 080521/22/29 + greyfurt 080540) 05 ve 06'daki tahminleri tekrarlar.
Model tanımları 05 ve 06 ile aynıdır; yalnızca kontrol grubu değişir. Sonuçlar ana tahminlerin yanında
raporlanmalıdır.

Çalıştırma: python kod/08_limonsuz_narenciye.py   (01-07 önce çalıştırılmış olmalı)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ortak import KOK, TABLO

C = KOK / "cikti"
def oku(ad):
    d = pd.read_csv(C / ad, index_col=0, encoding="utf-8-sig")
    d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp()
    return d

e = oku("veri_ihracat_did.csv")
hal = oku("veri_did_panel.csv")
kir = oku("veri_kiraz.csv")
NAR3 = ["Portakal", "Mandalina", "Greyfurt"]
nar3_ton = e[[f"{g}_ihr_ton" for g in NAR3]].fillna(0).sum(axis=1)
nar3_hal = hal[["hacim_ton_PORTAKAL", "hacim_ton_MANDALİNA", "hacim_ton_GREYFURT"]].sum(axis=1)
assert (nar3_ton > 0).all() and (nar3_hal > 0).all()

def ay_kukla(idx, aylar=range(2, 13)):
    X = pd.DataFrame({"const": 1.0}, index=idx)
    for m in aylar:
        X[f"ay{m:02d}"] = (idx.month == m).astype(float)
    return X

def tahmin(f, X, c, lag):
    r = sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return r.params[c], r.bse[c], r.pvalues[c], int(r.nobs)

sat = []
# (1) Elma ihracatı ve üçlü fark (05 ile aynı tasarım)
seriler = {
    "Export quantity": np.log(e["elma_ihr_ton"]) - np.log(nar3_ton),
    "Triple difference: exports vs. wholesale":
        (np.log(e["elma_ihr_ton"]) - np.log(hal["hacim_ton_ELMA"].reindex(e.index)))
        - (np.log(nar3_ton) - np.log(nar3_hal.reindex(e.index))),
}
for ad, f in seriler.items():
    f = f.dropna()
    for pad, pb in [("From Apr 2025", "2025-04-01"), ("From Aug 2025", "2025-08-01")]:
        X = ay_kukla(f.index); X["POST"] = (f.index >= pb).astype(float)
        b, se, p, n = tahmin(f, X, "POST", 6)
        sat.append({"Crop": "Apple", "Outcome": ad, "Comparison group": "Citrus excl. lemon",
                    "Period": pad, "N": n, "β": b, "HAC s.e.": se, "p": p, "Effect (%)": (np.exp(b) - 1) * 100})

# (2) Elma hal hacmi (04 ile aynı tasarım)
f = (np.log(hal["hacim_ton_ELMA"]) - np.log(hal[["hacim_ton_PORTAKAL", "hacim_ton_MANDALİNA", "hacim_ton_GREYFURT"]]).mean(axis=1)).loc["2018-01":]
for pad, pb in [("From Apr 2025", "2025-04-01"), ("From Aug 2025", "2025-08-01")]:
    X = ay_kukla(f.index); X["POST"] = (f.index >= pb).astype(float)
    b, se, p, n = tahmin(f, X, "POST", 6)
    sat.append({"Crop": "Apple", "Outcome": "Wholesale volume", "Comparison group": "Citrus excl. lemon",
                "Period": pad, "N": n, "β": b, "HAC s.e.": se, "p": p, "Effect (%)": (np.exp(b) - 1) * 100})

# (3) Kiraz (06 ile aynı tasarım: Mayıs-Temmuz)
s = kir[kir.index.month.isin([5, 6, 7])].copy()
s["nar3"] = nar3_ton.reindex(s.index)
s["nar3_hal"] = nar3_hal.reindex(s.index)
s["kir_hal"] = hal["hacim_ton_KİRAZ"].reindex(s.index)
kseriler = {"Export quantity": np.log(s["kiraz_ton"]) - np.log(s["nar3"]),
            "Export unit value (USD)": np.log(s["kiraz_bd"]) - np.log(
                (e.loc[s.index, [f"{g}_ihr_ton" for g in NAR3]].fillna(0).values *
                 e.loc[s.index, [f"{g}_bd" for g in NAR3]].fillna(0).values).sum(axis=1) / s["nar3"]),
            "Triple difference: exports vs. wholesale":
                (np.log(s["kiraz_ton"]) - np.log(s["kir_hal"])) - (np.log(s["nar3"]) - np.log(s["nar3_hal"]))}
for ad, f in kseriler.items():
    X = ay_kukla(f.index, aylar=[6, 7])
    X["S2025"] = (f.index.year == 2025).astype(float); X["S2026"] = (f.index.year == 2026).astype(float)
    r = sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    f0 = f[f.index.year <= 2024]
    plas = []
    for y in range(2019, 2025):
        X0 = ay_kukla(f0.index, aylar=[6, 7]); X0["S"] = (f0.index.year == y).astype(float)
        plas.append(sm.OLS(f0, X0).fit().params["S"])
    plas = np.array(plas)
    for c in ["S2025", "S2026"]:
        b = r.params[c]
        sat.append({"Crop": "Sweet cherry", "Outcome": ad, "Comparison group": "Citrus excl. lemon",
                    "Period": c[1:] + " season", "N": int(r.nobs), "β": b, "HAC s.e.": r.bse[c], "p": r.pvalues[c],
                    "Effect (%)": (np.exp(b) - 1) * 100,
                    "Placebos with |β| ≥ |actual| (of 6)": int((np.abs(plas) >= abs(b)).sum())})

S = pd.DataFrame(sat)
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))

# Marketing-year export totals relative to 2018/19-2023/24 (descriptive)
x = pd.DataFrame({"Apple": e["elma_ihr_ton"], "Citrus excl. lemon": nar3_ton,
                  "Lemon": e["Limon_ihr_ton"].fillna(0)})
py = np.where(x.index.month >= 8, x.index.year, x.index.year - 1)
a = x.groupby(py).sum().loc[2018:2025]
oran = a / a.loc[2018:2023].mean()
oran.index = [f"{i}/{str(i + 1)[-2:]}" for i in oran.index]
print("\n", oran.round(2).to_string())

with pd.ExcelWriter(TABLO / "tablolar_limonsuz_narenciye.xlsx") as xw:
    S.to_excel(xw, sheet_name="L1_tahminler", index=False)
    oran.to_excel(xw, sheet_name="L2_pazarlama_yili_oran")
print("\nKaydedildi.")
