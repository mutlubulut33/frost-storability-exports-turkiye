# -*- coding: utf-8 -*-
"""
03_don_kirilma.py
Nisan 2025 zirai donunun hal-ihracat fiyat ilişkisine etkisi. Tüm örneklem 2018-01 .. 2026-08.

  K1  Don öncesi/sonrası etki regresyonları:  y_t = a + ay kuklaları + δ·DON_t (+ trend) + ε
      (HAC/Newey-West standart hatalar) — ihracat miktarı, hal hacmi, fiyatlar, fiyat oranı,
      ihracatın hal hacmine oranı
  K2  Kestirim (predictive) Chow testi: don öncesi modelin 2025-04 .. 2026-08 dönemindeki uyumu
  K3  Don öncesi modelden bir adım ileri tahmin hataları (sabit parametre): öncesi vs sonrası
  K4  Tüm örneklem ECM'de don etkileşimleri (uyarlama hızı ve uzun dönem katsayısı değişti mi?)
  Ş4  Olay grafiği: fiyat oranı ve ihracat/hal hacmi oranı ;  Ş5  Tüm örneklem CUSUM

Model tanımları 02 betiğiyle aynıdır:
  ANA : ARDL(1,1)  lnH ~ lnX, lnER  (kur uzun dönemde)
  ALT : NARDL(1,2) lnH ~ X⁺, X⁻ ; ΔlnER yalnızca kısa dönemde
Çalıştırma:  python kod/03_don_kirilma.py
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import recursive_olsresiduals
from ortak import *
warnings.filterwarnings("ignore")

d = nardl_degiskenleri(veri_oku())            # kısmi toplamlar 2018-01'den itibaren
BAS = d.index[MAKS_GECIKME]                   # 2018-05
DON = pd.Timestamp(DON_BASLANGIC)
ON_SON = pd.Timestamp(ON_DON_SON)              # 2024-12 (ana örneklemin sonu)
rapor = []
def yaz(s=""):
    print(s); rapor.append(str(s))
yaz(f"Tüm örneklem {d.index[0]:%Y-%m} .. {d.index[-1]:%Y-%m}; don sonrası {int(d['don'].sum())} ay")

# ---------------------------------------------------------------------------
# K1 Etki regresyonları
# ---------------------------------------------------------------------------
d["ln_ihr_miktar"] = np.log(d["ihr_miktar_ton"])
d["ln_hal_hacim"] = np.log(d["hal_hacim_tum_ton"])
d["ln_oran"] = np.log(d["oran_hal_ihr"])
d["ihr_hal_orani"] = d["ihr_miktar_ton"] / d["hal_hacim_tum_ton"]
d["trend"] = np.arange(len(d))
hedef = {"ln_ihr_miktar": "ln İhracat miktarı", "ln_hal_hacim": "ln Hal hacmi",
         "lnH": "ln Hal fiyatı (USD)", "lnX": "ln İhracat birim değeri (USD)",
         "ln_oran": "ln (Hal / ihracat fiyatı)", "ihr_hal_orani": "İhracat / hal hacmi"}
k1 = []
for v, ad in hedef.items():
    for trend in (False, True):
        X = pd.DataFrame({"const": 1.0}, index=d.index)
        for m in range(2, 13):
            X[f"ay{m:02d}"] = (d.index.month == m).astype(float)
        if trend:
            X["trend"] = d["trend"]
        X["DON"] = d["don"].astype(float)
        r = sm.OLS(d[v], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        delta = r.params["DON"]
        k1.append({"Bağımlı": ad, "Trend": "Var" if trend else "Yok", "δ (DON)": delta,
                   "Std.hata (HAC)": r.bse["DON"], "p": r.pvalues["DON"],
                   "Yüzde etki (ln ise, %)": (np.exp(delta) - 1) * 100 if v.startswith("ln") else np.nan,
                   "Ön dönem ort.": d.loc[:DON - pd.offsets.MonthBegin(1), v].mean(),
                   "Don sonrası ort.": d.loc[DON:, v].mean(), "N": int(r.nobs)})
K1 = pd.DataFrame(k1)
yaz("\nK1 Don etkisi:"); yaz(K1.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Model tanımları
# ---------------------------------------------------------------------------
ek_er = pd.DataFrame({"D.lnER": d["lnER"].diff()})
MODELLER = {
    "ANA-ARDL(1,1)": dict(xs=["lnX", "lnER"], p=1, q=1, ek=None),
    "ALT-NARDL(1,2)": dict(xs=["Xp", "Xn"], p=1, q=2, ek=ek_er),
}
def tasarim(ad, veri=d, bas=BAS):
    m = MODELLER[ad]
    ek = None if m["ek"] is None else m["ek"]
    return ecm_tasarim(veri, "lnH", m["xs"], m["p"], m["q"], ek=ek, bas=bas)

# ---------------------------------------------------------------------------
# K2 Kestirim Chow testi ve K3 tahmin hataları
#   Ön örneklem: 2018-05 .. 2025-03 (don öncesi tüm veri) ; test: 2025-04 .. 2026-08
#   Ayrıca ön örneklem = 2018-05 .. 2024-12 (02'deki ana örneklem) ile de raporlanır.
# ---------------------------------------------------------------------------
k2, k3 = [], []
for ad in MODELLER:
    yy, X = tasarim(ad)
    for on_son in [DON - pd.offsets.MonthBegin(1), ON_SON]:
        on = yy.index <= on_son
        test = yy.index >= DON
        r_on = sm.OLS(yy[on], X[on]).fit()
        sec = on | test
        r_tum = sm.OLS(yy[sec], X[sec]).fit()
        n1, n2, k = int(on.sum()), int(test.sum()), X.shape[1]
        F = ((r_tum.ssr - r_on.ssr) / n2) / (r_on.ssr / (n1 - k))
        p = 1 - stats.f.cdf(F, n2, n1 - k)
        # Aradaki aylar (varsa) testten hariç: on_son 2024-12 iken 2025-01..03 kullanılmaz
        k2.append({"Model": ad, "Ön örneklem sonu": f"{on_son:%Y-%m}", "n1": n1,
                   "Test dönemi": f"{DON:%Y-%m}..{yy.index[-1]:%Y-%m}", "n2": n2, "k": k,
                   "Chow F": F, "p": p})
    # K3: 2025-03'e kadar tahmin edilen sabit parametreli model ile bir adım ileri tahminler
    on = yy.index < DON
    r_on = sm.OLS(yy[on], X[on]).fit()
    hata_on = r_on.resid
    hata_son = yy[~on] - r_on.predict(X[~on])
    for etk, h in [("Don öncesi (örneklem içi)", hata_on), ("Don sonrası (örneklem dışı)", hata_son)]:
        k3.append({"Model": ad, "Dönem": etk, "N": len(h), "Ortalama hata": h.mean(),
                   "RMSE": np.sqrt((h**2).mean()), "MAE": h.abs().mean(),
                   "|hata| > 2σ olan ay sayısı": int((h.abs() > 2 * np.sqrt(r_on.scale)).sum())})
K2 = pd.DataFrame(k2); K3 = pd.DataFrame(k3)
yaz("\nK2 Kestirim Chow testi:"); yaz(K2.round(4).to_string(index=False))
yaz("\nK3 Tahmin hataları (ΔlnH):"); yaz(K3.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# K4 Tüm örneklem ECM, don etkileşimleri
#   ΔlnH = ... + DON + DON×lnH_{t-1} + DON×(uzun dönem fiyat değişken(ler)i)_{t-1}
# ---------------------------------------------------------------------------
k4 = []
for ad in MODELLER:
    m = MODELLER[ad]
    fiyat = [x for x in m["xs"] if x != "lnER"]
    ek = pd.DataFrame({"DON": d["don"].astype(float),
                       "DONxlnH.L1": d["don"] * d["lnH"].shift(1)}, index=d.index)
    for x in fiyat:
        ek[f"DONx{x}.L1"] = d["don"] * d[x].shift(1)
    if m["ek"] is not None:
        ek = ek.join(m["ek"])
    yy, X = ecm_tasarim(d, "lnH", m["xs"], m["p"], m["q"], ek=ek, bas=BAS)
    r = sm.OLS(yy, X).fit()
    r_hac = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    etk = [c for c in X.columns if c.startswith("DON")]
    w = wald(r, np.vstack([R_vektor(r, {c: 1}) for c in etk]))
    for c in ["lnH.L1"] + [f"{x}.L1" for x in fiyat] + etk:
        k4.append({"Model": ad, "Katsayı": c, "Değer": r.params[c], "Std.hata": r.bse[c],
                   "p": r.pvalues[c], "p (HAC)": r_hac.pvalues[c]})
    k4.append({"Model": ad, "Katsayı": "Wald: tüm DON terimleri = 0", "Değer": w["F"],
               "Std.hata": np.nan, "p": w["p_F"], "p (HAC)": np.nan})
    # Don öncesi ve sonrası uyarlama hızı
    k4.append({"Model": ad, "Katsayı": "ρ (don öncesi)", "Değer": r.params["lnH.L1"],
               "Std.hata": np.nan, "p": np.nan, "p (HAC)": np.nan})
    rs = r.params["lnH.L1"] + r.params["DONxlnH.L1"]
    var = (r.cov_params().loc["lnH.L1", "lnH.L1"] + r.cov_params().loc["DONxlnH.L1", "DONxlnH.L1"]
           + 2 * r.cov_params().loc["lnH.L1", "DONxlnH.L1"])
    k4.append({"Model": ad, "Katsayı": "ρ (don sonrası) = ρ + DON×lnH.L1", "Değer": rs,
               "Std.hata": np.sqrt(var), "p": 2 * (1 - stats.t.cdf(abs(rs / np.sqrt(var)), r.df_resid)),
               "p (HAC)": np.nan})
    (TABLO / f"ozet_don_etkilesim_{ad}.txt").write_text(r.summary().as_text(), encoding="utf-8")
K4 = pd.DataFrame(k4)
yaz("\nK4 Don etkileşimleri:"); yaz(K4.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Ş4 Olay grafiği
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(d.index, d["hal_fiyat_3cesit_USD"], label="Hal fiyatı", color="#2b6cb0", lw=1.6)
ax[0].plot(d.index, d["ihr_bd_USD"], label="İhracat birim değeri", color="#c05621", lw=1.6)
ax[0].set_ylabel("USD/kg"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(d.index, d["oran_hal_ihr"], color="#4a5568", lw=1.4)
ax[1].axhline(1, color="grey", ls="--", lw=.8); ax[1].set_ylabel("Hal / ihracat fiyatı")
ax[2].bar(d.index, d["ihr_hal_orani"] * 100, width=25, color="#c05621", alpha=.8)
ax[2].set_ylabel("İhracat / hal hacmi (%)")
for a in ax:
    a.axvline(DON, color="red", lw=.9, ls=":"); a.spines[["top", "right"]].set_visible(False)
ax[0].set_title("Nisan 2025 donu öncesi ve sonrası elma fiyatları ve ihracat payı", fontsize=10)
plt.tight_layout(); plt.savefig(SEKIL / "sekil4_don_olay_grafigi.png", dpi=200); plt.close()

# ---------------------------------------------------------------------------
# Ş5 Tüm örneklem CUSUM (ALT model)  -  yapısal kırılma görsel kontrolü
# ---------------------------------------------------------------------------
cusum_ozet = []
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for a, ad in zip(axs, MODELLER):
    yy, X = tasarim(ad)
    r = sm.OLS(yy, X).fit()
    rr = recursive_olsresiduals(r)[3]
    rr = rr[~np.isnan(rr)]
    n = len(rr)
    cus = np.cumsum(rr) / np.std(rr, ddof=1)
    sinir = 0.948 * np.sqrt(n) + 2 * 0.948 * np.arange(1, n + 1) / np.sqrt(n)
    t = yy.index[-n:]
    a.plot(t, cus, color="#2b6cb0"); a.plot(t, sinir, "r--", lw=.8); a.plot(t, -sinir, "r--", lw=.8)
    a.axvline(DON, color="red", lw=.8, ls=":"); a.set_title(f"CUSUM tüm örneklem: {ad}", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
    asim = t[np.abs(cus) > sinir]
    cusum_ozet.append({"Model": ad, "Sınır aşımı": len(asim) > 0,
                       "İlk aşım": asim[0].strftime("%Y-%m") if len(asim) else "-"})
plt.tight_layout(); plt.savefig(SEKIL / "sekil5_cusum_tum_orneklem.png", dpi=200); plt.close()
K5 = pd.DataFrame(cusum_ozet)
yaz("\nK5 CUSUM tüm örneklem:"); yaz(K5.to_string(index=False))

with pd.ExcelWriter(TABLO / "tablolar_don_kirilma.xlsx") as xw:
    K1.to_excel(xw, sheet_name="K1_don_etkisi", index=False)
    K2.to_excel(xw, sheet_name="K2_kestirim_chow", index=False)
    K3.to_excel(xw, sheet_name="K3_tahmin_hatalari", index=False)
    K4.to_excel(xw, sheet_name="K4_don_etkilesim", index=False)
    K5.to_excel(xw, sheet_name="K5_cusum", index=False)
(TABLO / "rapor_03_don_kirilma.txt").write_text("\n".join(rapor), encoding="utf-8")
print("\nKaydedildi.")
