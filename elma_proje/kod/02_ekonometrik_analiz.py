# -*- coding: utf-8 -*-
"""
02_ekonometrik_analiz.py
Don öncesi ana örneklem (2018-01 .. 2024-12) üzerinde:
  T1  Betimsel istatistikler
  T2  Birim kök testleri (ADF, KPSS, Zivot-Andrews)
  T3  Gecikme seçimi (BIC)
  T4  ARDL / NARDL sınır testi (PSS 2001, case 3)
  T5  Uzun ve kısa dönem katsayılar (ARDL ve NARDL)
  T6  Asimetri Wald testleri
  T7  Depolama dönemi etkisi (H3)
  T8  Tanı testleri
  T9  Sağlamlık: Huber robust tahmin, HAC standart hatalar, çeşit düzeyi
  Ş2  NARDL dinamik çarpanlar ;  Ş3  CUSUM / CUSUMSQ

Çalıştırma:  python kod/02_ekonometrik_analiz.py
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, het_arch,
                                          linear_reset, recursive_olsresiduals,
                                          het_breuschpagan)
from statsmodels.stats.stattools import jarque_bera
from ortak import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InterpolationWarning.*")
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)

tam = nardl_degiskenleri(veri_oku())
d = tam.loc[:ON_DON_SON].copy()
# Kısmi toplamlar ana örneklemin başından (2018-01) itibaren hesaplanır
d["Xp"], d["Xn"] = kismi_toplam(d["lnX"])
BAS = d.index[MAKS_GECIKME]             # ortak tahmin örnekleminin ilk ayı (2018-05)
rapor = []                              # konsol/özet metni
def yaz(s=""):
    print(s); rapor.append(str(s))

yaz(f"Ana örneklem: {d.index[0]:%Y-%m} .. {d.index[-1]:%Y-%m} ({len(d)} ay); "
    f"tahmin örneklemi {BAS:%Y-%m} başlangıçlı")

# ---------------------------------------------------------------------------
# T1 Betimsel istatistikler
# ---------------------------------------------------------------------------
deg = {"hal_fiyat_3cesit_USD": "Hal fiyatı (USD/kg)",
       "ihr_bd_USD": "İhracat birim değeri (USD/kg)",
       "usdtry": "USD/TRY",
       "ihr_miktar_ton": "İhracat miktarı (ton)",
       "hal_hacim_tum_ton": "Hal işlem hacmi (ton)",
       "oran_hal_ihr": "Hal / ihracat fiyat oranı"}
bet = []
for donem, s in [("2018-01..2024-12", tam.loc[:ON_DON_SON]),
                 ("2025-04..2026-08", tam.loc[DON_BASLANGIC:])]:
    for c, ad in deg.items():
        x = s[c]
        bet.append({"Dönem": donem, "Değişken": ad, "N": len(x), "Ortalama": x.mean(),
                    "Std.sapma": x.std(), "Min": x.min(), "Maks": x.max(),
                    "Değişim katsayısı": x.std() / x.mean()})
T1 = pd.DataFrame(bet)

# ---------------------------------------------------------------------------
# T2 Birim kök testleri
# ---------------------------------------------------------------------------
br = []
for v, ad in [("lnH", "ln Hal fiyatı"), ("lnX", "ln İhracat birim değeri"),
              ("lnER", "ln USD/TRY")]:
    for tip, s in [("Düzey", d[v]), ("Birinci fark", d[v].diff().dropna())]:
        a = adfuller(s, regression="c", autolag="BIC")
        k = kpss(s, regression="c", nlags="auto")
        satir = {"Değişken": ad, "Seri": tip, "ADF istatistik": a[0], "ADF p": a[1],
                 "ADF gecikme": a[2], "KPSS istatistik": k[0], "KPSS p (≥/≤ tablo sınırı)": k[1]}
        if tip == "Düzey":
            z = zivot_andrews(s, regression="c", autolag="BIC")
            satir.update({"ZA istatistik": z[0], "ZA p": z[1],
                          "ZA kırılma tarihi": s.index[z[4]].strftime("%Y-%m")})
        br.append(satir)
T2 = pd.DataFrame(br)
yaz("\nT2 Birim kök:"); yaz(T2.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# T3 Gecikme seçimi
# ---------------------------------------------------------------------------
X_LIN = ["lnX", "lnER"]
X_NL = ["Xp", "Xn", "lnER"]
sec_lin = bic_secim(d, "lnH", X_LIN, bas=BAS)
sec_nl = bic_secim(d, "lnH", X_NL, bas=BAS)
P_L, Q_L = int(sec_lin.p[0]), int(sec_lin.q[0])
P_N, Q_N = int(sec_nl.p[0]), int(sec_nl.q[0])
T3 = pd.concat([sec_lin.assign(Model="ARDL"), sec_nl.assign(Model="NARDL")])
yaz(f"\nT3 Seçilen gecikmeler (BIC): ARDL p={P_L}, q={Q_L} ; NARDL p={P_N}, q={Q_N}")

res_lin = ols(*ecm_tasarim(d, "lnH", X_LIN, P_L, Q_L, bas=BAS))
res_nl = ols(*ecm_tasarim(d, "lnH", X_NL, P_N, Q_N, bas=BAS))

# ---------------------------------------------------------------------------
# T4 Sınır testi  (kendi F hesabımız + statsmodels UECM ile çapraz kontrol)
# ---------------------------------------------------------------------------
def sinir_testi(res, xs, ad):
    """PSS (2001) case 3 F ve t_BDM testi. Kritik değerler ve p-değerleri, bu örneklem
    büyüklüğü (n) ve regresör sayısı (k) için simülasyonla üretilir (ortak.pss_simulasyon).
    Not: statsmodels UECM.bounds_test kullanılmadı; çünkü (i) mevsim kuklalarını içermeyen
    ayrı bir model tahmin ediyor, (ii) k'yı y dahil sayıyor."""
    duzey = ["lnH.L1"] + [f"{x}.L1" for x in xs]
    R = np.vstack([R_vektor(res, {c: 1}) for c in duzey])
    w = wald(res, R)
    t_rho = float(res.tvalues["lnH.L1"])
    sim = pss_simulasyon(len(xs), int(res.nobs))
    satir = {"Model": ad, "k": len(xs), "n": int(res.nobs), "F istatistiği": w["F"],
             "t_BDM (ρ)": t_rho}
    satir.update(sinir_kritik(sim, w["F"], t_rho))
    return satir

# Simülatörün doğrulanması: n=1000, k=2 için PSS (2001) Tablo CI(iii) ve CII(iii) değerleri
_dog = pss_simulasyon(2, 1000, nsim=20000)
DOGRULAMA = pd.DataFrame({
    "Anlamlılık": ["%10", "%5", "%1"],
    "Simüle F I(0)": [np.quantile(_dog["F_I0"], q) for q in (0.90, 0.95, 0.99)],
    "PSS F I(0)": [3.17, 3.79, 5.15],
    "Simüle F I(1)": [np.quantile(_dog["F_I1"], q) for q in (0.90, 0.95, 0.99)],
    "PSS F I(1)": [4.14, 4.85, 6.36],
    "Simüle t I(0)": [np.quantile(_dog["t_I0"], q) for q in (0.10, 0.05, 0.01)],
    "PSS t I(0)": [-2.57, -2.86, -3.43],
    "Simüle t I(1)": [np.quantile(_dog["t_I1"], q) for q in (0.10, 0.05, 0.01)],
    "PSS t I(1)": [-3.21, -3.53, -4.10]})

s1 = sinir_testi(res_lin, X_LIN, "ARDL")
s2 = sinir_testi(res_nl, X_NL, "NARDL")
T4 = pd.DataFrame([s1, s2])
yaz("\nT4 Sınır testi:"); yaz(T4.round(3).T.to_string())

# ---------------------------------------------------------------------------
# T5 Katsayılar
# ---------------------------------------------------------------------------
def katsayi_tablosu(res, xs, ad):
    sat = []
    for x in xs:
        th, se, t, p = uzun_donem(res, "lnH", x)
        sat.append({"Model": ad, "Bölüm": "Uzun dönem", "Değişken": x,
                    "Katsayı": th, "Std.hata": se, "t": t, "p": p})
    for c in res.params.index:
        if c == "const" or c.startswith("ay"):
            continue
        bolum = "ECT (ρ)" if c == "lnH.L1" else ("Düzey (L1)" if c.endswith(".L1") and not c.startswith("D.") else "Kısa dönem")
        sat.append({"Model": ad, "Bölüm": bolum, "Değişken": c, "Katsayı": res.params[c],
                    "Std.hata": res.bse[c], "t": res.tvalues[c], "p": res.pvalues[c]})
    sat.append({"Model": ad, "Bölüm": "Uyum", "Değişken": "R2 / düz.R2 / N",
                "Katsayı": res.rsquared, "Std.hata": res.rsquared_adj, "t": res.nobs, "p": np.nan})
    return pd.DataFrame(sat)

T5 = pd.concat([katsayi_tablosu(res_lin, X_LIN, "ARDL"),
                katsayi_tablosu(res_nl, X_NL, "NARDL")])
T5["Anlamlılık"] = T5["p"].apply(lambda p: yildiz(p) if pd.notna(p) else "")
yaz("\nT5 Katsayılar:"); yaz(T5.round(4).to_string(index=False))
# Yarı ömür (ay): dengesizliğin yarısının kapanma süresi
for ad, r in [("ARDL", res_lin), ("NARDL", res_nl)]:
    rho = r.params["lnH.L1"]
    yaz(f"{ad}: ρ = {rho:.3f}, yarı ömür ≈ {np.log(0.5)/np.log(1+rho):.1f} ay")

# ---------------------------------------------------------------------------
# T6 Asimetri testleri
# ---------------------------------------------------------------------------
asi = []
# Uzun dönem: θ+ = θ-  <=>  β+ = β-   (aynı ρ)
w = wald(res_nl, R_vektor(res_nl, {"Xp.L1": 1, "Xn.L1": -1}))
thp = uzun_donem(res_nl, "lnH", "Xp")[0]; thn = uzun_donem(res_nl, "lnH", "Xn")[0]
asi.append({"Test": "Uzun dönem asimetri (θ⁺ = θ⁻)", "θ⁺ / Σδ⁺": thp, "θ⁻ / Σδ⁻": thn,
            "F": w["F"], "p": w["p_F"]})
# Kısa dönem: Σ ΔX⁺ katsayıları = Σ ΔX⁻ katsayıları
kp = [c for c in res_nl.params.index if c.startswith("D.Xp")]
kn = [c for c in res_nl.params.index if c.startswith("D.Xn")]
w = wald(res_nl, R_vektor(res_nl, {**{c: 1 for c in kp}, **{c: -1 for c in kn}}))
asi.append({"Test": "Kısa dönem asimetri (Σδ⁺ = Σδ⁻)",
            "θ⁺ / Σδ⁺": res_nl.params[kp].sum(), "θ⁻ / Σδ⁻": res_nl.params[kn].sum(),
            "F": w["F"], "p": w["p_F"]})
T6 = pd.DataFrame(asi)
yaz("\nT6 Asimetri:"); yaz(T6.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# T7 Depolama dönemi etkisi (H3) - iki aşamalı ECM
#    ECT_t = lnH_t - θ⁺·X⁺_t - θ⁻·X⁻_t - θ_ER·lnER_t   (NARDL uzun dönem katsayıları)
#    ΔlnH_t = c + ay kuklaları + λ·ECT_{t-1} + λ_S·(ECT_{t-1}×S_t)
#             + δ⁺ΔX⁺_t + δ⁻ΔX⁻_t + φ⁺(ΔX⁺_t×S_t) + φ⁻(ΔX⁻_t×S_t) + a·ΔlnH_{t-1} + ε
#    S_t = 1 (Kasım-Mayıs depolama dönemi)
# ---------------------------------------------------------------------------
th = {x: uzun_donem(res_nl, "lnH", x)[0] for x in X_NL}
ect = d["lnH"] - sum(th[x] * d[x] for x in X_NL)
S = d["depolama"]
ek = pd.DataFrame({"ECT.L1": ect.shift(1), "ECTxS": ect.shift(1) * S,
                   "D.Xp": d["Xp"].diff(), "D.Xn": d["Xn"].diff(),
                   "D.XpxS": d["Xp"].diff() * S, "D.XnxS": d["Xn"].diff() * S,
                   "D.lnH.L1": d["lnH"].diff().shift(1)}, index=d.index)
Z = pd.DataFrame({"const": 1.0}, index=d.index)
for m in range(2, 13):
    Z[f"ay{m:02d}"] = (d.index.month == m).astype(float)
tasarim = pd.concat([d["lnH"].diff().rename("D.lnH"), Z, ek], axis=1).dropna().loc[BAS:]
res_s = ols(tasarim.iloc[:, 0], tasarim.iloc[:, 1:])
res_s_hac = sm.OLS(tasarim.iloc[:, 0], tasarim.iloc[:, 1:]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
T7 = pd.DataFrame({"Katsayı": res_s.params, "Std.hata": res_s.bse, "p": res_s.pvalues,
                   "Std.hata (HAC)": res_s_hac.bse, "p (HAC)": res_s_hac.pvalues})
T7 = T7[~T7.index.str.startswith("ay")]
# Birleşik test: depolama etkileşimleri sıfır mı?
w_s = wald(res_s, np.vstack([R_vektor(res_s, {c: 1}) for c in ["ECTxS", "D.XpxS", "D.XnxS"]]))
T7.loc["Wald: λ_S = φ⁺ = φ⁻ = 0 (F, p)"] = [w_s["F"], np.nan, w_s["p_F"], np.nan, np.nan]
T7.loc["R2 / N"] = [res_s.rsquared, np.nan, res_s.nobs, np.nan, np.nan]
yaz("\nT7 Depolama etkisi:"); yaz(T7.round(4).to_string())

# ---------------------------------------------------------------------------
# T8 Tanı testleri
# ---------------------------------------------------------------------------
def tani(res, ad):
    bg = acorr_breusch_godfrey(res, nlags=12)
    ar = het_arch(res.resid, nlags=12)
    jb = jarque_bera(res.resid)
    bp = het_breuschpagan(res.resid, res.model.exog)
    rs = linear_reset(res, power=2, use_f=True)
    return [{"Model": ad, "Test": "Breusch-Godfrey LM (12)", "İstatistik": bg[2], "p": bg[3]},
            {"Model": ad, "Test": "ARCH LM (12)", "İstatistik": ar[2], "p": ar[3]},
            {"Model": ad, "Test": "Breusch-Pagan", "İstatistik": bp[2], "p": bp[3]},
            {"Model": ad, "Test": "Jarque-Bera", "İstatistik": jb[0], "p": jb[1]},
            {"Model": ad, "Test": "Ramsey RESET (kare)", "İstatistik": float(rs.fvalue), "p": float(rs.pvalue)}]
T8 = pd.DataFrame(tani(res_lin, "ARDL") + tani(res_nl, "NARDL"))
yaz("\nT8 Tanı testleri:"); yaz(T8.round(4).to_string(index=False))

# Aykırı gözlemler (standartlaştırılmış kalıntı > 2.5)
std_k = res_nl.resid / np.sqrt(res_nl.scale)
aykiri = std_k[abs(std_k) > 2.5]
yaz("\nNARDL |std. kalıntı| > 2.5 olan aylar: " +
    ", ".join(f"{i:%Y-%m} ({v:.2f})" for i, v in aykiri.items()))

# ---------------------------------------------------------------------------
# T9 Sağlamlık
# ---------------------------------------------------------------------------
yy, X = ecm_tasarim(d, "lnH", X_NL, P_N, Q_N, bas=BAS)
rlm = sm.RLM(yy, X, M=sm.robust.norms.HuberT()).fit()
hac = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
def lr_rlm(r, x):
    return -r.params[f"{x}.L1"] / r.params["lnH.L1"]
sag = []
for x in X_NL:
    sag.append({"Değişken (uzun dönem)": x, "NARDL-OLS": uzun_donem(res_nl, "lnH", x)[0],
                "NARDL-Huber": lr_rlm(rlm, x)})
sag.append({"Değişken (uzun dönem)": "ρ (ECT)", "NARDL-OLS": res_nl.params["lnH.L1"],
            "NARDL-Huber": rlm.params["lnH.L1"]})
T9a = pd.DataFrame(sag)
T9b = pd.DataFrame({"OLS std.hata": res_nl.bse, "HAC std.hata": hac.bse,
                    "OLS p": res_nl.pvalues, "HAC p": hac.pvalues})
T9b = T9b[~T9b.index.str.startswith("ay")]
# Çeşit düzeyi: Starking ve Golden hal fiyatı ~ aynı çeşidin ihracat birim değeri (ARDL)
ces = []
for c in ["Starking", "Golden"]:
    dc = d.copy()
    dc["lnH"] = np.log(dc[f"hal_fiyat_{c}_USD"]); dc["lnX"] = np.log(dc[f"ihr_bd_{c}_USD"])
    sc = bic_secim(dc, "lnH", X_LIN, bas=BAS)
    pc, qc = int(sc.p[0]), int(sc.q[0])
    rc = ols(*ecm_tasarim(dc, "lnH", X_LIN, pc, qc, bas=BAS))
    duzey = ["lnH.L1", "lnX.L1", "lnER.L1"]
    wc = wald(rc, np.vstack([R_vektor(rc, {k: 1}) for k in duzey]))
    uc = sinir_kritik(pss_simulasyon(2, int(rc.nobs)), wc["F"], rc.tvalues["lnH.L1"])
    th_x = uzun_donem(rc, "lnH", "lnX")
    ces.append({"Çeşit": c, "p,q": f"{pc},{qc}", "Sınır F": wc["F"],
                "p (I(1))": uc["F p (I(1))"], "θ_lnX": th_x[0], "θ_lnX p": th_x[3],
                "ρ": rc.params["lnH.L1"], "ρ p": rc.pvalues["lnH.L1"]})
T9c = pd.DataFrame(ces)
yaz("\nT9a Huber:"); yaz(T9a.round(4).to_string(index=False))
yaz("\nT9c Çeşit düzeyi:"); yaz(T9c.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Ş2 Dinamik çarpanlar (NARDL): X⁺ ve X⁻'de kalıcı 1 birimlik değişime lnH'nin kümülatif tepkisi
# ---------------------------------------------------------------------------
def dinamik_carpan(res, sok, ufuk=24, p=P_N, q=Q_N):
    b = res.params
    H = np.zeros(ufuk + p + 1); Xs = {x: np.zeros(ufuk + p + 1) for x in X_NL}
    t0 = p
    Xs[sok][t0:] = 1.0                     # t0'da kalıcı birim artış
    for t in range(t0, len(H)):
        dH = b["lnH.L1"] * H[t - 1]
        for x in X_NL:
            dH += b[f"{x}.L1"] * Xs[x][t - 1]
            for j in range(q):
                c = f"D.{x}.L{j}"
                if c in b: dH += b[c] * (Xs[x][t - j] - Xs[x][t - j - 1])
        for i in range(1, p):
            dH += b[f"D.lnH.L{i}"] * (H[t - i] - H[t - i - 1])
        H[t] = H[t - 1] + dH
    return H[t0:]
mp = dinamik_carpan(res_nl, "Xp"); mn = dinamik_carpan(res_nl, "Xn")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mp, label="Pozitif değişim (X⁺)", color="#2b6cb0", lw=1.8)
ax.plot(-mn, label="Negatif değişim (−X⁻, işareti çevrilmiş)", color="#c05621", lw=1.8)
ax.plot(mp + mn, label="Asimetri (fark)", color="#4a5568", ls="--", lw=1.2)
ax.axhline(uzun_donem(res_nl, "lnH", "Xp")[0], color="#2b6cb0", lw=.6, ls=":")
ax.axhline(-uzun_donem(res_nl, "lnH", "Xn")[0], color="#c05621", lw=.6, ls=":")
ax.axhline(0, color="black", lw=.5)
ax.set_xlabel("Ay"); ax.set_ylabel("lnH kümülatif tepkisi")
ax.set_title("NARDL dinamik çarpanlar: ihracat fiyatındaki %1 değişime hal fiyatının tepkisi", fontsize=9)
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(SEKIL / "sekil2_dinamik_carpanlar.png", dpi=200); plt.close()
pd.DataFrame({"ay": range(len(mp)), "X+ tepki": mp, "X- tepki": mn,
              "asimetri (X+ + X-)": mp + mn}).to_csv(TABLO / "S2_dinamik_carpanlar.csv",
                                                     index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Ş3 CUSUM / CUSUMSQ (NARDL)
# ---------------------------------------------------------------------------
rr = recursive_olsresiduals(res_nl, skip=None, alpha=0.95)
rres = rr[3][~np.isnan(rr[3])]           # standartlaştırılmış rekürsif kalıntılar
k = int(res_nl.df_model) + 1; n = len(rres)
cusum = np.cumsum(rres) / np.std(rres, ddof=1)
a = 0.948                                 # %5 anlamlılık
sinir = a * np.sqrt(n) + 2 * a * np.arange(1, n + 1) / np.sqrt(n)
cusq = np.cumsum(rres**2) / np.sum(rres**2)
c0 = 0.21  # yaklaşık %5 kritik değer (Durbin 1969, n/2 - 1 ≈ 25-30 için)
beklenen = np.arange(1, n + 1) / n
tarih = yy.index[-n:]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(tarih, cusum, color="#2b6cb0"); ax[0].plot(tarih, sinir, "r--", lw=.8); ax[0].plot(tarih, -sinir, "r--", lw=.8)
ax[0].set_title("CUSUM (%5 sınırlar)", fontsize=9)
ax[1].plot(tarih, cusq, color="#2b6cb0"); ax[1].plot(tarih, beklenen + c0, "r--", lw=.8); ax[1].plot(tarih, beklenen - c0, "r--", lw=.8)
ax[1].set_title("CUSUMSQ (yaklaşık %5 sınırlar)", fontsize=9)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(SEKIL / "sekil3_cusum_nardl.png", dpi=200); plt.close()
yaz(f"\nCUSUM sınır aşımı: {bool((abs(cusum) > sinir).any())} ; "
    f"CUSUMSQ sınır aşımı: {bool((abs(cusq - beklenen) > c0).any())}")


# ---------------------------------------------------------------------------
# T10 Alternatif tanım: kur yalnızca kısa dönem kontrolü (ΔlnER_t), uzun dönemde yok.
#     Gerekçe: USD cinsinden fiyatlar 2018-2024'te I(0)/I(1) sınırında, lnER ise açıkça I(1);
#     kurun asıl etkisi TL hal fiyatlarının yapışkanlığından doğan kısa dönem etkidir.
#     NOT: Bu tanım ana tanımdan SONRA denenmiştir; makalede iki tanım birlikte raporlanmalıdır.
# ---------------------------------------------------------------------------
ek_er = pd.DataFrame({"D.lnER": d["lnER"].diff()})
alt_sat, alt_kat = [], []
for xs, ad in [(["lnX"], "ARDL-alt"), (["Xp", "Xn"], "NARDL-alt")]:
    tab = []
    for p_ in range(1, MAKS_GECIKME + 1):
        for q_ in range(1, MAKS_GECIKME + 1):
            r_ = ols(*ecm_tasarim(d, "lnH", xs, p_, q_, ek=ek_er, bas=BAS))
            tab.append((r_.bic, p_, q_, r_))
    _, pa, qa, ra = min(tab, key=lambda z: z[0])
    duzey = ["lnH.L1"] + [f"{x}.L1" for x in xs]
    wa = wald(ra, np.vstack([R_vektor(ra, {c: 1}) for c in duzey]))
    kd = sinir_kritik(pss_simulasyon(len(xs), int(ra.nobs)), wa["F"], ra.tvalues["lnH.L1"])
    sat = {"Model": ad, "p,q": f"{pa},{qa}", "n": int(ra.nobs), "F": wa["F"],
           "t_BDM": ra.tvalues["lnH.L1"], **kd, "ρ": ra.params["lnH.L1"],
           "ΔlnER katsayısı": ra.params["D.lnER"], "ΔlnER p": ra.pvalues["D.lnER"]}
    for x in xs:
        th_ = uzun_donem(ra, "lnH", x)
        sat[f"θ_{x}"] = th_[0]; sat[f"θ_{x} p"] = th_[3]
    if len(xs) == 2:
        wl = wald(ra, R_vektor(ra, {"Xp.L1": 1, "Xn.L1": -1}))
        kp_ = [c for c in ra.params.index if c.startswith("D.Xp")]
        kn_ = [c for c in ra.params.index if c.startswith("D.Xn")]
        wk = wald(ra, R_vektor(ra, {**{c: 1 for c in kp_}, **{c: -1 for c in kn_}}))
        sat.update({"LR asimetri F": wl["F"], "LR asimetri p": wl["p_F"],
                    "SR asimetri F": wk["F"], "SR asimetri p": wk["p_F"]})
    alt_sat.append(sat)
    (TABLO / f"ozet_{ad}.txt").write_text(ra.summary().as_text(), encoding="utf-8")
T10 = pd.DataFrame(alt_sat)
yaz("\nT10 Alternatif tanım (kur yalnızca kısa dönem):"); yaz(T10.round(4).T.to_string())

# ---------------------------------------------------------------------------
# Kayıt
# ---------------------------------------------------------------------------
with pd.ExcelWriter(TABLO / "tablolar_ana_analiz.xlsx") as xw:
    T1.to_excel(xw, sheet_name="T1_betimsel", index=False)
    T2.to_excel(xw, sheet_name="T2_birim_kok", index=False)
    T3.to_excel(xw, sheet_name="T3_gecikme_secimi", index=False)
    T4.to_excel(xw, sheet_name="T4_sinir_testi", index=False)
    T5.to_excel(xw, sheet_name="T5_katsayilar", index=False)
    T6.to_excel(xw, sheet_name="T6_asimetri", index=False)
    T7.to_excel(xw, sheet_name="T7_depolama")
    T8.to_excel(xw, sheet_name="T8_tani", index=False)
    T9a.to_excel(xw, sheet_name="T9a_huber", index=False)
    T9b.to_excel(xw, sheet_name="T9b_HAC")
    T9c.to_excel(xw, sheet_name="T9c_cesit", index=False)
    T10.to_excel(xw, sheet_name="T10_alternatif_tanim", index=False)
    DOGRULAMA.to_excel(xw, sheet_name="Ek_PSS_dogrulama", index=False)
for ad, r in [("ARDL", res_lin), ("NARDL", res_nl), ("Depolama_ECM", res_s)]:
    (TABLO / f"ozet_{ad}.txt").write_text(r.summary().as_text(), encoding="utf-8")
(TABLO / "rapor_02_ana_analiz.txt").write_text("\n".join(rapor), encoding="utf-8")
print("\nTablolar ve şekiller kaydedildi.")
