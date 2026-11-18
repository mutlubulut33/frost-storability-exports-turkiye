# -*- coding: utf-8 -*-
"""
ortak.py  -  Analiz betiklerinin ortak kullandığı yardımcı fonksiyonlar.

Model: ARDL / NARDL hata düzeltme (UECM) biçimi, Pesaran-Shin-Smith (2001) "case 3"
(kısıtsız sabit, trend yok) + 11 aylık mevsim kuklası.

  ΔlnH_t = c + Σ_m s_m D_m + ρ·lnH_{t-1} + Σ_k β_k·X_{k,t-1}
           + Σ_{i=1}^{p-1} a_i·ΔlnH_{t-i} + Σ_k Σ_{j=j0}^{q-1} b_{k,j}·ΔX_{k,t-j} + ε_t

NARDL'de ihracat birim değeri (lnX) Shin, Yu & Greenwood-Nimmo (2014) gibi
pozitif / negatif kısmi toplamlara ayrılır:  X⁺_t = Σ max(ΔlnX,0), X⁻_t = Σ min(ΔlnX,0).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "cikti"
TABLO = CIKTI / "tablolar"
SEKIL = CIKTI / "sekiller"
TABLO.mkdir(parents=True, exist_ok=True)
SEKIL.mkdir(parents=True, exist_ok=True)

MAKS_GECIKME = 4          # gecikme seçimi için üst sınır; ortak örneklem için ilk 4 ay ayrılır
ON_DON_SON = "2024-12"    # ana (don öncesi) örneklemin son ayı
DON_BASLANGIC = "2025-04" # Nisan 2025 zirai donu


def veri_oku():
    d = pd.read_csv(CIKTI / "veri_analiz.csv", index_col=0, encoding="utf-8-sig")
    d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp()
    d.index.freq = "MS"
    return d


def kismi_toplam(seri):
    """Pozitif ve negatif kısmi toplamlar (ilk gözlemde 0)."""
    dx = seri.diff().fillna(0.0)
    return dx.clip(lower=0).cumsum(), dx.clip(upper=0).cumsum()


def nardl_degiskenleri(d):
    d = d.copy()
    d["Xp"], d["Xn"] = kismi_toplam(d["lnX"])
    return d


def ecm_tasarim(d, y, xs, p, q, mevsim=True, nedensel=False, ek=None, bas=None):
    """
    UECM tasarım matrisi.
      y        : bağımlı değişken adı (düzey)
      xs       : uzun dönem açıklayıcılar (düzey) listesi
      p        : y için ARDL gecikme derecesi (ΔlnH gecikmeleri 1..p-1)
      q        : x'ler için ARDL derecesi (ΔX gecikmeleri j0..q-1)
      nedensel : True ise ΔX_t (eşanlı) çıkarılır, ΔX gecikmeleri 1..q-1 (tahmin için)
      ek       : tasarıma eklenecek hazır sütunlar (DataFrame)
      bas      : tahmin örnekleminin ilk tarihi (ortak örneklem için)
    Döndürür: (bağımlı ΔlnH, X DataFrame)
    """
    Z = pd.DataFrame(index=d.index)
    Z["const"] = 1.0
    if mevsim:
        for m in range(2, 13):
            Z[f"ay{m:02d}"] = (d.index.month == m).astype(float)
    Z[f"{y}.L1"] = d[y].shift(1)
    for x in xs:
        Z[f"{x}.L1"] = d[x].shift(1)
    dy = d[y].diff()
    for i in range(1, p):
        Z[f"D.{y}.L{i}"] = dy.shift(i)
    j0 = 1 if nedensel else 0
    for x in xs:
        dxx = d[x].diff()
        for j in range(j0, q):
            Z[f"D.{x}.L{j}"] = dxx.shift(j)
    if ek is not None:
        Z = Z.join(ek)
    yy = dy.rename(f"D.{y}")
    tam = pd.concat([yy, Z], axis=1).dropna()
    if bas is not None:
        tam = tam.loc[bas:]
    return tam.iloc[:, 0], tam.iloc[:, 1:]


def ols(yy, X, cov="nonrobust"):
    return sm.OLS(yy, X).fit(cov_type=cov) if cov != "nonrobust" else sm.OLS(yy, X).fit()


def wald(res, R, r=None):
    """H0: R b = r. F ve ki-kare istatistiği (OLS kovaryansı)."""
    b = res.params.values
    V = res.cov_params().values
    R = np.atleast_2d(R)
    r = np.zeros(R.shape[0]) if r is None else np.asarray(r)
    fark = R @ b - r
    W = float(fark @ np.linalg.solve(R @ V @ R.T, fark))
    J = R.shape[0]
    F = W / J
    return {"F": F, "p_F": 1 - stats.f.cdf(F, J, res.df_resid),
            "Wald": W, "p_Wald": 1 - stats.chi2.cdf(W, J), "df": (J, int(res.df_resid))}


def R_vektor(res, katsayilar):
    """Verilen {ad: ağırlık} sözlüğünden kısıt satırı."""
    r = np.zeros(len(res.params))
    for ad, w in katsayilar.items():
        r[list(res.params.index).index(ad)] = w
    return r


def uzun_donem(res, y, x):
    """θ_x = -β_x / ρ ; delta yöntemiyle standart hata."""
    ad_r, ad_b = f"{y}.L1", f"{x}.L1"
    rho, beta = res.params[ad_r], res.params[ad_b]
    V = res.cov_params().loc[[ad_r, ad_b], [ad_r, ad_b]].values
    theta = -beta / rho
    g = np.array([beta / rho**2, -1 / rho])       # ∂θ/∂ρ , ∂θ/∂β
    se = float(np.sqrt(g @ V @ g))
    t = theta / se
    return theta, se, t, 2 * (1 - stats.t.cdf(abs(t), res.df_resid))


def bic_secim(d, y, xs, mevsim=True, nedensel=False, bas=None):
    """p, q ∈ {1..MAKS_GECIKME}; ortak örneklemde BIC'e göre seçim."""
    sonuc = []
    for p in range(1, MAKS_GECIKME + 1):
        for q in range(1, MAKS_GECIKME + 1):
            yy, X = ecm_tasarim(d, y, xs, p, q, mevsim, nedensel, bas=bas)
            r = ols(yy, X)
            sonuc.append({"p": p, "q": q, "n": int(r.nobs), "k": X.shape[1],
                          "AIC": r.aic, "BIC": r.bic})
    t = pd.DataFrame(sonuc).sort_values("BIC").reset_index(drop=True)
    return t


def yildiz(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def pss_simulasyon(k, n, nsim=20000, tohum=20261011):
    """
    PSS (2001) case 3 sınır testi için sonlu örneklem kritik değerleri ve p-değerleri
    (Narayan 2005 yaklaşımı). H0 (eşbütünleşme yok) altında:
        Δy_t = e_t ; x'ler I(1) (rasgele yürüyüş, üst sınır) veya I(0) (beyaz gürültü, alt sınır)
    Regresyon: Δy_t = c + ρ y_{t-1} + Σ β x_{t-1} + ε ;  F = Wald(ρ = β = 0)/(k+1),
    t = ρ'nun t istatistiği.
    Döndürür: dict(F_I0, F_I1, t_I0, t_I1) -> simüle edilmiş istatistik dizileri
    """
    rng = np.random.default_rng(tohum)
    sonuc = {"F_I0": np.empty(nsim), "F_I1": np.empty(nsim),
             "t_I0": np.empty(nsim), "t_I1": np.empty(nsim)}
    J = k + 1
    for i in range(nsim):
        e = rng.standard_normal((n + 1, k + 1))
        y = np.cumsum(e[:, 0])
        for tip, x in (("I0", e[:, 1:]), ("I1", np.cumsum(e[:, 1:], axis=0))):
            X = np.column_stack([np.ones(n), y[:-1], x[:-1]])
            dy = np.diff(y)
            XtX_inv = np.linalg.inv(X.T @ X)
            b = XtX_inv @ X.T @ dy
            u = dy - X @ b
            s2 = u @ u / (n - X.shape[1])
            V = s2 * XtX_inv
            bb = b[1:]; VV = V[1:, 1:]
            sonuc[f"F_{tip}"][i] = bb @ np.linalg.solve(VV, bb) / J
            sonuc[f"t_{tip}"][i] = b[1] / np.sqrt(V[1, 1])
    return sonuc


def sinir_kritik(sim, F, t):
    """Simüle dağılımlardan kritik değerler ve gözlenen F, t için p-değerleri."""
    kd = {}
    for a in (0.10, 0.05, 0.01):
        kd[f"F KD %{int(a*100)} I(0)"] = np.quantile(sim["F_I0"], 1 - a)
        kd[f"F KD %{int(a*100)} I(1)"] = np.quantile(sim["F_I1"], 1 - a)
        kd[f"t KD %{int(a*100)} I(0)"] = np.quantile(sim["t_I0"], a)
        kd[f"t KD %{int(a*100)} I(1)"] = np.quantile(sim["t_I1"], a)
    kd["F p (I(0))"] = float(np.mean(sim["F_I0"] >= F))
    kd["F p (I(1))"] = float(np.mean(sim["F_I1"] >= F))
    kd["t p (I(0))"] = float(np.mean(sim["t_I0"] <= t))
    kd["t p (I(1))"] = float(np.mean(sim["t_I1"] <= t))
    return kd
