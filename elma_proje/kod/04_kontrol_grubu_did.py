# -*- coding: utf-8 -*-
"""
04_kontrol_grubu_did.py
Farkların farkı (DiD): Nisan 2025 donunun ELMA hal hacmi ve fiyatına etkisi,
dondan etkilenmeyen ürünler kontrol grubu olarak.  Kaynak: yalnızca HKS (ham_veri/HKS_2017_bugun_TAM.csv)

Önceden belirlenen tasarım (sonuçlar görülmeden önce yazılmıştır):
  Tedavi  : ELMA (tüm elma ürün adları: ELMA, ELMA STARKING, ELMA GOLDEN, ELMA GRANNY SMITH, ELMA AMASYA)
  Kontrol A (ana) : narenciye  - PORTAKAL, MANDALİNA, LİMON, GREYFURT (tüm alt ürün adlarıyla)
  Kontrol B       : depolanabilir tarla ürünleri - PATATES, SOĞAN KURU
  Kontrol C       : A + B
  Doğrulama (dondan etkilendiği bilinen ürünler, etkinin GÖRÜLMESİ beklenir): KİRAZ, ŞEFTALİ, ERİK
  Dönem   : 2018-01 .. 2026-09 (2017 HKS kapsamı düşük olduğu için dışarıda)
  Sonuç değişkenleri: ln aylık hal işlem hacmi ; ln hacim ağırlıklı ortalama fiyat (TL/kg)

Tahmin:  D_t = ln y_ELMA,t − ortalama_c(ln y_c,t)
         D_t = a + Σ ay kuklaları + β·POST_t + ε_t       (Newey-West HAC, 6 gecikme)
  β : don sonrası elmanın kontrol grubuna göre ek değişimi (log puan); %etki = exp(β) − 1
  POST1 = 2025-04 ve sonrası (don ayından itibaren)
  POST2 = 2025-08 ve sonrası (yeni hasat sezonu, Ağustos-Temmuz pazarlama yılı)
  Takvim yılı/ay sabit etkileri farkta ortak şokları (enflasyon, kur, talep) giderir; ay kuklaları
  ürünler arası farklı mevsimselliği giderir.
Ek: pazarlama yılı (Ağustos-Temmuz) olay çalışması; zamanda ve uzayda plasebo testleri.
Çalıştırma:  python kod/04_kontrol_grubu_did.py
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

URUN = {  # grup adı -> HKS "Ürün Adı" için düzenli ifade
    "ELMA": r"^ELMA($| (STARKING|GOLDEN|GRANNY SMITH|AMASYA)$)",
    "PORTAKAL": r"^PORTAKAL", "MANDALİNA": r"^MANDALİNA", "LİMON": r"^LİMON",
    "GREYFURT": r"^GREYFURT", "PATATES": r"^PATATES", "SOĞAN KURU": r"^SOĞAN KURU$",
    "KİRAZ": r"^KİRAZ", "ŞEFTALİ": r"^ŞEFTALİ", "ERİK": r"^ERİK",
}
KONTROL = {"A_narenciye": ["PORTAKAL", "MANDALİNA", "LİMON", "GREYFURT"],
           "B_patates_sogan": ["PATATES", "SOĞAN KURU"]}
KONTROL["C_hepsi"] = KONTROL["A_narenciye"] + KONTROL["B_patates_sogan"]
POST = {"POST1 (≥2025-04)": "2025-04-01", "POST2 (≥2025-08)": "2025-08-01"}
BAS, SON = "2018-01-01", "2026-09-01"

# ---------------------------------------------------------------------------
# Panel: ürün x ay
# ---------------------------------------------------------------------------
hks = pd.read_csv(KOK / "ham_veri" / "HKS_2017_bugun_TAM.csv", encoding="utf-8-sig")
hks["Tarih"] = pd.to_datetime(hks["Tarih"], format="%d.%m.%Y")
hks = hks[(hks["İşlem Hacmi"] > 0) & (hks["Birim Adı"] == "Kg")
          & (hks["Ürün Türü"] != "Organik Tarım")]
hks["deger"] = hks["Ortalama Fiyat"] * hks["İşlem Hacmi"]
seriler = {}
for g, desen in URUN.items():
    s = hks[hks["Ürün Adı"].str.match(desen)]
    a = s.groupby(pd.Grouper(key="Tarih", freq="MS"))[["deger", "İşlem Hacmi"]].sum()
    a = a.loc[BAS:SON]
    seriler[g] = pd.DataFrame({"hacim_ton": a["İşlem Hacmi"] / 1e3,
                               "fiyat_TL": a["deger"] / a["İşlem Hacmi"]})
panel = pd.concat(seriler, names=["urun", "Tarih"]).reset_index()
panel["ln_hacim"] = np.log(panel["hacim_ton"])
panel["ln_fiyat"] = np.log(panel["fiyat_TL"])
eksik = panel.pivot(index="Tarih", columns="urun", values="hacim_ton").isna().sum()
print("Eksik ay sayısı (ürün bazında):", eksik[eksik > 0].to_dict() or "yok")
genis = {v: panel.pivot(index="Tarih", columns="urun", values=v) for v in ["ln_hacim", "ln_fiyat"]}
genis["ln_hacim"].index.freq = "MS"

def ay_kukla(idx):
    X = pd.DataFrame({"const": 1.0}, index=idx)
    for m in range(2, 13):
        X[f"ay{m:02d}"] = (idx.month == m).astype(float)
    return X

def did(fark, post_bas):
    X = ay_kukla(fark.index)
    X["POST"] = (fark.index >= post_bas).astype(float)
    r = sm.OLS(fark, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return r

def fark_serisi(v, tedavi, kontrol):
    return (genis[v][tedavi] - genis[v][kontrol].mean(axis=1, skipna=False)).dropna()

# ---------------------------------------------------------------------------
# D1 Ana DiD sonuçları
# ---------------------------------------------------------------------------
sat = []
for v, vad in [("ln_hacim", "ln Hal hacmi"), ("ln_fiyat", "ln Hal fiyatı (TL)")]:
    for kad, kl in KONTROL.items():
        f = fark_serisi(v, "ELMA", kl)
        for pad, pb in POST.items():
            r = did(f, pb)
            b = r.params["POST"]
            sat.append({"Sonuç": vad, "Kontrol": kad, "Post tanımı": pad, "N": int(r.nobs),
                        "β": b, "Std.hata (HAC)": r.bse["POST"], "p": r.pvalues["POST"],
                        "%95 GA alt": r.conf_int().loc["POST", 0], "%95 GA üst": r.conf_int().loc["POST", 1],
                        "% etki": (np.exp(b) - 1) * 100})
D1 = pd.DataFrame(sat)
print("\nD1 DiD:"); print(D1.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# D2 Doğrulama: dondan etkilendiği bilinen ürünler (aynı tasarım, kontrol C)
#     Bu ürünlerin hasat dışı aylarda işlem hacmi olmayabilir -> yalnızca dolu aylar.
# ---------------------------------------------------------------------------
sat = []
for u in ["KİRAZ", "ŞEFTALİ", "ERİK"]:
    for v, vad in [("ln_hacim", "ln Hal hacmi"), ("ln_fiyat", "ln Hal fiyatı (TL)")]:
        f = fark_serisi(v, u, KONTROL["C_hepsi"])
        r = did(f, POST["POST1 (≥2025-04)"])
        sat.append({"Ürün": u, "Sonuç": vad, "N": int(r.nobs), "β": r.params["POST"],
                    "Std.hata (HAC)": r.bse["POST"], "p": r.pvalues["POST"],
                    "% etki": (np.exp(r.params["POST"]) - 1) * 100})
D2 = pd.DataFrame(sat)
print("\nD2 Doğrulama (dondan etkilenen ürünler):"); print(D2.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# D3 Olay çalışması: pazarlama yılı (Ağustos-Temmuz) kuklaları, referans 2023/24
#    D_t = a + ay kuklaları + Σ_{y≠2023/24} γ_y·PY_y + ε
# ---------------------------------------------------------------------------
def pazarlama_yili(idx):
    y = np.where(idx.month >= 8, idx.year, idx.year - 1)
    return pd.Series([f"{a}/{str(a+1)[-2:]}" for a in y], index=idx)
REF = "2023/24"
olay = []
for v, vad in [("ln_hacim", "ln Hal hacmi"), ("ln_fiyat", "ln Hal fiyatı (TL)")]:
    for kad in ["A_narenciye", "B_patates_sogan"]:
        f = fark_serisi(v, "ELMA", KONTROL[kad])
        py = pazarlama_yili(f.index)
        X = ay_kukla(f.index)
        for y in sorted(py.unique()):
            if y != REF:
                X[f"PY_{y}"] = (py == y).astype(float)
        r = sm.OLS(f, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        for c in [c for c in X.columns if c.startswith("PY_")]:
            olay.append({"Sonuç": vad, "Kontrol": kad, "Pazarlama yılı": c[3:],
                         "γ": r.params[c], "Std.hata": r.bse[c], "p": r.pvalues[c],
                         "GA alt": r.conf_int().loc[c, 0], "GA üst": r.conf_int().loc[c, 1]})
        olay.append({"Sonuç": vad, "Kontrol": kad, "Pazarlama yılı": REF + " (ref)",
                     "γ": 0.0, "Std.hata": 0.0, "p": np.nan, "GA alt": 0.0, "GA üst": 0.0})
D3 = pd.DataFrame(olay)
D3["sira"] = D3["Pazarlama yılı"].str[:4].astype(int)
D3 = D3.sort_values(["Sonuç", "Kontrol", "sira"]).drop(columns="sira")
print("\nD3 Olay çalışması:"); print(D3.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for a, (v, vad) in zip(axs, [("ln Hal hacmi", "Hal hacmi"), ("ln Hal fiyatı (TL)", "Hal fiyatı")]):
    for (kad, renk, kay) in [("A_narenciye", "#2b6cb0", -0.12), ("B_patates_sogan", "#c05621", 0.12)]:
        s = D3[(D3["Sonuç"] == v) & (D3["Kontrol"] == kad)]
        x = np.arange(len(s)) + kay
        a.errorbar(x, s["γ"], yerr=[s["γ"] - s["GA alt"], s["GA üst"] - s["γ"]], fmt="o",
                   color=renk, capsize=3, ms=4, label=f"Kontrol {kad[0]}")
    a.set_xticks(np.arange(len(s))); a.set_xticklabels([e if e != "2026/27" else "2026/27*" for e in s["Pazarlama yılı"].str[:7]], rotation=45, fontsize=8)
    a.axhline(0, color="black", lw=.6); a.axvline(len(s) - 2.5, color="red", ls=":", lw=.9)  # 2024/25 ile 2025/26 arası (ilk dondan etkilenen hasat)
    a.set_title(f"Elma − kontrol: {vad} (ref. 2023/24, %95 GA)", fontsize=9)
    a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=8)
fig.text(0.01, 0.01, "* 2026/27 yalnızca Ağustos-Eylül 2026. Pazarlama yılı: Ağustos-Temmuz.", fontsize=7)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(SEKIL / "sekil6_olay_calismasi_did.png", dpi=200); plt.close()

# ---------------------------------------------------------------------------
# D4 Plasebo testleri
#   (a) Zamanda: sahte don tarihleri = 2019..2024 Nisan; yalnızca gerçek don öncesi veri
#       (sahte tarih + 16 ay penceresi, 2025-03 ile sınırlı) ; kontrol A ve B
#   (b) Uzayda: her kontrol ürünü "tedavi", kalan kontrol ürünleri kontrol ; POST1
# ---------------------------------------------------------------------------
plas = []
for v, vad in [("ln_hacim", "ln Hal hacmi"), ("ln_fiyat", "ln Hal fiyatı (TL)")]:
    for kad in ["A_narenciye", "B_patates_sogan"]:
        f_tam = fark_serisi(v, "ELMA", KONTROL[kad])
        for yil in range(2019, 2025):
            sahte = pd.Timestamp(f"{yil}-04-01")
            son = min(sahte + pd.DateOffset(months=16), pd.Timestamp("2025-03-01"))
            f = f_tam.loc[:son]
            r = did(f, sahte)
            plas.append({"Tür": "Zamanda", "Sonuç": vad, "Kontrol": kad, "Sahte tedavi": f"{yil}-04",
                         "β": r.params["POST"], "p": r.pvalues["POST"]})
        r = did(f_tam, POST["POST1 (≥2025-04)"])
        plas.append({"Tür": "Gerçek", "Sonuç": vad, "Kontrol": kad, "Sahte tedavi": "2025-04 (gerçek)",
                     "β": r.params["POST"], "p": r.pvalues["POST"]})
    for u in KONTROL["C_hepsi"]:
        kl = [k for k in KONTROL["C_hepsi"] if k != u]
        r = did(fark_serisi(v, u, kl), POST["POST1 (≥2025-04)"])
        plas.append({"Tür": "Uzayda", "Sonuç": vad, "Kontrol": "C (kendisi hariç)", "Sahte tedavi": u,
                     "β": r.params["POST"], "p": r.pvalues["POST"]})
    r = did(fark_serisi(v, "ELMA", KONTROL["C_hepsi"]), POST["POST1 (≥2025-04)"])
    plas.append({"Tür": "Gerçek", "Sonuç": vad, "Kontrol": "C_hepsi", "Sahte tedavi": "ELMA (gerçek)",
                 "β": r.params["POST"], "p": r.pvalues["POST"]})
D4 = pd.DataFrame(plas)
print("\nD4 Plasebo:"); print(D4.round(3).to_string(index=False))
# Özet: gerçek |β|, plasebolardan kaçının üstünde?
ozet = []
for (tur, v), g in D4[D4["Tür"] != "Gerçek"].groupby(["Tür", "Sonuç"]):
    if tur == "Zamanda":
        for kad in ["A_narenciye", "B_patates_sogan"]:
            gg = g[g["Kontrol"] == kad]
            gercek = D4[(D4["Tür"] == "Gerçek") & (D4["Sonuç"] == v) & (D4["Kontrol"] == kad)]["β"].iloc[0]
            ozet.append({"Tür": tur, "Sonuç": v, "Kontrol": kad, "Gerçek β": gercek,
                         "Plasebo sayısı": len(gg), "|plasebo β| ≥ |gerçek β|": int((gg["β"].abs() >= abs(gercek)).sum()),
                         "Plasebo β aralığı": f"{gg['β'].min():.3f} .. {gg['β'].max():.3f}"})
    else:
        gercek = D4[(D4["Tür"] == "Gerçek") & (D4["Sonuç"] == v) & (D4["Kontrol"] == "C_hepsi")]["β"].iloc[0]
        ozet.append({"Tür": tur, "Sonuç": v, "Kontrol": "C", "Gerçek β": gercek,
                     "Plasebo sayısı": len(g), "|plasebo β| ≥ |gerçek β|": int((g["β"].abs() >= abs(gercek)).sum()),
                     "Plasebo β aralığı": f"{g['β'].min():.3f} .. {g['β'].max():.3f}"})
D4o = pd.DataFrame(ozet)
print("\nD4 Plasebo özeti:"); print(D4o.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Kayıt
# ---------------------------------------------------------------------------
genis_cikti = panel.pivot(index="Tarih", columns="urun", values=["hacim_ton", "fiyat_TL"])
genis_cikti.columns = [f"{a}_{b}" for a, b in genis_cikti.columns]
genis_cikti.index = genis_cikti.index.strftime("%Y-%m"); genis_cikti.index.name = "Ay"
genis_cikti.to_csv(KOK / "cikti" / "veri_did_panel.csv", encoding="utf-8-sig", float_format="%.6f")
with pd.ExcelWriter(TABLO / "tablolar_did.xlsx") as xw:
    D1.to_excel(xw, sheet_name="D1_DiD", index=False)
    D2.to_excel(xw, sheet_name="D2_dogrulama", index=False)
    D3.to_excel(xw, sheet_name="D3_olay_calismasi", index=False)
    D4.to_excel(xw, sheet_name="D4_plasebo", index=False)
    D4o.to_excel(xw, sheet_name="D4_plasebo_ozet", index=False)
print("\nKaydedildi.")
