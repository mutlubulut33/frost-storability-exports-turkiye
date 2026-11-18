# -*- coding: utf-8 -*-
"""
09_makale_tablolari.py
Makaledeki (İngilizce, MDPI) tabloları ve şekilleri, 01-07 betiklerinin ürettiği dosyalardan üretir.
Hiçbir sayı elle girilmez; her tablo hücresi bir sonuç dosyasından okunur.

Girdiler : cikti/veri_*.csv, cikti/tablolar/tablolar_*.xlsx
Çıktılar : cikti/makale/Table1..Table5 (.xlsx ve .md), cikti/makale/Figure1..3 (.png, 300 dpi)
Çalıştırma: python kod/09_makale_tablolari.py   (01-08 önce çalıştırılmış olmalı)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortak import KOK

C = KOK / "cikti"; T = C / "tablolar"; M = C / "makale"; M.mkdir(exist_ok=True)
for eski in M.glob("*"):
    eski.unlink()   # eski adlandırmalı dosyalar kalmasın
MAVI, TURUNCU, GRI = "#2b6cb0", "#c05621", "#4a5568"

def oku_csv(ad):
    d = pd.read_csv(C / ad, index_col=0, encoding="utf-8-sig")
    d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp()
    return d

def kaydet(df, ad, not_=""):
    df.to_excel(M / f"{ad}.xlsx", index=False)
    md = df.to_markdown(index=False, disable_numparse=True)
    (M / f"{ad}.md").write_text(md + ("\n\n" + not_ if not_ else ""), encoding="utf-8")
    print(f"\n{ad}\n{md}")

def f(x, k=3):
    return "" if pd.isna(x) else f"{x:.{k}f}"

def p_bicim(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

# ---------------------------------------------------------------------------
# Table 1. Marketing-year overview (Panel A apple, Panel B cherry season)
# ---------------------------------------------------------------------------
pan = oku_csv("veri_did_panel.csv")
ihr = oku_csv("veri_ihracat_did.csv")
ana = oku_csv("veri_analiz.csv")
kir = oku_csv("veri_kiraz.csv")
NAR = ["PORTAKAL", "MANDALİNA", "LİMON", "GREYFURT"]

x = pd.DataFrame(index=ihr.index)
x["hal"] = pan["hacim_ton_ELMA"]
x["hal_usd_deger"] = pan["hacim_ton_ELMA"] * pan["fiyat_TL_ELMA"] / ana["usdtry"]
x["ihr"] = ihr["elma_ihr_ton"]
x["ihr_usd"] = ihr["elma_ihr_ton"] * ihr["elma_bd"]
x["nar_hal"] = pan[[f"hacim_ton_{u}" for u in NAR]].sum(axis=1)
x["nar_ihr"] = ihr["nar_ihr_ton"]
x = x.loc["2018-08":"2026-07"]
py = np.where(x.index.month >= 8, x.index.year, x.index.year - 1)
a = x.groupby(py).sum()
t1a = pd.DataFrame({
    "Marketing year": [f"{i}/{str(i + 1)[-2:]}" for i in a.index],
    "Apple wholesale volume (kt)": (a["hal"] / 1e3).map(lambda v: f(v, 0)),
    "Apple exports (kt)": (a["ihr"] / 1e3).map(lambda v: f(v, 1)),
    "Exports / wholesale (%)": (a["ihr"] / a["hal"] * 100).map(lambda v: f(v, 1)),
    "Apple wholesale price (USD/kg)": (a["hal_usd_deger"] / a["hal"]).map(lambda v: f(v, 2)),
    "Apple export unit value (USD/kg)": (a["ihr_usd"] / a["ihr"]).map(lambda v: f(v, 2)),
    "Citrus wholesale volume (kt)": (a["nar_hal"] / 1e3).map(lambda v: f(v, 0)),
    "Citrus exports (kt)": (a["nar_ihr"] / 1e3).map(lambda v: f(v, 0)),
})
kaydet(t1a, "Table1_PanelA_apple_marketing_years",
       "Note: Marketing year = August-July. Wholesale price = volume-weighted HKS price converted at the monthly "
       "average USD/TRY buying rate. Sources: HKS, TurkStat, CBRT; authors' calculations (09_makale_tablolari.py).")

s = kir[kir.index.month.isin([5, 6, 7])].copy()
s["hal"] = pan.loc[s.index, "hacim_ton_KİRAZ"]
s["usd"] = s["kiraz_ton"] * s["kiraz_bd"]
b = s.groupby(s.index.year)[["hal", "kiraz_ton", "usd", "nar_ihr_ton", "kavun_karpuz_ton"]].sum()
t1b = pd.DataFrame({
    "Season (May-July)": b.index.astype(str),
    "Cherry wholesale volume (kt)": (b["hal"] / 1e3).map(lambda v: f(v, 1)),
    "Sweet cherry exports (kt)": (b["kiraz_ton"] / 1e3).map(lambda v: f(v, 1)),
    "Cherry export unit value (USD/kg)": (b["usd"] / b["kiraz_ton"]).map(lambda v: f(v, 2)),
    "Citrus exports (kt)": (b["nar_ihr_ton"] / 1e3).map(lambda v: f(v, 0)),
    "Melon and watermelon exports (kt)": (b["kavun_karpuz_ton"] / 1e3).map(lambda v: f(v, 1)),
})
kaydet(t1b, "Table1_PanelB_cherry_seasons", "Note: Sweet cherry = HS 080929. Sources: HKS, TurkStat.")

# ---------------------------------------------------------------------------
# Table 2. Wholesale DiD (apple) + validation products
# ---------------------------------------------------------------------------
SON = {"ln Hal hacmi": "Wholesale volume", "ln Hal fiyatı (TL)": "Wholesale price (TRY)",
       "ln İhracat miktarı": "Export quantity", "ln İhracat birim değeri": "Export unit value (USD)",
       "DDD: ln(İhr/Hal) elma − narenciye": "Triple difference: exports vs. wholesale",
       "Üçlü fark ln(İhr/Hal)": "Triple difference: exports vs. wholesale"}
KON = {"A_narenciye": "Citrus", "B_patates_sogan": "Potato and onion", "C_hepsi": "Citrus, potato and onion",
       "Narenciye toplamı": "Citrus (total)", "4 grubun log ortalaması": "Citrus (mean of 4 groups)",
       "Narenciye": "Citrus", "Kavun+karpuz": "Melon and watermelon"}
POSTA = {"POST1 (≥2025-04)": "From Apr 2025", "POST2 (≥2025-08)": "From Aug 2025"}

def tablo_did(df, post_sut):
    return pd.DataFrame({
        "Outcome": df["Sonuç"].map(SON), "Comparison group": df["Kontrol"].map(KON),
        "Post-frost period": df[post_sut].map(POSTA), "N": df["N"].astype(int).astype(str),
        "β": df["β"].map(f), "HAC s.e.": df["Std.hata (HAC)"].map(f), "p": df["p"].map(p_bicim),
        "Effect (%)": df["% etki"].map(lambda v: f(v, 1))})

d1 = pd.read_excel(T / "tablolar_did.xlsx", sheet_name="D1_DiD")
kaydet(tablo_did(d1, "Post tanımı"), "TableS1_wholesale_DiD_all_groups",
       "Note: Equation (1). β in log points; effect = exp(β) − 1. Newey-West HAC standard errors, 6 lags.")
d2 = pd.read_excel(T / "tablolar_did.xlsx", sheet_name="D2_dogrulama")
URUN = {"KİRAZ": "Cherry", "ŞEFTALİ": "Peach", "ERİK": "Plum"}
kaydet(pd.DataFrame({"Product": d2["Ürün"].map(URUN), "Outcome": d2["Sonuç"].map(SON),
                     "N": d2["N"].astype(str), "β": d2["β"].map(f), "HAC s.e.": d2["Std.hata (HAC)"].map(f),
                     "p": d2["p"].map(p_bicim), "Effect (%)": d2["% etki"].map(lambda v: f(v, 1))}),
       "TableS2_validation_products",
       "Note: Comparison group: citrus, potato and onion. Post-frost period from April 2025.")

# ---------------------------------------------------------------------------
# Table 3. Apple exports DiD and triple difference
# ---------------------------------------------------------------------------
e1 = pd.read_excel(T / "tablolar_ihracat_did.xlsx", sheet_name="E1_E4_DiD_DDD")
kaydet(tablo_did(e1, "Post"), "TableS3_exports_DiD_DDD_citrus",
       "Note: Equations (1) and (2). Export data end in August 2026. Newey-West HAC standard errors, 6 lags.")


# ---------------------------------------------------------------------------
# Table 3 Panel B / Table 4 Panel B. Citrus excluding lemon (post hoc, 08_limonsuz_narenciye.py)
# ---------------------------------------------------------------------------
L = pd.read_excel(T / "tablolar_limonsuz_narenciye.xlsx", sheet_name="L1_tahminler")
def tablo_L(df):
    return pd.DataFrame({"Outcome": df["Outcome"], "Comparison group": df["Comparison group"],
                         "Period": df["Period"], "N": df["N"].astype(str), "β": df["β"].map(f),
                         "HAC s.e.": df["HAC s.e."].map(f), "p": df["p"].map(p_bicim),
                         "Effect (%)": df["Effect (%)"].map(lambda v: f(v, 1))})
kaydet(tablo_L(L[L["Crop"] == "Apple"]), "TableS4_apple_citrus_excl_lemon",
       "Note: Post hoc. Lemon (HS 080550) excluded from the comparison group because apple and lemon exports "
       "were made subject to registration by the same communiqué (Resmî Gazete No. 33014, 11 September 2025).")
Lk = L[L["Crop"] == "Sweet cherry"].copy()
Tk = tablo_L(Lk); Tk["Placebos with |β| ≥ |actual| (of 6)"] = Lk["Placebos with |β| ≥ |actual| (of 6)"].astype(int).astype(str).values
kaydet(Tk, "TableS6_cherry_citrus_excl_lemon",
       "Note: Post hoc, not part of the pre-analysis plan. May-July exports of citrus other than lemon are small "
       "(often a few hundred tonnes per month) and also fell in 2025.")

# ---------------------------------------------------------------------------
# Table 4. Cherry (pre-registered)
# ---------------------------------------------------------------------------
k = pd.read_excel(T / "tablolar_kiraz_on_kayitli.xlsx")
TED = {"Kiraz (080929)": "Sweet cherry", "Kiraz+vişne": "Sweet + sour cherry"}
kaydet(pd.DataFrame({
    "Outcome": k["Sonuç"].map(SON), "Treated": k["Tedavi"].map(TED), "Comparison group": k["Kontrol"].map(KON),
    "Season": k["Sezon"].astype(str), "β": k["β"].map(f), "HAC s.e.": k["Std.hata (HAC)"].map(f),
    "p": k["p"].map(p_bicim), "Effect (%)": k["% etki"].map(lambda v: f(v, 1)),
    "Placebo β range 2019-2024": k["Plasebo β aralığı (2019-24)"],
    "Placebos with |β| ≥ actual (of 6)": k["|plasebo| ≥ |β| sayısı (6 içinde)"].astype(str)}),
    "TableS5_cherry_full",
    "Note: Equation (3); May-July 2018-2026, N = 27. Newey-West HAC standard errors, 2 lags. "
    "Placebos: each 2018-2024 season except 2018 treated in turn, estimated on 2018-2024 data.")

# ---------------------------------------------------------------------------
# Table 5. Robustness: placebo summary + potato/onion descriptive comparison
# ---------------------------------------------------------------------------
pl = pd.read_excel(T / "tablolar_did.xlsx", sheet_name="D4_plasebo_ozet")
e3 = pd.read_excel(T / "tablolar_ihracat_did.xlsx", sheet_name="E3_plasebo")
TUR = {"Zamanda": "Time (Apr 2019-Apr 2024)", "Uzayda": "Space (each comparison product)"}
rows = []
for _, r in pl.iterrows():
    rows.append({"Outcome": SON[r["Sonuç"]], "Placebo type": TUR[r["Tür"]],
                 "Comparison group": KON.get(r["Kontrol"], "Citrus, potato and onion" if r["Kontrol"] == "C" else r["Kontrol"]),
                 "Actual β (from Apr 2025)": f(r["Gerçek β"]), "Placebo β range": r["Plasebo β aralığı"],
                 "Placebos with |β| ≥ |actual|": f"{int(r['|plasebo β| ≥ |gerçek β|'])} of {int(r['Plasebo sayısı'])}"})
for s_, g in e3.groupby("Sonuç", sort=False):
    gercek = g["Gerçek β (2025-04)"].iloc[0]
    rows.append({"Outcome": SON[s_], "Placebo type": TUR["Zamanda"], "Comparison group": "Citrus (total)",
                 "Actual β (from Apr 2025)": f(gercek), "Placebo β range": f"{g['β'].min():.3f} .. {g['β'].max():.3f}",
                 "Placebos with |β| ≥ |actual|": f"{int((g['β'].abs() >= abs(gercek)).sum())} of {len(g)}"})
kaydet(pd.DataFrame(rows), "TableS7_placebo_summary",
       "Note: Rows 1-6 apple wholesale; last two rows apple exports.")
u3 = pd.read_excel(T / "tablolar_patates_sogan.xlsx", sheet_name="U3_don_oncesi_orana_gore").rename(columns={"Unnamed: 0": "Marketing year"})
u3 = u3[["Marketing year", "Elma", "Narenciye", "Patates", "Soğan", "Patates+Soğan"]]
u3.columns = ["Marketing year", "Apple", "Citrus", "Potato", "Onion", "Potato + onion"]
for c_ in u3.columns[1:]:
    u3[c_] = u3[c_].map(lambda v: f(v, 2))
kaydet(u3, "TableS8_exports_relative_to_preperiod",
       "Note: Export quantity in each marketing year divided by the 2018/19-2023/24 average.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
GENISLIK = 6.69   # inç = 17 cm, MDPI tam sayfa genişliği; şekiller bu boyutta çizilir (küçültme yok)

def sekil_kaydet(fig, ad):
    """PNG (600 dpi), TIFF (600 dpi, LZW) ve vektörel PDF. Açıklamalar şekilde değil, makaledeki başlıkta."""
    fig.savefig(M / f"{ad}.png", dpi=600, facecolor="white")
    from PIL import Image
    with Image.open(M / f"{ad}.png") as im:          # TIFF: RGB (saydamlık kanalı yok), LZW, 600 dpi
        im.convert("RGB").save(M / f"{ad}.tiff", compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(M / f"{ad}.pdf")
    plt.close(fig)
DON = pd.Timestamp("2025-04-01")

# Figure 1
fig, ax = plt.subplots(1, 2, figsize=(GENISLIK, 2.6), gridspec_kw={"width_ratios": [2.2, 1]})
z = pd.DataFrame({"hal": pan["hacim_ton_ELMA"], "ihr": ihr["elma_ihr_ton"]}).loc["2018-01":"2026-08"] / 1e3
ax[0].plot(z.index, z["hal"], color=MAVI, lw=1.4, label="Wholesale volume (HKS)")
ax[0].plot(z.index, z["ihr"], color=TURUNCU, lw=1.4, label="Exports (TurkStat)")
ax[0].set_yscale("log"); ax[0].set_ylabel("Thousand tonnes per month (log scale)")
ax[0].axvline(DON, color="red", ls=":", lw=1); ax[0].text(DON, ax[0].get_ylim()[1] * 0.7, " April 2025\n frost", color="red", fontsize=7, va="top")
ax[0].legend(frameon=False, loc="lower left"); ax[0].set_title("(a) Apple", loc="left")
yil = b.index
ax[1].bar(yil.astype(str), b["kiraz_ton"] / 1e3, color=[TURUNCU if y == 2025 else "#a0aec0" for y in yil])
ax[1].set_ylabel("Sweet cherry exports, May-July (kt)"); ax[1].tick_params(axis="x", rotation=45)
ax[1].set_title("(b) Sweet cherry", loc="left")
plt.tight_layout(); sekil_kaydet(fig, "Figure1_overview")

# Figure 2: wholesale event study
d3 = pd.read_excel(T / "tablolar_did.xlsx", sheet_name="D3_olay_calismasi")
fig, axs = plt.subplots(1, 2, figsize=(GENISLIK, 2.8))
for a_, (s_, bas) in zip(axs, [("ln Hal hacmi", "(a) Wholesale volume"), ("ln Hal fiyatı (TL)", "(b) Wholesale price")]):
    for kad, renk, kay in [("A_narenciye", MAVI, -0.12), ("B_patates_sogan", TURUNCU, 0.12)]:
        g = d3[(d3["Sonuç"] == s_) & (d3["Kontrol"] == kad)].copy()
        g["y"] = g["Pazarlama yılı"].str[:7]
        xx = np.arange(len(g)) + kay
        a_.errorbar(xx, g["γ"], yerr=[g["γ"] - g["GA alt"], g["GA üst"] - g["γ"]], fmt="o", ms=4, capsize=3,
                    color=renk, label=f"vs. {KON[kad].lower()}")
    a_.set_xticks(np.arange(len(g))); a_.set_xticklabels([v + ("*" if v == "2026/27" else "") for v in g["y"]], rotation=45)
    a_.axhline(0, color="black", lw=.6); a_.axvline(len(g) - 2.5, color="red", ls=":", lw=1)
    a_.set_ylabel("Apple minus comparison (log points)"); a_.set_title(bas, loc="left"); a_.legend(frameon=False)
plt.tight_layout(); sekil_kaydet(fig, "FigureS1_wholesale_event_study")

# Figure 3: export event study
e2 = pd.read_excel(T / "tablolar_ihracat_did.xlsx", sheet_name="E2_olay_calismasi")
fig, axs = plt.subplots(1, 3, figsize=(GENISLIK, 2.6))
for a_, (s_, bas) in zip(axs, [("ln İhracat miktarı", "(a) Export quantity"),
                               ("ln İhracat birim değeri", "(b) Export unit value"),
                               ("DDD: ln(İhr/Hal) elma − narenciye", "(c) Triple difference")]):
    g = e2[e2["Sonuç"] == s_].copy(); g["y"] = g["Pazarlama yılı"].str[:7]
    xx = np.arange(len(g))
    a_.errorbar(xx, g["γ"], yerr=[g["γ"] - g["GA alt"], g["GA üst"] - g["γ"]], fmt="o", ms=4, capsize=3, color=TURUNCU)
    a_.set_xticks(xx); a_.set_xticklabels([v + ("*" if v == "2026/27" else "") for v in g["y"]], rotation=45)
    a_.axhline(0, color="black", lw=.6); a_.axvline(len(g) - 2.5, color="red", ls=":", lw=1)
    a_.set_title(bas, loc="left")
axs[0].set_ylabel("Apple minus citrus (log points)")
plt.tight_layout(); sekil_kaydet(fig, "FigureS2_export_event_study")
print("\nŞekiller kaydedildi:", sorted(p.name for p in M.glob("*.png")))


# ===========================================================================
# ANA METİN: Table 2 (elma, tek tablo), Table 3 (kiraz, tek tablo), Figure 2 (4 panel)
# Karşılaştırma grubu: baştan planlanan tanımlar (hal: narenciye; ihracat: toplam narenciye)
# ===========================================================================
def hucre(r):
    return f"{r['β']:.3f} ({r['Std.hata (HAC)']:.3f})"
satirlar = [("Wholesale volume", d1, "ln Hal hacmi", "A_narenciye", "Post tanımı"),
            ("Wholesale price (TRY)", d1, "ln Hal fiyatı (TL)", "A_narenciye", "Post tanımı"),
            ("Export quantity", e1, "ln İhracat miktarı", "Narenciye toplamı", "Post"),
            ("Export unit value (USD)", e1, "ln İhracat birim değeri", "Narenciye toplamı", "Post"),
            ("Triple difference: exports vs. wholesale", e1, "DDD: ln(İhr/Hal) elma − narenciye", "Narenciye toplamı", "Post")]
t2 = []
for ad, df, son, kon, ps in satirlar:
    g = df[(df["Sonuç"] == son) & (df["Kontrol"] == kon)].set_index(ps)
    a_, b_ = g.loc["POST1 (≥2025-04)"], g.loc["POST2 (≥2025-08)"]
    t2.append({"Outcome": ad,
               "From April 2025: β (s.e.)": hucre(a_), "p ": p_bicim(a_["p"]), "Effect (%) ": f(a_["% etki"], 1),
               "From August 2025: β (s.e.)": hucre(b_), "p": p_bicim(b_["p"]), "Effect (%)": f(b_["% etki"], 1)})
kaydet(pd.DataFrame(t2), "Table2_apple_main",
       "Note: Equations (1)-(2); comparison group: citrus. β in log points with Newey-West HAC standard errors "
       "(6 lags) in parentheses; effect = exp(β) − 1. N = 105 (wholesale) and 104 (exports).")

t3 = []
for son, ad in [("ln İhracat miktarı", "Export quantity"), ("ln İhracat birim değeri", "Export unit value (USD)"),
                ("Üçlü fark ln(İhr/Hal)", "Triple difference: exports vs. wholesale")]:
    for sez in [2025, 2026]:
        sat_ = {"Outcome": ad, "Season": str(sez)}
        for kon, et in [("Narenciye", "Citrus (pre-registered)"), ("Kavun+karpuz", "Melon and watermelon")]:
            g = k[(k["Sonuç"] == son) & (k["Kontrol"] == kon) & (k["Tedavi"] == "Kiraz (080929)") & (k["Sezon"] == sez)]
            if len(g):
                r = g.iloc[0]
                sat_[f"{et}: β (s.e.)"] = hucre(r); sat_[f"{et}: p"] = p_bicim(r["p"])
                sat_[f"{et}: placebos ≥ |β|"] = f"{int(r['|plasebo| ≥ |β| sayısı (6 içinde)'])}/6"
            else:
                sat_[f"{et}: β (s.e.)"] = "–"; sat_[f"{et}: p"] = "–"; sat_[f"{et}: placebos ≥ |β|"] = "–"
        t3.append(sat_)
kaydet(pd.DataFrame(t3), "Table3_cherry_main",
       "Note: Equation (3); sweet cherry (HS 080929), May-July 2018-2026, N = 27. β with Newey-West HAC standard "
       "errors (2 lags). Placebos: number of 2019-2024 seasons (of 6) with an absolute coefficient at least as large. "
       "Triple difference not available for melon and watermelon (no wholesale comparison series).")

fig, axs = plt.subplots(2, 2, figsize=(GENISLIK, 5.0))
panel = [(axs[0, 0], d3, "ln Hal hacmi", "(a) Wholesale volume"),
         (axs[0, 1], d3, "ln Hal fiyatı (TL)", "(b) Wholesale price"),
         (axs[1, 0], e2, "ln İhracat miktarı", "(c) Export quantity"),
         (axs[1, 1], e2, "DDD: ln(İhr/Hal) elma − narenciye", "(d) Triple difference: exports vs. wholesale")]
for a_, df, s_, bas in panel:
    g = df[df["Sonuç"] == s_].copy()
    if "Kontrol" in g:
        g = g[g["Kontrol"] == "A_narenciye"]
    g["y"] = g["Pazarlama yılı"].str[:7]
    xx = np.arange(len(g))
    a_.errorbar(xx, g["γ"], yerr=[g["γ"] - g["GA alt"], g["GA üst"] - g["γ"]], fmt="o", ms=4, capsize=3, color=MAVI)
    a_.set_xticks(xx); a_.set_xticklabels([v + ("*" if v == "2026/27" else "") for v in g["y"]], rotation=45, fontsize=6.5)
    a_.axhline(0, color="black", lw=.6); a_.axvline(len(g) - 2.5, color="red", ls=":", lw=1)
    a_.set_title(bas, loc="left")
axs[0, 0].set_ylabel("Apple minus citrus (log points)"); axs[1, 0].set_ylabel("Apple minus citrus (log points)")
plt.tight_layout(); sekil_kaydet(fig, "Figure2_apple_event_study")
print("\nAna metin: Table1 (A,B), Table2, Table3, Figure1, Figure2. Diğerleri ek dosya (S).")


# ===========================================================================
# Table S9 (SONRADAN EKLENDİ, 02.11.2026): Aylık zamanlama — elma ihracatı, 2018-2024 aynı ay ortalamasına oran.
# Amaç: ihracat düşüşünün, kayıt şartı (RG 11.09.2025, sayı 33014) öncesinde başlayıp başlamadığını betimlemek.
# ===========================================================================
ts = pd.DataFrame({"Apple": ihr["elma_ihr_ton"], "Citrus": ihr["nar_ihr_ton"]})
on_ = ts.loc["2018-01":"2024-12"]
ort = on_.groupby(on_.index.month).mean()
oran = ts / ort.reindex(ts.index.month).values
pencere = oran.loc["2025-01":"2026-08"]
s9 = pd.DataFrame({"Month": pencere.index.strftime("%Y-%m"),
                   "Apple exports (t)": ts.loc[pencere.index, "Apple"].map(lambda v: f(v, 0)),
                   "Apple: ratio to 2018-2024 same-month mean": pencere["Apple"].map(lambda v: f(v, 2)),
                   "Citrus: ratio to 2018-2024 same-month mean": pencere["Citrus"].map(lambda v: f(v, 2))})
kaydet(s9, "TableS9_monthly_timing",
       "Note: Post hoc, descriptive. The export registration requirement for apples took effect on 11 September 2025.")
