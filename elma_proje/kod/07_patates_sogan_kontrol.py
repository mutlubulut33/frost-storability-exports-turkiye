# -*- coding: utf-8 -*-
"""
07_patates_sogan_kontrol.py
Patates (HS6 070190) ve kuru soğan (070310) ihracatının elma ihracatı için ek kontrol grubu
olarak uygunluğunun sınanması ve pazarlama yılı bazında betimsel karşılaştırma.

Uygunluk ölçütleri (veri incelenmeden önce, 16.10.2026 tarihli yazışmada belirlendi):
  1) Her ay anlamlı ihracat var mı? (sıfır / <100 ton ay olmamalı; log farkı bozulur)
  2) İhracat kısıtlaması izleri (ani çöküşler) var mı?
  3) Don öncesi dönemde elmayla paralel seyir (pazarlama yılı farklarının istikrarı)
Ölçütler sağlanmazsa aylık DiD yapılmaz; yalnızca betimsel karşılaştırma raporlanır.
Çalıştırma: python kod/07_patates_sogan_kontrol.py  (01-05 önce çalıştırılmış olmalı)
"""
import numpy as np
import pandas as pd
from ortak import KOK, TABLO

HAM, CIKTI = KOK / "ham_veri", KOK / "cikti"

def tuik_hs6(yol):
    d = pd.read_excel(yol, header=None, skiprows=10).iloc[:, :7]
    d.columns = ["yil", "ay", "kod", "ad", "usd", "tl", "kg"]
    d["yil"] = d["yil"].where(d["yil"].astype(str).str.fullmatch(r"\d{4}")).ffill()
    d["ay"] = d["ay"].where(d["ay"].astype(str).str.fullmatch(r"\d{1,2}")).ffill()
    d = d[d["kod"].astype(str).str.fullmatch(r"\d+")].copy()
    d["kod"] = d["kod"].astype(str).str.zfill(6)
    d[["yil", "ay"]] = d[["yil", "ay"]].astype(int)
    d["kg"] = d["kg"].astype(float)
    d["Tarih"] = pd.to_datetime(dict(year=d.yil, month=d.ay, day=1))
    return d

dosyalar = sorted(HAM.glob("TUIK_ihracat_patates_*.xlsx")) + sorted(HAM.glob("TUIK_ihracat_sogan_*.xlsx"))
ps = pd.concat([tuik_hs6(f) for f in dosyalar])
assert not ps.duplicated(["yil", "ay", "kod"]).any()
idx = pd.date_range("2018-01-01", "2026-08-01", freq="MS")
p = ps.pivot_table(index="Tarih", columns="kod", values="kg", aggfunc="sum").reindex(idx).fillna(0) / 1e3

e = pd.read_csv(CIKTI / "veri_ihracat_did.csv", index_col=0, encoding="utf-8-sig")
e.index = pd.PeriodIndex(e.index, freq="M").to_timestamp()
x = pd.DataFrame({"Elma": e["elma_ihr_ton"], "Narenciye": e["nar_ihr_ton"],
                  "Patates": p["070190"], "Soğan": p["070310"]})
x["Patates+Soğan"] = x["Patates"] + x["Soğan"]

# Ölçüt 1-2: düşük ihracatlı aylar
U1 = pd.DataFrame({"Ürün": x.columns,
                   "Ay sayısı": len(x),
                   "<100 ton olan ay": [(x[c] < 100).sum() for c in x],
                   "Bu aylar": [", ".join(x.index[x[c] < 100].strftime("%Y-%m")) for c in x]})
print(U1.to_string(index=False))

# Pazarlama yılı (Ağustos-Temmuz) toplamları; 2026/27 eksik olduğu için dahil değil
py = np.where(x.index.month >= 8, x.index.year, x.index.year - 1)
a = (x.groupby(py).sum() / 1e3).loc[2018:2025]
a.index = [f"{i}/{str(i + 1)[-2:]}" for i in a.index]
on = ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24"]
oran = a.div(a.loc[on].mean())
fark = pd.DataFrame({f"ln(Elma) − ln({k})": np.log(a["Elma"]) - np.log(a[k])
                     for k in ["Narenciye", "Patates+Soğan"]})
fark_ozet = pd.DataFrame({"Don öncesi (2018/19-2023/24) min": fark.loc[on].min(),
                          "Don öncesi max": fark.loc[on].max(),
                          "Don öncesi ort.": fark.loc[on].mean(),
                          "2025/26": fark.loc["2025/26"],
                          "2025/26 − don öncesi ort.": fark.loc["2025/26"] - fark.loc[on].mean()})
print("\nPazarlama yılı ihracatı (bin ton):"); print(a.round(1).to_string())
print("\nDon öncesi ortalamaya oran:"); print(oran.round(2).to_string())
print("\nLog farkları:"); print(fark.round(2).to_string()); print(fark_ozet.round(2).to_string())

with pd.ExcelWriter(TABLO / "tablolar_patates_sogan.xlsx") as xw:
    U1.to_excel(xw, sheet_name="U1_dusuk_aylar", index=False)
    a.to_excel(xw, sheet_name="U2_pazarlama_yili_bin_ton")
    oran.to_excel(xw, sheet_name="U3_don_oncesi_orana_gore")
    fark.to_excel(xw, sheet_name="U4_log_farklari")
    fark_ozet.to_excel(xw, sheet_name="U5_log_fark_ozet")
print("\nKaydedildi.")
