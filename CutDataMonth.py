#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kış aylarından (Aralık/Ocak/Şubat) **tek bir AYLık** veriyi süzer ve dosyaya kaydeder.

Varsayılanlar:
  - Girdi: C:\Users\erenf\FEKE.csv  (yoksa C:\Users\erenf\FEKE.xlsx denenir)
  - Yıl/Ay: 2024 / Ocak
  - Zaman kolonu: 'DateTime' (yoksa yaygın alternatif adları dener)
  - Çıktı: C:\Users\erenf\FEKE_2024_01.csv  ( --excel ile .xlsx )

Örnek kullanım (isteğe bağlı):
  # CSV -> CSV
  python CutDataMonth.py --input C:\Users\erenf\FEKE.csv --year 2023 --month dec --out C:\Users\erenf\FEKE_2023_12.csv
  # Excel -> XLSX
  python CutDataMonth.py --input C:\Users\erenf\FEKE.xlsx --sheet Sheet1 --year 2024 --month jan --excel
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# ----- Varsayılan yollar -----
DEFAULT_INPUTS = [r"C:\Users\erenf\FEKE.csv", r"C:\Users\erenf\FEKE.xlsx"]
DEFAULT_YEAR   = 2024
DEFAULT_MONTH  = "jan"   # 1 / 2 / 12 veya jan/feb/dec
DEFAULT_OUT_CSV  = r"C:\Users\erenf\FEKE_2024_01.csv"
DEFAULT_OUT_XLSX = r"C:\Users\erenf\FEKE_2024_01.xlsx"

# ----- Yardımcılar -----
def _read_csv_smart(p):
    """CSV'de önce virgül, sonra noktalı virgül, ardından sniff dener."""
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep=";")
        except Exception:
            import csv
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(8192)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    delim = dialect.delimiter
                except Exception:
                    delim = ","
            return pd.read_csv(p, sep=delim, engine="python")

def read_any(path, sheet=None):
    """CSV veya Excel'i oku."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl  # Excel motoru
        except Exception as e:
            sys.exit("Excel okumak için 'openpyxl' gerekli. Kurulum: pip install openpyxl\nAyrıntı: %s" % e)
        return pd.read_excel(p, sheet_name=(sheet if sheet else 0), engine="openpyxl")
    else:
        return _read_csv_smart(p)

def detect_time_column(df, user_col=None):
    """Kullanıcı belirttiyse onu, yoksa yaygın isimlerden birini kullan."""
    if user_col:
        if user_col in df.columns:
            return user_col
        sys.exit(f"--timecol='{user_col}' bulunamadı. Kolonlar: {list(df.columns)}")
    for c in ["DateTime", "datetime", "timestamp", "time", "date"]:
        if c in df.columns:
            return c
    sys.exit(f"Zaman kolonu bulunamadı. Kolonlar: {list(df.columns)}\n"
             f"Lütfen tarih/saat kolonunu 'DateTime' olarak adlandırın veya --timecol ile belirtin.")

def parse_month(m):
    """--month değerini (1/2/12 veya jan/feb/dec) 1..12'ye çevirir ve kış ayı kontrolü yapar."""
    m_str = str(m).strip().lower()
    alias = {
        "1": 1, "01": 1, "jan": 1, "ocak": 1,
        "2": 2, "02": 2, "feb": 2, "subat": 2, "şubat": 2,
        "12": 12, "dec": 12, "aralik": 12, "aralık": 12
    }
    if m_str not in alias:
        sys.exit("--month sadece {1/jan, 2/feb, 12/dec} olabilir.")
    month = alias[m_str]
    if month not in (12, 1, 2):
        sys.exit("Sadece kış ayları (Aralık=12, Ocak=1, Şubat=2) desteklenir.")
    return month

def month_bounds(year, month):
    """Ayın [start, end) zaman aralığını döndürür."""
    start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0, second=0)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0, second=0)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1, hour=0, minute=0, second=0)
    return start, end

def find_existing_default_input():
    for s in DEFAULT_INPUTS:
        if Path(s).exists():
            return s
    return None

# ----- Ana Akış -----
def main():
    ap = argparse.ArgumentParser(description="Kış aylarından seçilen bir ayı (Aralık/Ocak/Şubat) filtreler ve kaydeder.")
    ap.add_argument("--input", "-i", default=None, help="Girdi dosyası (CSV/XLSX). Verilmezse FEKE.csv/xlsx aranır.")
    ap.add_argument("--sheet", default=None, help="Excel sayfa adı (gerekirse).")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR, help=f"Yıl (varsayılan: {DEFAULT_YEAR}).")
    ap.add_argument("--month", default=DEFAULT_MONTH, help=f"Ay: 1/2/12 veya jan/feb/dec (varsayılan: {DEFAULT_MONTH}).")
    ap.add_argument("--timecol", default=None, help="Zaman kolonu adı (varsayılan: DateTime vb.).")
    ap.add_argument("--out", "-o", default=None, help="Çıktı yolu. Verilmezse otomatik oluşturulur.")
    ap.add_argument("--excel", action="store_true", help="CSV yerine .xlsx olarak yaz.")
    args = ap.parse_args()

    # Girdi yolu
    in_path = args.input
    if not in_path:
        in_path = find_existing_default_input()
        if not in_path:
            msg = "Girdi bulunamadı. Lütfen --input verin veya şu yollardan biri mevcut olsun:\n" + "\n".join(DEFAULT_INPUTS)
            sys.exit(msg)
    in_path = Path(in_path)
    print(f"[INFO] Girdi: {in_path}")

    # Oku
    df = read_any(str(in_path), args.sheet)
    print(f"[INFO] Okundu: {len(df)} satır")

    # Zaman kolonu
    timecol = detect_time_column(df, args.timecol)

    # Tarihe çevir + geçersizleri at
    t = pd.to_datetime(df[timecol], errors="coerce")
    before = len(df)
    df = df.loc[~t.isna()].copy()
    if len(df) < before:
        print(f"[INFO] Geçersiz tarih/saat içeren {before - len(df)} satır atlandı.")
    t = pd.to_datetime(df[timecol], errors="coerce")

    # Ay/yıl + aralık
    month = parse_month(args.month)
    start, end = month_bounds(args.year, month)

    # Filtre
    mask = (t >= start) & (t < end)
    out_df = df.loc[mask].copy()
    print(f"[RESULT] Seçilen aralık: {start} .. {end}  →  satır: {len(out_df)}")

    # Çıktı yolu
    if args.out:
        out_path = Path(args.out)
    else:
        suffix = ".xlsx" if args.excel else ".csv"
        out_name = f"{in_path.stem}_{start.year:04d}_{start.month:02d}{suffix}"
        out_path = in_path.parent / out_name

        # Varsayılanla birebir aynı olsun istersen:
        # out_path = Path(DEFAULT_OUT_XLSX if args.excel else DEFAULT_OUT_CSV)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Yaz
    if args.excel:
        try:
            import openpyxl  # güvence
        except Exception:
            print("[WARN] 'openpyxl' bulunamadı, CSV kaydediliyor.")
            out_path = out_path.with_suffix(".csv")
            out_df.to_csv(out_path, index=False)
        else:
            out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    print(f"[DONE] Yazıldı: {out_path}")

if __name__ == "__main__":
    main()
