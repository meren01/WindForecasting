import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Dosyayı oku
print("Dosya okunuyor...")
df = pd.read_csv(r"C:\Users\erenf\FEKE_2024_01.csv")

# --- EKSTRA GÜVENLİK ADIMI: ZAMAN SIRALAMASI ---
# Eğer Excel içinde satırlar karışıksa, bölmeden önce sıraya dizelim.
# Böylece "İlk %80" gerçekten "Ayın başı" olur.
if "DateTime" in df.columns:
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    print("Veri tarih sırasına göre dizildi (Garantilendi).")

# 2. Karıştırmadan %80 - %20 Böl (İSTEDİĞİN GİBİ DÜZELTİLDİ)
# shuffle=False -> Karıştırma YOK. Sırayı bozma.
# random_state'e gerek yok çünkü rastgelelik yok.
df_train, df_test = train_test_split(df, test_size=0.20, shuffle=False)

# 3. Dosyaları kaydet
df_train.to_csv(r"C:\Users\erenf\feke_80.csv", index=False)
df_test.to_csv(r"C:\Users\erenf\feke_20.csv", index=False)

print(f"-"*30)
print(f"İŞLEM TAMAM (Strict Split - Sızıntı Yok)")
print(f"Toplam Veri: {len(df)}")
print(f"Eğitim (feke_80): {len(df_train)} satır (İlk 26 Gün)")
print(f"Test (feke_20)  : {len(df_test)} satır (Son 4 Gün)")
print(f"-"*30)
print(f"UYARI: Bu dosyaları kullanırken modeline 'Lag' (Prev_Speed) eklemeyi unutma!")
print(f"Çünkü son 4 günün havası çok farklı olduğu için modelin geçmiş veriye ihtiyacı var.")