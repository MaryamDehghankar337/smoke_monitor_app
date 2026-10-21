# Sentinel Smoke & Air Pollution Monitor

اپ Streamlit برای پایش دود و آلاینده‌های جوی در خلیج فارس و تنگه هرمز با استفاده از Sentinel-2 و Sentinel-5P.

## امکانات

- جست‌وجوی موقعیت جغرافیایی با Nominatim
- انتخاب و رسم محدوده مطالعه روی نقشه
- جست‌وجوی تصاویر Sentinel-2 از CDSE STAC
- ورود به Copernicus Data Space Ecosystem
- محاسبه ماسک دود Sentinel-2 بر پایه شاخص دود و ماسک ابر، سایه، برف و آب
- نمایش نقشه ماسک دود، شاخص دود و پیکسل‌های معتبر
- دانلود GeoTIFF و بسته PNG زمین‌مرجع
- انتخاب تیک‌دار آلاینده‌های Sentinel-5P:
  - CH4
  - CO
  - NO2
  - SO2
  - O3
  - HCHO
  - UV Aerosol Index
- ساخت سری زمانی روزانه آلاینده‌ها
- دانلود جدول سری زمانی به صورت CSV

## اجرا

```bash
python -m venv venv
venv\Scripts\activate       # ویندوز
source venv/bin/activate    # لینوکس یا مک

pip install -r requirements.txt
streamlit run smoke_monitor_app.py
```

## نکته علمی

ماسک دود Sentinel-2 در این نسخه یک ماسک کاندید دود است و به‌طور قطعی دود را تأیید نمی‌کند.
ابر، سایه، برف و آب با استفاده از لایه SCL و شاخص‌های طیفی حذف می‌شوند.
