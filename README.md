# E-Ticarette Müşteri Kaybı Tahmini

E-ticaret müşterilerinin müşteri kaybı (`Churn`) durumunu tahmin etmek için **Logistic Regression** ve **Random Forest** modellerini karşılaştıran Python projesi. Veri hazırlama, model eğitimi ve iki farklı doğrulama yöntemiyle performans değerlendirmesi içerir.

## Amaç

Müşteri davranışlarını ve alışveriş özelliklerini kullanarak müşteri kaybını sınıflandırmak, iki modelin başarısını karşılaştırmak ve sınıflandırma metriklerini yorumlamak. Proje eğitim ve model karşılaştırması amacıyla hazırlanmıştır.

## Kullanılan teknolojiler

- Python
- pandas ve NumPy
- scikit-learn
- Matplotlib ve Seaborn

## Dosyalar

| Dosya | Açıklama |
| --- | --- |
| `veri_dersi_müşteri_kaybı.py` | Veri hazırlama, eğitim, değerlendirme ve grafikler |
| `e_commerce_dataset_temizlenmis.csv` | Kodun kullandığı veri dosyası |
| `requirements.txt` | Python bağımlılıkları |
| `.gitignore` | Yerel ortam ve geçici dosyalar için Git hariç tutma kuralları |

## Veri seti

Hedef sütun `Churn`, müşteri tanımlayıcı sütunu `CustomerID`'dir. Veri dosyasında müşteri kıdemi (`Tenure`), giriş cihazı, ödeme yöntemi, tercih edilen ürün kategorisi, memnuniyet puanı, şikâyet durumu, sipariş sayısı ve son siparişten bu yana geçen süre gibi alanlar bulunur.

`CustomerID` model girdilerinden çıkarılır. Veri setinin özgün kaynak ve lisans bilgileri proje dosyalarında belirtilmemiştir.

## Çalışma akışı

1. CSV dosyası okunur.
2. Aynı kategoriyi ifade eden bazı değerler birleştirilir: `Phone` → `Mobile Phone`, `CC` → `Credit Card`, `COD` → `Cash on Delivery` ve ürün kategorisindeki `Mobile` → `Mobile Phone`.
3. Eksik değer içeren sütunlar kendi medyanlarıyla doldurulur.
4. `CustomerID` kaldırılır; metin sütunları `LabelEncoder` ile sayısallaştırılır.
5. Veriler %80 eğitim, %20 test olarak ayrılır (`random_state=42`).
6. Logistic Regression (`max_iter=2000`) ve Random Forest (`n_estimators=100`, `random_state=42`) eğitilir.
7. Test sonuçları ve karmaşıklık matrisleri gösterilir.
8. Her iki model 5 katlı, karıştırılmış `KFold` ile ayrıca değerlendirilir.

## Kurulum ve çalıştırma

Terminali proje klasöründe açın. Python ve pip kurulu olmalıdır. Windows PowerShell için:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe "veri_dersi_müşteri_kaybı.py"
```

macOS / Linux için:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python "veri_dersi_müşteri_kaybı.py"
```

CSV dosyasını Python dosyasıyla aynı klasörde tutun ve komutu bu klasörden çalıştırın. Grafiklerin gösterilebilmesi için grafik arayüzünü destekleyen bir Python ortamı gerekir. Program grafik penceresinde beklerse devam etmek için pencereyi kapatın.

## Çıktılar ve değerlendirme

Her model için terminalde şu metrikler yazdırılır:

| Metrik | Ne anlatır? |
| --- | --- |
| Accuracy | Tüm tahminlerin ne kadarının doğru olduğu |
| Precision | Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu |
| Recall | Gerçek pozitiflerin ne kadarının yakalandığı |
| F1-score | Precision ve recall'un harmonik ortalaması |

Hold-out değerlendirmesinde her model için bir karmaşıklık matrisi açılır. Çapraz doğrulamada ise beş katın ortalama metrikleri yazdırılır. Sonuçlar iki ondalık basamakla gösterilir; model veya rapor dosyası otomatik kaydedilmez.

Bu README'ye doğrulanmış bir çalışma çıktısı eklenmediğinden sayısal başarı iddiası verilmemiştir. Sonuçları kendi ortamınızda kodu çalıştırarak elde edebilirsiniz. Bağımlılık sürümleri sabitlenmediği için ortamlar arasında sonuç farkları oluşabilir.

## Yöntemin sınırlamaları

- Eksik değer doldurma ve kategorik kodlama veri bölünmeden önce yapılır. Özellikle medyanların tüm veriden hesaplanması değerlendirmeye bilgi sızmasına yol açabilir. Daha sağlam bir değerlendirmede ön işleme her eğitim bölümü içinde öğrenilmelidir.
- `LabelEncoder`, kategoriler arasında sayısal bir sıralama oluşturur. Özellikle Logistic Regression için sırasız kategorilerde one-hot kodlama tercih edilebilir.
- Logistic Regression öncesinde sayısal özellikler ölçeklenmez.
- Hold-out bölmesinde `stratify` kullanılmaz; çapraz doğrulama `StratifiedKFold` yerine `KFold` kullanır. Sınıf oranlarının korunması garanti edilmez.
- Çapraz doğrulama tüm veri üzerinde yapılır; hold-out sonuçlarından tamamen bağımsız ikinci bir deney olarak yorumlanmamalıdır.

Mevcut çalışma bir eğitim projesidir; üretim ortamına hazır bir müşteri karar sistemi olarak değerlendirilmemelidir.
