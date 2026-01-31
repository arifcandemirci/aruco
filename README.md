# ArUco / ChArUco Çalışma Alanı

Bu depo; ArUco marker üretimi, ChArUco tabanlı kamera kalibrasyonu ve canlı görüntü üzerinde marker/board poz kestirimi için hazırlanmış bir çalışma alanıdır. Aşağıda tüm dosyaların ne işe yaradığı ve nasıl kullanılacağı özetlenmiştir.

## Gereksinimler
- Python 3.x
- OpenCV (contrib sürümü gerekli): `opencv-contrib-python`
- NumPy

Örnek kurulum:
```bash
pip install opencv-contrib-python numpy
```

## Klasör Yapısı ve Dosyalar

### Kökte
- `README.md`: Bu doküman.
- `A4_Navigasyon_Board_2cm.png`: A4 üzerine basılabilir, 2 cm ölçekli hazır navigasyon board görseli.

### `calibration/`
Kamera kalibrasyonu için ChArUco board kullanımı.
- `calibration/calibration.py`: ChArUco görüntülerinden kalibrasyon yapar, sonuçları `calibration_matrix.npy` ve `distortion_coefficients.npy` olarak kaydeder.
- `calibration/capture_img.py`: Kameradan belirli aralıklarla otomatik fotoğraf çekip `calibration_images/` içine kaydeder.
- `calibration/calibration_images/*.jpg`: Örnek kalibrasyon fotoğrafları (24 adet).
- `calibration/calibration_matrix.npy`: Kalibrasyon sonucu kamera matrisi.
- `calibration/distortion_coefficients.npy`: Kalibrasyon sonucu bozulma katsayıları.

### `detection/`
ArUco tespiti ve poz kestirimi.
- `detection/utils.py`: ArUco sözlük haritası ve çizim yardımcı fonksiyonları.
- `detection/detect_aruco_images.py`: Tek bir resimde ArUco tespiti.
- `detection/detect_aruco_camera.py`: Kameradan canlı ArUco tespiti.
- `detection/indv_pose_estimation.py`: Tek marker için poz kestirimi (kalibrasyon dosyaları gerekir).
- `detection/board_pose_estimation.py`: GridBoard (board) poz kestirimi (kalibrasyon dosyaları gerekir).

### `boards/`
Marker ve board üretim/görsel dosyaları.
- `boards/marker_gen.py`: Tek marker üretir (`markers/marker_0.png`).
- `boards/checkerboard_gen.py`: ChArUco board üretir (`charuco_real.png`).
- `boards/board_gen.py`: 1000 adet ArUco marker’ı A4 sayfalara dağıtır (`all_markers/`).
- `boards/charucoboard_original_10_5_0.21_0.297_0.04_0_meters.pdf`: Örnek hazır ChArUco PDF.
- `boards/charuco_real.png`: Üretilmiş ChArUco görseli.
- `boards/hizalama_board.png`: Hizalama için kullanılan board görseli.
- `boards/markers/marker_0.png`: Üretilmiş tek marker örneği.
- `boards/markers/singlemarkerssource.jpg`: Kaynak görsel.

### `all_markers/`
- `markers_page_*.png`: `board_gen.py` tarafından üretilen A4 sayfalık marker sayfaları.

### Diğer
- `__pycache__/` ve `detection/__pycache__/`: Python derleme önbellek dosyaları.

## Kullanım

> Not: Script’ler çoğunlukla **çalıştırıldıkları dizine göre** dosya yolları kullanır. Bu yüzden belirtilen dizinlerde çalıştırmanız önemlidir.

### 1) ChArUco board üretimi (isteğe bağlı)
```bash
cd boards
python checkerboard_gen.py
```
`boards/charuco_real.png` dosyası oluşur. Bunu A4’e doğru ölçekte basarak kalibrasyon için kullanın.

### 2) Kalibrasyon için fotoğraf çekimi
```bash
cd calibration
python capture_img.py
```
`calibration/calibration_images/` içine 24 adet görsel kaydedilir.

### 3) Kamera kalibrasyonu
```bash
cd calibration
python calibration.py
```
Başarılı olursa `calibration_matrix.npy` ve `distortion_coefficients.npy` üretilir.

### 4) Tek marker tespiti (resim üzerinden)
```bash
cd detection
python detect_aruco_images.py --image /path/to/image.png --type DICT_4X4_50
```
Not: Kod içinde sözlük sabit olarak `DICT_4X4_50` seçilmiş. `--type` argümanı şu an etkisiz.

### 5) Canlı ArUco tespiti (kamera)
```bash
cd detection
python detect_aruco_camera.py
```
`q` ile çıkabilirsiniz.

### 6) Tek marker poz kestirimi (kalibrasyon gerekli)
```bash
cd detection
python indv_pose_estimation.py
```
Terminale `x,y,z` konumları basılır ve görüntüde eksenler çizilir.

### 7) Board (GridBoard) poz kestirimi (kalibrasyon gerekli)
```bash
cd detection
python board_pose_estimation.py
```
Board tespit edilince `x,y,z` ve açı bilgisi ekrana yazdırılır.

## Parametreler
Kalibrasyon ve tespit doğruluğu için şu sabitleri kendi donanımınıza göre ayarlamanız gerekebilir:
- `calibration/calibration.py`: `BOARD_SIZE`, `CHARUCO_SQUARE_SIZE`, `CHARUCO_MARKER_SIZE`, `ARUCO_DICT_TYPE`
- `detection/indv_pose_estimation.py`: `MARKER_LENGTH`
- `detection/board_pose_estimation.py`: `MARKER_LENGTH`, `SEPARATION`, `BOARD_SIZE`

## Notlar
- OpenCV ArUco/ChArUco fonksiyonları için **opencv-contrib-python** gereklidir.
- Kamera erişimi Linux’ta V4L2 ile yapılmıştır; farklı cihazlarda `VideoCapture` parametrelerini güncellemeniz gerekebilir.
