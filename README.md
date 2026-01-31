# ArUco / ChArUco Çalışma Alanı

ArUco marker üretimi, ChArUco ile kamera kalibrasyonu ve canlı görüntüde poz kestirimi için basit bir çalışma alanı.

## Gereksinimler
- Python 3.x
- `opencv-contrib-python`
- `numpy`

Örnek kurulum:
```bash
pip install opencv-contrib-python numpy
```

## Kullanım (Özet)
> Not: Script’ler çoğunlukla **çalıştırıldıkları dizine göre** dosya yolları kullanır.

### 1) ChArUco board üretimi (isteğe bağlı)
```bash
cd boards
python checkerboard_gen.py
```

### 2) Kalibrasyon için fotoğraf çekimi
```bash
cd calibration
python capture_img.py
```

### 3) Kamera kalibrasyonu
```bash
cd calibration
python calibration.py
```

### 4) Marker poz kestirimi (kalibrasyon gerekli)
Bu aşama sonunda kameranın markerlara göre konumunu ve pozisyonunu öğrenebiliyoruz.
```bash
cd detection
python indv_pose_estimation.py
```

İsteğe bağlı: Board (GridBoard) poz kestirimi için `detection/board_pose_estimation.py` kullanılabilir.

## Faydalı Scriptler
- `detection/detect_aruco_camera.py`: Canlı ArUco tespiti.
- `detection/detect_aruco_images.py`: Tek görselde ArUco tespiti.
