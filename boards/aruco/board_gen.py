import cv2
import numpy as np
import os

# --- AYARLAR ---
DICT_TYPE = cv2.aruco.DICT_4X4_1000
TOTAL_MARKERS = 1000
MARKERS_PER_ROW = 8
MARKERS_PER_COL = 10
MARKERS_PER_PAGE = MARKERS_PER_ROW * MARKERS_PER_COL # Sayfa başı 80 marker

# A4 300 DPI Piksel Değerleri
A4_WIDTH = 2480
A4_HEIGHT = 3508

# Marker ve Yazı Boyutları (Piksel)
MARKER_SIZE = 200 # 2cm'ye yakın baskı için
GAP = 60          # Markerlar arası boşluk
TEXT_SPACE = 40   # Yazı için ayrılan alan

def generate_marker_pages():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    
    if not os.path.exists("all_markers"):
        os.makedirs("all_markers")

    current_id = 0
    page_num = 1

    while current_id < TOTAL_MARKERS:
        # Beyaz sayfa oluştur
        page = np.ones((A4_HEIGHT, A4_WIDTH), dtype=np.uint8) * 255
        
        for row in range(MARKERS_PER_COL):
            for col in range(MARKERS_PER_ROW):
                if current_id >= TOTAL_MARKERS:
                    break
                
                # 1. Marker'ı oluştur
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, current_id, MARKER_SIZE)
                
                # 2. Yerleşimi hesapla
                x_start = 100 + col * (MARKER_SIZE + GAP)
                y_start = 100 + row * (MARKER_SIZE + GAP + TEXT_SPACE)
                
                # 3. Marker'ı sayfaya yapıştır
                page[y_start:y_start+MARKER_SIZE, x_start:x_start+MARKER_SIZE] = marker_img
                
                # 4. ID Yazısını ekle
                text = f"ID: {current_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Yazıyı ortala
                text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
                text_x = x_start + (MARKER_SIZE - text_size[0]) // 2
                text_y = y_start + MARKER_SIZE + 35
                
                cv2.putText(page, text, (text_x, text_y), font, 0.8, (0), 2, cv2.LINE_AA)
                
                current_id += 1

        # Sayfayı kaydet
        filename = f"all_markers/markers_page_{page_num}.png"
        cv2.imwrite(filename, page)
        print(f"✅ Sayfa {page_num} kaydedildi ({current_id}/1000)")
        page_num += 1

if __name__ == "__main__":
    print("🚀 1000 Marker üretimi başlatılıyor...")
    generate_marker_pages()
    print("🏁 İşlem tamamlandı. 'all_markers' klasörüne bakabilirsin.")