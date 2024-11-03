import cv2
from utils import detect_and_draw_orientation, pallet_analysis

# Mở camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gọi hàm để phát hiện và vẽ các đường định hướng
    frame_with_orientation = detect_and_draw_orientation(frame)

    # Gọi hàm phân tích để lấy thông tin chi tiết các pallet (nếu cần)
    rectangles = pallet_analysis(frame)

    # Hiển thị số lượng hình chữ nhật phát hiện được (phục vụ kiểm tra)
    print(f"Number of detected rectangles: {len(rectangles)}")

    # Hiển thị ảnh với các kết quả
    cv2.imshow('Pallet Detection and Orientation', frame_with_orientation)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
