import cv2
import numpy as np
import imutils

# Hàm để phát hiện và vẽ các đường định hướng pallet
def detect_and_draw_orientation(frame, small_rectangles):
    for rect_info in small_rectangles:
        center, width, height, angle = rect_info

        # Xác định xem hình chữ nhật là horizontal hay vertical
        if width > height:
            main_horizontal = rect_info
            draw_orientation_line(frame, center, width, angle, "Horizontal", (255, 0, 0))
        else:
            main_vertical = rect_info
            draw_orientation_line(frame, center, height, angle + 90, "Vertical", (0, 0, 255))

    return frame

# Hàm phụ để vẽ các đường định hướng
def draw_orientation_line(frame, center, length, angle, label, color):
    start_point = (int(center[0] - length * np.cos(np.radians(angle))),
                   int(center[1] - length * np.sin(np.radians(angle))))
    end_point = (int(center[0] + length * np.cos(np.radians(angle))),
                 int(center[1] + length * np.sin(np.radians(angle))))
    cv2.line(frame, start_point, end_point, color, 2)
    cv2.putText(frame, label, (start_point[0] + 10, start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# Hàm phân tích thông tin của pallet
def pallet_analysis(frame, min_area=1000, max_area=2500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    processed_edges = cv2.erode(dilated, kernel, iterations=1)

    contours = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    detected_rectangles = []  # Danh sách lưu các thông tin hình chữ nhật nhỏ

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect
            detected_rectangles.append((center, width, height, angle))
    
    return detected_rectangles

