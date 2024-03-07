# Скрипт для увелечения/уменьшения количества fps на видео до 60
import cv2
cap = cv2.VideoCapture('/Users/kirill/Desktop/REC-20240304213159.mp4')
output_file = ('/Users/kirill/Desktop/1000_cohsep.mp4')
output_fps = 60.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps, (int (cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()