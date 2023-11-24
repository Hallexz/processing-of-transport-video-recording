import cv2
import numpy as np

# Загрузите предварительно обученную модель и конфигурацию
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Загрузите имена классов из файла 'coco.names'
with open('datasets', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Загрузите видео
cap = cv2.VideoCapture('your_video.mp4')

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Обнаружьте объекты на кадре
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Показать информацию на экране
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # 2 - это ID класса 'car' в файле 'coco.names'
                # Получить координаты объекта
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Нарисуйте прямоугольник вокруг объекта
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)

    cv2.imshow('Real-time Car Detection', frame)

    # Выход из цикла, если пользователь нажмет 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()