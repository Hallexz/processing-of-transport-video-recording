import json
import os
import pathlib
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np




def getJson(json_path, directory):
    """Функция для извлечения данных из JSON файла."""
    filename1= pathlib.Path(video_path).stem
    json_file = os.path.join(directory, filename + '.json')

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'areas' in data:
                print('Areas:')
                print(json.dumps(data['areas'], indent=4))
            if 'zones' in data:
                print('Zones:')
                print(json.dumps(data['zones'], indent=4))
        return json_file
    else:
        print(f'Файл {json_file} не найден')
        

directory1 = '/home/hallex/Документы/hattt/datasets'

for filename in os.listdir(directory1):
    if filename.endswith(".mp4"):
        video_path = os.path.join(directory1, filename)
        cap = cv2.VideoCapture(video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                cv2.imshow('Frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break

        cap.release()
        cv2.destroyAllWindows()
        
# Инициализируйте глобальные переменные
start_point = (0, 0)
end_point = (0, 0)
drawing = False

# Функция для рисования прямоугольника на изображении
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image_copy = image.copy()
            cv2.rectangle(image_copy, start_point, (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

# Создайте окно и назначьте функцию обратного вызова
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', image_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
        
        


