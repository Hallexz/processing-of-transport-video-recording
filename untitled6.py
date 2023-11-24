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



def create_model():
    """Создание модели"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model

def compile_model(model):
    """Компиляция модели"""
    opt = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


def train_model(model, x_train, y_train, x_val, y_val):
    """Обучение модели"""
    callbacks = set_callbacks()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=callbacks)

    return history

def print_history(history):
    """Вывод истории обучения в консоли"""
    print("History of training:\n", history.history)

def process_video(video_path, json_directory):
    """Обработка видео"""
    json_data = getJson(video_path, json_directory)
    if json_data is None:
        return

    cap = cv2.VideoCapture(video_path)
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    processed_frames = []
    # Путь для сохранения первого кадра
    save_path = r"C:\Users\diner\Desktop"

    for i, frame in enumerate(frames):
        # Конвертируем координаты из процентов в пиксели
        areas = [[[int(x[0]*frame.shape[1]), int(x[1]*frame.shape[0])] for x in area] for area in json_data['areas']]
        zones = [[[int(x[0]*frame.shape[1]), int(x[1]*frame.shape[0])] for x in zone] for zone in json_data['zones']]

        # Инициализируем трекеры для каждой области
        trackers = [cv2.Tracker for _ in areas]
        for tracker, area in zip(trackers, areas):
            bbox = cv2.boundingRect(np.array(area))
            tracker.init(frame, bbox)

for tracker, area in zip(trackers, areas):
            # Обновляем трекер и получаем новую область
            ret, bbox = tracker.update(frame)

            if ret:
                # Рисуем область на кадре
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Выводим сообщение, если отслеживание не удалось
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Рисуем зоны
        for zone in zones:
            cv2.polylines(frame, [np.array(zone)], True, (0, 255, 0), 2)

        # Сохраняем первый кадр после сортировки областей и зон
        if i == 0:
            cv2.imwrite(os.path.join(save_path, 'first_frame.jpg'), frame)

        frame = cv2.resize(frame, (128, 128)) # Изменено на 128x128 для соответствия входу модели
        frame = image.img_to_array(frame)
        frame = frame/255
        processed_frames.append(frame)

    processed_frames = np.array(processed_frames)
    predictions = model.predict(processed_frames)

    return predictions

# Создание модели
model = create_model()

# Компиляция модели
compile_model(model)


#   # Предположим, что     'images' - это ваш массив изображений размером (num_images, 128, 128, 3),
#   # а                    'labels' - это метки классов для этих изображений, где каждая метка - это число от 0 до 2.
#   
#   # Преобразуем метки классов в формат one-hot encoding
#   labels = to_categorical(labels, num_classes=3)
#   
#   # Разделяем данные на обучающую и валидационную выборки
#   x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
#   
#   # Обучаем модель
#   history = train_model(model, x_train, y_train, x_val, y_val)



#   # Вывод истории обучения в консоли
#   print_history(history) 




# Использование функций
json_data = getJson(r"C:\Users\diner\Desktop\KRA-2-7-2023-08-23-evening.mp4", r"D:\DATA\markup\jsons")
predictions = process_video(r"C:\Users\diner\Desktop\KRA-2-7-2023-08-23-evening.mp4")
print(predictions)