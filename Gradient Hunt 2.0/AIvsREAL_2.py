!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c induction-task-2025
!unzip /content/induction-task-2025.zip
!kaggle competitions download -c induction-task
!unzip induction-task.zip
!unzip /content/New_Data.zip

import os
import cv2

def merge_images_from_folders(folder1, folder2, output_folder, resize_to=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    
    def process_and_save_images(input_folder):
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, resize_to)
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, img_resized)
    
    process_and_save_images(folder1)
    process_and_save_images(folder2)

folder1_path = '/content/Data/Train/AI'
folder2_path = '/content/New_Data/AI'
output_folder_path = '/content/AI_Combine'
merge_images_from_folders(folder1_path, folder2_path, output_folder_path)

folder1_path = '/content/Data/Train/Real'
folder2_path = '/content/New_Data/Real'
output_folder_path = '/content/REAL_Combine'
merge_images_from_folders(folder1_path, folder2_path, output_folder_path)

import numpy as np
from PIL import Image
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_images(image_folder, image_size=(224, 224)):
    image_array_list = []
    sorted_filenames = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))],
        key=extract_number
    )
    for filename in sorted_filenames:
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize(image_size)
        image_array_list.append(np.array(img))
    return np.stack(image_array_list)

aio_array = load_images("/content/AI_Combine")
real_array = load_images("/content/REAL_Combine")

ai_labels = np.zeros(len(ai_array))
real_labels = np.ones(len(real_array))
X = np.concatenate((ai_array, real_array), axis=0)
Y = np.concatenate((ai_labels, real_labels), axis=0)

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=42)
X = X / 255.0

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

test_array = load_images("/content/Test_Images")
test_array = test_array / 255.0

predict2 = model.predict(test_array)

import csv
image_ids = [f'image_{i}' for i in range(200)]
labels = ['Real' if pred > 0.5 else 'AI' for pred in predict2]

with open('submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'Label'])
    for img_id, label in zip(image_ids, labels):
        writer.writerow([img_id, label])
