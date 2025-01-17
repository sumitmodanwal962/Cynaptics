!pip install -q kaggle

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c induction-task

!unzip induction-task.zip

import os
import cv2

image_folder = "/content/Data/Train/Real"
output_folder = "/content/Real_resized"
os.makedirs(output_folder, exist_ok=True)

new_size = (224, 224)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(output_folder, filename), img_resized)

print("All images have been reshaped and saved.")

image_folder = "/content/Data/Train/AI"
output_folder = "/content/AI_resized"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(output_folder, filename), img_resized)

print("All images have been reshaped and saved.")

import numpy as np
from PIL import Image

image_folder = "/content/Real_resized"
image_array_list = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        image_array_list.append(img_array)

real_array = np.stack(image_array_list)
print(f"Final array shape: {real_array.shape}")

image_folder = "/content/AI_resized"
image_size = (224, 224)
image_array_list = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize(image_size)
        img_array = np.array(img)
        image_array_list.append(img_array)

ai_array = np.stack(image_array_list)
print(f"Final array shape: {ai_array.shape}")

ai_labels = np.zeros(len(ai_array))
real_labels = np.ones(len(real_array))

X = np.concatenate((ai_array, real_array), axis=0)
Y = np.concatenate((ai_labels, real_labels), axis=0)

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=42)

X = X / 255.0

print(X.shape)
print(Y.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, Y, epochs=15, batch_size=32)

image_folder = "/content/Data/Test"
output_folder = "/content/Test_resized"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(output_folder, filename), img_resized)

print("All images have been reshaped and saved.")

import re

image_folder = "/content/Test_resized"
image_array_list = []

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

sorted_filenames = sorted(
    [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")],
    key=extract_number
)

for filename in sorted_filenames:
    img_path = os.path.join(image_folder, filename)
    img = Image.open(img_path)
    img_array = np.array(img)
    image_array_list.append(img_array)

test_array = np.stack(image_array_list)
print(f"Final array shape: {test_array.shape}")

test_array = test_array / 255.0
prediction = model.predict(test_array)

import csv

image_ids = [f'image_{i+1}' for i in range(200) if i+1 != 62]
labels = ['Real' if pred > 0.5 else 'AI' for pred in prediction]

if len(image_ids) != len(labels):
    print("Error: Mismatch between number of image IDs and predictions.")
else:
    with open('submission.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Label'])
        for img_id, label in zip(image_ids, labels):
            writer.writerow([img_id, label])
    print("Submission file created: submission.csv")
