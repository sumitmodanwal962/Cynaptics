
# Step 1: Install Kaggle library
!pip install kaggle

# Step 2: Upload kaggle.json file
from google.colab import files
files.upload()

# Step 3: Move kaggle.json to the correct location and set permissions
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download the dataset
!kaggle datasets download -d yewtsing/pretty-face

# Step 5: Unzip the downloaded dataset
import zipfile
with zipfile.ZipFile('pretty-face.zip', 'r') as zip_ref:
    zip_ref.extractall('pretty_face')

# Step 6: Verify dataset content
import os
print(os.listdir('pretty_face'))

# Load and display an image
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('/content/pretty_face/face/face/000025.png')
cv2_imshow(img)
resized = cv.resize(img, (256, 256))
cv2_imshow(resized)

# Function to convert images to NumPy array using Pillow
from PIL import Image

def images_to_numpy_pillow(image_folder_path, target_size=(64, 64)):
    image_arrays = []
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(image_path).convert('RGB')
                img_resized = img.resize(target_size)
                img_array = np.array(img_resized)
                image_arrays.append(img_array)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
    return np.stack(image_arrays)

image_folder_path = '/content/pretty_face/face/face/'
images_array = images_to_numpy_pillow(image_folder_path, target_size=(64, 64))
print(f"Final array shape: {images_array.shape}")

# Build the Generator Model
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(8 * 8 * 128, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.Reshape((8, 8, 128)),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(128, (3, 3), padding='same'),
        layers.LeakyReLU(0.2),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(128, (3, 3), padding='same'),
        layers.LeakyReLU(0.2),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')
    ])
    return model

# Build the Discriminator Model
def build_discriminator(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), strides=2, input_shape=input_shape, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Conv2D(128, (3, 3), strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Conv2D(256, (3, 3), strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile the GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    return model

latent_dim = 100
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Normalize dataset
X_train = (images_array.astype(np.float32) - 127.5) / 127.5

# Train the GAN
def train_gan(epochs=500, batch_size=64, latent_dim=100, X_train=X_train):
    half_batch = batch_size // 2
    total_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(total_batches):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_images = X_train[idx]
            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_images = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss[0]} - G Loss: {g_loss}")
        if epoch % 5 == 0:
            save_and_print_generated_images(epoch)

# Save and print generated images
def save_and_print_generated_images(epoch, examples=10, figsize=(10, 2)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(1, examples, figsize=figsize)
    for i in range(examples):
        axs[i].imshow(generated_images[i])
        axs[i].axis('off')
    plt.show()
    fig.savefig(f"gan_generated_image_{epoch}.png")
    plt.close()

# Start training the GAN
train_gan(epochs=500, batch_size=64, latent_dim=latent_dim, X_train=X_train)
