from keras.preprocessing.image import ImageDataGenerator
import pathlib
import cv2
import numpy as np


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

data_dir= 'dog/'
data_dir = pathlib.Path(data_dir)
dog = list(data_dir.glob('*'))
dog_images_dict = {
    'dog': list(data_dir.glob('*.jpg'))
}

X, y = [], []

for dog, images in dog_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(300,300))
        X.append(resized_img)
        y.append(dog_images_dict[dog])

X = np.array(X)
y = np.array(y)

i = 0
for batch in datagen.flow(X, batch_size=1,
                          save_to_dir='dog/augmented_img', save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 10:
        break

