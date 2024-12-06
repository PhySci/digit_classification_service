from sklearn.datasets import load_digits
from PIL import Image
import numpy as np
import zipfile
import os


def main():
    digits = load_digits()
    os.makedirs('images', exist_ok=True)
    for i in range(5):
        image_data = digits.images[i]
        image = Image.fromarray(np.uint8(image_data * 16))  # Преобразование в 8-битное изображение
        image.save(f'images/digit_{i}.png')

    with zipfile.ZipFile('digits_images.zip', 'w') as zipf:
        for i in range(10):
            image_data = digits.images[i]
            image = Image.fromarray(np.uint8(image_data * 16))
            image_path = f'images/digit_{i}.png'
            image.save(image_path)
            zipf.write(image_path, os.path.basename(image_path))


if __name__ == "__main__":
    main()