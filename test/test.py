import io
import unittest
import requests
import os
from PIL import Image
import numpy as np
from sklearn.datasets import load_digits
import zipfile
import shutil


"""
Тесты для сервиса
+ Предсказание для одного изображения
+ Предсказание для нескольких изображений
- Предсказание для изображения, которое не является цифрой
- Предсказание для zip-архива, который не содержит изображений
- Предсказание для файла, который не является изображением
- Предсказание для пустого zip-архива
- Проверка размера изображения
- Отправка цветного изображения
- Измерение качества
"""


class TestApp(unittest.TestCase):
    BASE_URL = "http://localhost:8050"

    @classmethod
    def setUpClass(cls) -> None:
        digits = load_digits()
        os.makedirs('./images', exist_ok=True)
        for i in range(5):
            image_data = digits.images[i]
            image = Image.fromarray(image_data, mode="L")
            image.save(f'./images/digit_{i}.png')

        with zipfile.ZipFile('./images/digits_images.zip', 'w') as zipf:
            for i in range(10):
                image_data = digits.images[i]
                image = Image.fromarray(np.uint8(image_data * 16))
                image_path = f'./images/digit_{i}.png'
                image.save(image_path)
                zipf.write(image_path, os.path.basename(image_path))

    # @classmethod
    # def tearDown(self) -> None:
    #     shutil.rmtree("./images")

    def test_predict_single_image(self):
        image_folder = os.path.join(os.path.dirname(__file__), "images")
        image_list = [el.path for el in os.scandir(image_folder) if el.is_file() and el.name.endswith(".png")] 

        for image_file_name in image_list:
            with open(image_file_name, "rb") as image_file:
                response = requests.post(f"{self.BASE_URL}/predict", files={"file": image_file})
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("digit", data)
            self.assertIn("confidence", data)

    def test_predict_batch_images(self):
        image_folder = os.path.join(os.path.dirname(__file__), "images")
        image_list = [el.path for el in os.scandir(image_folder) if el.is_file() and el.name.endswith(".png")]

        image_files = [("files", (os.path.basename(image_path), open(image_path, "rb"), "image/png")) for image_path in image_list]
        try:
            response = requests.post(f"{self.BASE_URL}/predict_batch", files=image_files)
        finally:
            for _, file_tuple in image_files:
                file_tuple[1].close()
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(image_list))
        for prediction in data:
            self.assertIn("digit", prediction)
            self.assertIn("confidence", prediction)

    def test_predict_batch_zip(self):
        zip_pth = os.path.join(os.path.dirname(__file__), "images", "digits_images.zip")
        with open(zip_pth, "rb") as zip_file:
            response = requests.post(f"{self.BASE_URL}/predict_batch", files={"files": zip_file})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        for prediction in data:
            self.assertIn("digit", prediction)
            self.assertIn("confidence", prediction)

    def test_metrics(self):
        BATCH_SIZE = 500
        digits = load_digits()
        acc = 0.0

        for i, (image, true_label) in enumerate(zip(digits.images, digits.target)):
            image = Image.fromarray((image*8).astype(np.uint8), mode="L")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            response = requests.post(f"{self.BASE_URL}/predict", files={"file": image_bytes})
            data = response.json()
            acc += data["digit"] == true_label

            if data["digit"] != true_label:
                pass
            if i > BATCH_SIZE:
                break

        acc /= BATCH_SIZE
        assert acc > 0.9, "Accuraty is {:4.3f} - too low".format(acc)


class TestModel(unittest.TestCase):

    def test_acc(self):
        from src.ml import predict_digit

        BATCH_SIZE = 500
        digits = load_digits()
        acc = 0.0

        for i, (image1, true_label) in enumerate(zip(digits.images, digits.target)):
            image2 = Image.fromarray((image1*8).astype(np.uint8), mode="L")
            predict_label, _ = predict_digit(image2)
            acc += predict_label == true_label
            if i > BATCH_SIZE:
                break
            if predict_label != true_label:
                pass

        acc /= BATCH_SIZE
        assert acc > 0.9, "Accuraty is {:4.3f} - too low".format(acc)

        

if __name__ == '__main__':
    unittest.main()
