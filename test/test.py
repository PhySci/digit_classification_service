import unittest
import requests
import os

"""
Тесты для сервиса
- Предсказание для одного изображения
- Предсказание для нескольких изображений
- Предсказание для zip-архива
- Предсказание для нескольких изображений в zip-архиве
- Предсказание для изображения, которое не является цифрой
- Предсказание для zip-архива, который не содержит изображений
- Предсказание для файла, который не является изображением
- Предсказание для пустого zip-архива
- Проверка размера изображения
- Отправка цветного изображения
"""

class TestApp(unittest.TestCase):
    BASE_URL = "http://localhost:8000"  # Убедитесь, что ваш сервер запущен на этом адресе

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

    @unittest.skip
    def test_predict_batch_images(self):
        image_files = ["test_image1.png", "test_image2.png"]  # Список изображений
        with open("test_image1.png", "rb") as image_file1, open("test_image2.png", "rb") as image_file2:
            response = requests.post(f"{self.BASE_URL}/predict_batch", files={"files": [image_file1, image_file2]})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        for prediction in data:
            self.assertIn("digit", prediction)
            self.assertIn("confidence", prediction)

    @unittest.skip
    def test_predict_batch_zip(self):
        # Открываем тестовый zip-архив
        with open("test_images.zip", "rb") as zip_file:
            response = requests.post(f"{self.BASE_URL}/predict_batch", files={"files": zip_file})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        for prediction in data:
            self.assertIn("digit", prediction)
            self.assertIn("confidence", prediction)


if __name__ == '__main__':
    unittest.main()
