from sklearn import svm
from skimage import filters, feature
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float
import numpy as np
from PIL import Image
import pickle
import os

model_pth = os.path.join(os.path.dirname(__file__), "../models", 'svm.pkl')

try:
    model = pickle.load(open(model_pth, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    raise e


def get_features(image: Image.Image):
    img = img_as_float(image)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = filters.gaussian(img, sigma=0.5)
    img = rescale_intensity(img, in_range='image', out_range=(0, 1))
    hog_features = feature.hog(img, orientations=9, pixels_per_cell=(4, 4),
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return np.array([hog_features])


def predict_digit(image: Image.Image) -> tuple[int, float]:
    features = get_features(image)
    probs = model.predict_proba(features)[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction]
    return int(prediction), float(confidence)
