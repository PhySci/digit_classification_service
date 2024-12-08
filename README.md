# ML Service for Digit Prediction

This project is a web service that uses a machine learning model to predict digits from images. The service is built using FastAPI and provides two main API endpoints for prediction: for a single image and for multiple images (including zip archives).

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation and Running](#installation-and-running)
3. [API Usage](#api-usage)
4. [Testing](#testing)
5. [Model](#model)
6. [Notes](#notes)
7. [Contact](#contact)

## Project Structure

- `src/app.py`: Main application file containing the API endpoints.
- `src/ml.py`: Module containing functions for feature extraction and digit prediction.
- `models/svm.pkl`: Saved SVM model for digit prediction.
- `test/test.py`: Test suite for verifying API functionality.
- `docker/Dockerfile`: Dockerfile for building the application container.
- `src/requirements.txt`: List of Python dependencies.
- `Makefile`: Scripts for building and running the Docker container.

## Installation and Running

### Local Setup

1. Ensure you have Python 3.10 installed.
2. Install the dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```
3. Run the application:
   ```bash
   python src/app.py
   ```

Web-service will be available at http://127.0.0.1:8000/

### Docker Setup

1. Build the Docker image:
   ```bash
   make build
   ```
2. Run the container:
   ```bash
   make run
   ```

## API Usage

### Single Image Prediction

- **Method**: POST
- **URL**: `/predict`
- **Parameters**: Image file in PNG format
- **Response**: JSON with predicted digit and confidence

### Batch Image Prediction

- **Method**: POST
- **URL**: `/predict_batch`
- **Parameters**: List of image files or a zip archive
- **Response**: JSON with predicted digits and confidence for each image

### Health Check

- **Method**: GET
- **URL**: `/health`
- **Response**: JSON with service status

## Testing

To run the tests, use the command:

```bash
python -m unittest discover -s test
```
## Model

The machine learning model is trained using SVM and saved in the `models/svm.pkl` file. It uses HOG features for digit prediction.

## Notes

- Ensure the model file `svm.pkl` is located in the `models` directory.
- All dependencies listed in `requirements.txt` must be installed for the service to function correctly.

## Ways to improve the service

- Fine-tune hyperparameters of HOG feature extractor and the SVM classifier;
- Or use simple CNN model for digit classification;
- Put trained model on S3 storage and download it on startup;
- Add authentication and authorization;
- Add CI/CD pipeline;
- Connect Sentry for monitoring and logging;
- Implement asynchronous processing of requests (i.e. queue);
- Add swagger documentation.


