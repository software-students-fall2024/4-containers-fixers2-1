"""
Tests for the machine learning client.
"""

import pytest
import numpy as np
import cv2
from flask import Flask
from werkzeug.datastructures import FileStorage
from unittest.mock import patch, MagicMock
from io import BytesIO
from datetime import datetime
from machine_learning_client.ml_client import app, emotion_dict
from unittest.mock import patch, MagicMock, ANY
from pymongo.errors import PyMongoError


#fix
# Mock the MongoDB collection
@pytest.fixture
def client():
    """
    Fixture to provide a Flask test client for testing routes.
    """
    print(app.url_map)
    app.config["TESTING"] = True
    app.secret_key = "test_secret_key"
    with app.test_client() as client:
        yield client

# Mock for MongoDB collection
@pytest.fixture
def mock_db():
    """
    Mock for MongoDB collection.
    """
    with patch(
        "machine_learning_client.ml_client.emotion_data_collection",
        autospec=True
    ) as mock_collection:
        yield mock_collection


# Mock for TensorFlow model
@pytest.fixture
def mock_model():
    """
    Mock for TensorFlow model.
    """
    with patch("machine_learning_client.ml_client.model") as model_mock:
        model_mock.predict.return_value = np.array(
            [[0.8, 0.1, 0.05, 0.03, 0.02]]
        )  # Predicts "Happy 😊"
        yield model_mock


def test_detect_emotion(client, mock_model, mock_db):
    """
    Test the /detect_emotion route with valid input.
    """
    # Mock face detection to return a face
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])  # Mock face detection
        
        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        response_data = response.get_json()
        assert "emotion" in response_data
        assert response_data["emotion"] in emotion_dict.values()


def test_invalid_image_input(client):
    """
    Test the /detect_emotion route with invalid input.
    """
    # Send POST request without an image
    response = client.post(
        "/detect_emotion", data={}, content_type="multipart/form-data"
    )
    assert response.status_code == 400
    response_data = response.get_json()
    assert "error" in response_data
    assert response_data["error"] == "No image file provided"


def test_model_error(client, mock_model):
    """
    Test the /detect_emotion route when the model fails.
    """
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])
        mock_model.predict.side_effect = Exception("Model prediction failed")

        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        response_data = response.get_json()
        assert "error" in response_data
        assert "Model prediction failed" in response_data["error"]

def test_detect_emotion_invalid_method(client):
    """
    Test the /detect_emotion route with an invalid method (GET instead of POST).
    """
    response = client.get("/detect_emotion")
    assert response.status_code == 405  # Method Not Allowed


"""
Tests for the machine learning client.
"""

import pytest
import numpy as np
import cv2
from flask import Flask
from werkzeug.datastructures import FileStorage
from unittest.mock import patch, MagicMock, ANY
from io import BytesIO
from datetime import datetime
from machine_learning_client.ml_client import app, emotion_dict
from pymongo.errors import PyMongoError

@pytest.fixture
def client():
    """
    Fixture to provide a Flask test client for testing routes.
    """
    app.config["TESTING"] = True
    app.secret_key = "test_secret_key"
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_db():
    """
    Mock for MongoDB collection.
    """
    with patch(
        "machine_learning_client.ml_client.emotion_data_collection",
        autospec=True
    ) as mock_collection:
        yield mock_collection

@pytest.fixture
def mock_model():
    """
    Mock for TensorFlow model.
    """
    with patch("machine_learning_client.ml_client.model") as model_mock:
        # Ensure predict returns proper shape for emotion classification
        model_mock.predict.return_value = np.array(
            [[0.8, 0.1, 0.05, 0.03, 0.02]]
        )
        yield model_mock

# Fix 1: Update test_detect_emotion to handle the face detection
def test_detect_emotion(client, mock_model, mock_db):
    """
    Test the /detect_emotion route with valid input.
    """
    # Mock face detection to return a face
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])  # Mock face detection
        
        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        response_data = response.get_json()
        assert "emotion" in response_data
        assert response_data["emotion"] in emotion_dict.values()

# Fix 2: Update model error test to mock face detection
def test_model_error(client, mock_model):
    """
    Test the /detect_emotion route when the model fails.
    """
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])
        mock_model.predict.side_effect = Exception("Model prediction failed")

        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        response_data = response.get_json()
        assert "error" in response_data
        assert "Model prediction failed" in response_data["error"]

# Fix 3: Update database error test
def test_database_insertion_error(client, mock_model, mock_db):
    """
    Test the /detect_emotion route when database insertion fails.
    """
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])
        mock_db.insert_one.side_effect = PyMongoError("Database insertion failed")

        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        response_data = response.get_json()
        assert "error" in response_data
        assert "Database insertion failed" in response_data["error"]


def test_missing_content_type(client):
    """
    Test the /detect_emotion route with missing Content-Type header.
    """
    response = client.post("/detect_emotion", data={})
    assert response.status_code == 400
    response_data = response.get_json()
    assert "error" in response_data
    assert response_data["error"] == "No image file provided"


def test_invalid_route(client):
    """
    Test accessing an invalid route.
    """
    response = client.get("/non_existent_route")
    assert response.status_code == 404  # Not Found


def test_invalid_method_on_detect_emotion(client):
    """
    Test the /detect_emotion route with an invalid HTTP method (GET).
    """
    response = client.get("/detect_emotion")
    assert response.status_code == 405  # Method Not Allowed


def test_mongodb_insertion_error(client, mock_model, mock_db):
    """Test /detect_emotion route when MongoDB insertion fails."""
    mock_model.predict.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3, 0.0, 0.0]]  # Mock prediction
    )
    mock_db.insert_one.side_effect = Exception("Mocked database error")

    dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode(".jpg", dummy_image)
    dummy_image_data = buffer.tobytes()

    file_storage = FileStorage(
        stream=BytesIO(dummy_image_data),
        filename="test_image.jpg",
        content_type="image/jpeg"
    )

    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])  # Mock a detected face
        
        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data"
        )

    assert response.status_code == 500
    response_data = response.get_json()
    assert "error" in response_data
    assert "Database insertion failed" in response_data["error"]



def test_alternative_emotion_prediction(client, mock_model, mock_db):
    """
    Test the /detect_emotion route when the model predicts "Sad 😢".
    """
    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])
        mock_model.predict.return_value = np.array(
            [[0.1, 0.3, 0.05, 0.03, 0.8, 0.2, 0.01]]
        )

        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        response_data = response.get_json()
        assert response_data["emotion"] == "Sad 😢"
        mock_db.insert_one.assert_called_once()


def test_model_not_loaded(client, mock_db):
    """
    Test the /detect_emotion route when the TensorFlow model is not loaded.
    """
    with patch("machine_learning_client.ml_client.model", None):  # Simulate missing model
        dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".jpg", dummy_image)
        dummy_image_data = buffer.tobytes()

        file_storage = FileStorage(
            stream=BytesIO(dummy_image_data),
            filename="test_image.jpg",
            content_type="image/jpeg",
        )

        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        response_data = response.get_json()
        assert "error" in response_data
        assert "Emotion detection model not loaded" in response_data["error"]  # Match the exact message



# def test_unhandled_exception_in_prediction(client, mock_model):
#     """
#     Test the /detect_emotion route when an unhandled exception occurs during model prediction.
#     """
#     mock_model.predict.side_effect = Exception("Unexpected error during prediction")

#     dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
#     _, buffer = cv2.imencode(".jpg", dummy_image)
#     dummy_image_data = buffer.tobytes()

#     file_storage = FileStorage(
#         stream=BytesIO(dummy_image_data),
#         filename="test_image.jpg",
#         content_type="image/jpeg",
#     )

#     response = client.post(
#         "/detect_emotion",
#         data={"image": file_storage},
#         content_type="multipart/form-data",
#     )

#     assert response.status_code == 500
#     response_data = response.get_json()
#     assert "error" in response_data
#     assert "Unexpected error during prediction" in response_data["error"]


def test_valid_prediction_with_mongodb_error(client, mock_model, mock_db):
    """
    Test the /detect_emotion route with valid prediction but MongoDB insertion fails.
    """
    mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])  # Mock prediction
    mock_db.insert_one.side_effect = Exception("Database insertion error")

    dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode(".jpg", dummy_image)
    dummy_image_data = buffer.tobytes()

    file_storage = FileStorage(
        stream=BytesIO(dummy_image_data),
        filename="test_image.jpg",
        content_type="image/jpeg",
    )

    with patch("cv2.CascadeClassifier.detectMultiScale") as mock_detect:
        mock_detect.return_value = np.array([[0, 0, 100, 100]])  # Mock face detection
        
        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    response_data = response.get_json()
    assert "error" in response_data
    assert "Database insertion failed" in response_data["error"]

# Test missing image file
def test_detect_emotion_no_image(client):
    """Test detect_emotion endpoint when no image is provided"""
    response = client.post("/detect_emotion")
    assert response.status_code == 400
    assert "No image file provided" in response.get_json()["error"]

# Test invalid image format
def test_detect_emotion_invalid_image(client):
    """Test detect_emotion endpoint with invalid image data"""
    # Create a very small invalid image data that will fail CV2 processing
    invalid_data = b"invalid"
    file_storage = FileStorage(
        stream=BytesIO(invalid_data),
        filename="invalid.jpg",
        content_type="image/jpeg"
    )
    
    response = client.post(
        "/detect_emotion",
        data={"image": file_storage},
        content_type="multipart/form-data"
    )
    
    assert response.status_code == 500
    error_message = response.get_json().get("error", "")
    assert isinstance(error_message, str)
    assert "Error" in error_message

# Test face detection failure
def test_detect_emotion_no_face(client, mock_model):
    """Test detect_emotion when no face is detected in the image."""
    dummy_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode(".jpg", dummy_image)
    dummy_image_data = buffer.tobytes()

    file_storage = FileStorage(
        stream=BytesIO(dummy_image_data),
        filename="test_image.jpg",
        content_type="image/jpeg"
    )

    with patch('cv2.CascadeClassifier.detectMultiScale') as mock_detect:
        mock_detect.return_value = np.array([])  # No faces detected
        
        response = client.post(
            "/detect_emotion",
            data={"image": file_storage},
            content_type="multipart/form-data"
        )

    assert response.status_code == 400
    response_data = response.get_json()
    assert "error" in response_data
    assert response_data["error"] == "No faces detected in the image."


# Test frame reading error
def test_run_emotion_detection_frame_error():
    """Test run_emotion_detection when frame reading fails"""
    with patch('cv2.VideoCapture') as mock_capture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Frame read failure
        mock_capture.return_value = mock_cap
        
        from machine_learning_client.ml_client import run_emotion_detection
        run_emotion_detection()  # Should exit gracefully
        
        mock_cap.release.assert_called_once()

# Test model loading
def test_model_loading():
    """Test that the model loads correctly"""
    with patch('tensorflow.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()
        
        # Re-import to trigger model loading
        import importlib
        import machine_learning_client.ml_client
        importlib.reload(machine_learning_client.ml_client)
        
        mock_load_model.assert_called_once()

# Test MongoDB connection
def test_mongodb_connection():
    """Test MongoDB connection initialization"""
    with patch('pymongo.MongoClient') as mock_client:
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        # Re-import to trigger MongoDB connection
        import importlib
        import machine_learning_client.ml_client
        importlib.reload(machine_learning_client.ml_client)
        
        mock_client.assert_called_once_with("mongodb://mongo:27017/")