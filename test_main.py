from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestMain:
    """Tests for root endpoint"""

    def test_read_main(self):
        """Test root endpoint returns correct message"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}


class TestPredict:
    """Tests for predict endpoint"""

    def test_predict_positive(self):
        """Test positive sentiment prediction"""
        response = client.post(
            "/predict/",
            json={"text": "I like machine learning!"}
        )
        json_data = response.json()
        assert response.status_code == 200
        assert json_data['label'] == 'POSITIVE'
        assert 'score' in json_data

    def test_predict_negative(self):
        """Test negative sentiment prediction"""
        response = client.post(
            "/predict/",
            json={"text": "I hate machine learning!"}
        )
        json_data = response.json()
        assert response.status_code == 200
        assert json_data['label'] == 'NEGATIVE'
        assert 'score' in json_data

    def test_predict_empty_text(self):
        """Test empty string input"""
        response = client.post(
            "/predict/",
            json={"text": ""}
        )
        assert response.status_code == 422  # validation error
