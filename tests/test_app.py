import unittest
from app import app

class FlaskTestCase(unittest.TestCase):
    # Ensure that the Flask app was set up correctly
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease Prediction', response.data)
    
    # Ensure that the /predict route behaves correctly
    def test_predict_no_input(self):
        tester = app.test_client(self)
        response = tester.post('/predict', data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    # Ensure that the prediction works correctly with valid input
    def test_predict_valid_input(self):
        tester = app.test_client(self)
        data = {
            'age': '63',
            'sex': '1',
            'cp': '3',
            'trestbps': '145',
            'chol': '233',
            'fbs': '1',
            'restecg': '0',
            'thalach': '150',
            'exang': '0',
            'oldpeak': '2.3',
            'slope': '0',
            'ca': '0',
            'thal': '1'
        }
        response = tester.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Result:', response.data)

    # Ensure that the prediction returns an appropriate message
    def test_predict_result_message(self):
        tester = app.test_client(self)
        data = {
            'age': '63',
            'sex': '1',
            'cp': '3',
            'trestbps': '145',
            'chol': '233',
            'fbs': '1',
            'restecg': '0',
            'thalach': '150',
            'exang': '0',
            'oldpeak': '2.3',
            'slope': '0',
            'ca': '0',
            'thal': '1'
        }
        response = tester.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b'Heart Disease' in response.data or b'No Heart Disease' in response.data)

if __name__ == '__main__':
    unittest.main()

