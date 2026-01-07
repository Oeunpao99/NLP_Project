# test_auth.py
import requests
import json

BASE_URL = "http://localhost:8001"

def test_password_change():
    # First login
    login_data = {
        "email": "test@example.com",
        "password": "oldpassword"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
    print(f"Login: {response.status_code}")
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Change password
        change_data = {
            "old_password": "oldpassword",
            "new_password": "newpassword123"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/change-password",
            json=change_data,
            headers=headers
        )
        
        print(f"Change password: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test with new password
        login_data["password"] = "newpassword123"
        response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        print(f"Login with new password: {response.status_code}")

if __name__ == "__main__":
    test_password_change()