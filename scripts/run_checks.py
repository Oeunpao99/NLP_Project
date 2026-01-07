import json
import uuid
from urllib import request, error

BASE = 'http://127.0.0.1:8001'

headers = {'Content-Type': 'application/json'}

email = f"ci_test_{uuid.uuid4().hex[:8]}@example.com"
password = 'testpass123'

print('Using email:', email)

# Register
try:
    data = json.dumps({'email': email, 'password': password}).encode('utf-8')
    req = request.Request(f'{BASE}/api/v1/auth/register', data=data, headers=headers, method='POST')
    with request.urlopen(req, timeout=10) as resp:
        print('Register status:', resp.status)
        print(resp.read().decode())
except error.HTTPError as e:
    print('Register failed:', e.code, e.read().decode())
    raise
except Exception as e:
    print('Register error:', e)
    raise

# Login
try:
    data = json.dumps({'email': email, 'password': password}).encode('utf-8')
    req = request.Request(f'{BASE}/api/v1/auth/login', data=data, headers=headers, method='POST')
    with request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode()
        print('Login status:', resp.status)
        print('Login response:', text)
        token = json.loads(text)['access_token']
except Exception as e:
    print('Login error:', e)
    raise

# Predict with token (GET endpoint)
try:
    auth_headers = {'Authorization': f'Bearer {token}'}
    url = f"{BASE}/api/v1/predict?text={request.quote('សាកល្បង')}&format=json"
    req = request.Request(url, headers=auth_headers, method='GET')
    with request.urlopen(req, timeout=10) as resp:
        print('Predict (auth) status:', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('Predict (auth) error:', e)
    raise

# Fetch history authenticated
try:
    auth_headers = {'Authorization': f'Bearer {token}'}
    url = f"{BASE}/api/v1/predictions?page=1&page_size=5"
    req = request.Request(url, headers=auth_headers, method='GET')
    with request.urlopen(req, timeout=10) as resp:
        print('History (auth) status:', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('History (auth) error:', e)
    raise

# Fetch history unauthenticated
try:
    url = f"{BASE}/api/v1/predictions?page=1&page_size=5"
    req = request.Request(url, method='GET')
    with request.urlopen(req, timeout=10) as resp:
        print('History (unauth) status:', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('History (unauth) error:', e)
    raise

print('Checks completed')