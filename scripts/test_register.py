import urllib.request, json

url = 'http://127.0.0.1:8001/api/v1/auth/register'
data = json.dumps({'email':'testuser_local@example.com','password':'pass12345'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        print('status', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('error', e)
    import traceback
    traceback.print_exc()