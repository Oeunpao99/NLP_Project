import requests

def main():
    url = 'http://127.0.0.1:8001/api/v1/predict'
    params = {'text': 'សាកល្បង Huawei 5G', 'format': 'html'}
    try:
        r = requests.get(url, params=params, timeout=10)
        print('STATUS', r.status_code)
        print(r.text)
    except Exception as e:
        print('ERROR', e)

if __name__ == '__main__':
    main()
