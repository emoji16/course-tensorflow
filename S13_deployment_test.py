import json, requests

a = [1.0, 2.0, 3.0]
input_data = json.dumps({
    'numbers':a
})
client_url = 'http://localhost:5000/cls'
r = requests.post(url = client_url, data = input_data)
print(r.text)