import requests

url = "https://data-neuron-live.onrender.com/calculate-similarity"
data = {
    "text1": "I love machine learning",
    "text2": "Machine learning is fascinating"
}

response = requests.post(url, json=data)
print(response.json())
# Output: {"similarity score": 0.75}