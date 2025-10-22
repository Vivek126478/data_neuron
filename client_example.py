import json
import sys
import requests


def main():
    if len(sys.argv) < 2:
        url = "http://127.0.0.1:5000/calculate-similarity"
    else:
        url = sys.argv[1]

    payload = {
        "text1": "nuclear body seeks new tech to improve safety",
        "text2": "terror suspects face arrest by authorities"
    }

    response = requests.post(url, json=payload, timeout=30)
    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
