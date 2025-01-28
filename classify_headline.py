#!/usr/bin/env python3
import sys
import requests


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_headline.py 'your headline text'")
        sys.exit(1)

    headline = " ".join(sys.argv[1:])
    url = "http://localhost:80/predict"
    payload = {"headline": headline}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Predicted category:", response.json()["category"])
    else:
        print("Error:", response.text)


if __name__ == "__main__":
    main()
