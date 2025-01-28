#!/usr/bin/env python3
import sys
import requests


def main():
    """
    Main function to classify headlines by sending them to a local prediction server.

    This function can classify headlines provided via command-line arguments
    or from a text file.

    Usage:
        python classify_headline.py 'headline1' 'headline2' ...
        python classify_headline.py --file headlines.txt

    Command-line Arguments:
        headlines (list of str): The headline texts to be classified, or
        --file followed by the path to a text file containing headlines.

    Raises:
        SystemExit: If no input is provided.
    """
    if len(sys.argv) < 2:
        print("Usage: python classify_headline.py 'headline1' 'headline2' ...")
        print("       python classify_headline.py --file headlines.txt")
        sys.exit(1)

    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Error: Please specify the file path after --file.")
            sys.exit(1)

        file_path = sys.argv[2]
        try:
            with open(file_path, "r") as f:
                # Split file content into separate lines
                headlines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            sys.exit(1)
    else:
        headlines = sys.argv[1:]

    url = "http://localhost:80/predict"

    for headline in headlines:
        payload = {"headline": headline}
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print(f"Headline: {headline}")
            print("Predicted category:", response.json()["category"])
        else:
            print(f"Headline: {headline}")
            print("Error:", response.text)
        print()  # Add a newline for better readability


if __name__ == "__main__":
    main()
