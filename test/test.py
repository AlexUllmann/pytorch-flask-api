import requests
resp = requests.post("http://localhost:5000/predict", files ={'file': open("other.png", "rb")} )
print(resp.json())
"""try:
    resp = requests.post("http://localhost:5000/predict", files={'file': open("other.png", "rb")})
    
    # 🌟 FIX: Print the entire JSON dictionary
    print("Server Response Status Code:", resp.status_code)
    print("Server Response Data:", resp.json())

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the Flask server. Make sure it is running at http://localhost:5000")
except Exception as e:
    print("An unexpected error occurred:", e)"""
