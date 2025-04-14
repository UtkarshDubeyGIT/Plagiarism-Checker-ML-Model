import requests

# URL of the Flask server
BASE_URL = "http://127.0.0.1:5000"

# Test 1: AI Detection
ai_text = "Terrorism is the unlawful use of violence and intimidation"

response_ai = requests.post(
    f"{BASE_URL}/detect-ai",
    json={"text": ai_text}
)

print("=== AI Detection Result ===")
print(response_ai.json())

# Test 2: Plagiarism Detection
source = "Climate change refers to significant changes in global temperatures and weather patterns over time."
suspicious = "Significant changes in the earth's weather and temperature over time are called climate change."

response_plag = requests.post(
    f"{BASE_URL}/check-plagiarism",
    json={
        "source": source,
        "suspicious": suspicious
    }
)

print("\n=== Plagiarism Detection Result ===")
print(response_plag.json())
