import google.generativeai as genai
import PIL.Image
import json
import os

genai.configure(api_key="AIzaSyC2kp2iCcA1JfLQJZRwpNQjTDCx0MEP-1w")

try:
    img = PIL.Image.open(os.path.join(os.getcwd(), r"factures_images\Facture Janv 2024(3)_page-0011.jpg"))
    # img = img.resize((700, int(img.height * 700 / img.width))) it's a bad idea to resize the image as it may affect the model's performance to identify the content of the image
    if img.mode != 'RGB':
        img = img.convert('RGB')
except FileNotFoundError:
    print(json.dumps({"status": False, "message": "Failed to open image"}, indent=4))
    exit()

try:
    with open("instructions.txt", "r") as file:
        instructions = file.read()
except FileNotFoundError:
    print(json.dumps({"status": False, "message": "Failed to read instructions file"}, indent=4))
    exit()

# Set up the model
generation_config = {
  "temperature": 0,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config)

response = model.generate_content([instructions, img], stream=False)
response.resolve()

result = response.text

# trim the result to remove the leading and trailing whitespaces
result = result.strip()

# check if the result contains ``` or ```json then replace them with empty string
if "```json" in result:
    result = result.replace("```json", "")
if "```" in result:
    result = result.replace("```", "")

try:
    result = json.loads(result)
    pretty_result = json.dumps(result, indent=4)
    print(pretty_result)
except json.JSONDecodeError:
    print("Failed to decode JSON from response. Response was:")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")