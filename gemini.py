import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
import json
import os

def extract_json_from_image(image_path) :
    try:
        img = PIL.Image.open(os.path.join(os.getcwd(), "factures_images", image_path)) 
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except FileNotFoundError as e:
        return(json.dumps({"status": False, "message": "Failed to open image " + str(e)}, indent=4))
    try:
        with open("instructions.txt", "r") as file:
            instructions = file.read()
    except FileNotFoundError:
        return(json.dumps({"status": False, "message": "Failed to read instructions file"}, indent=4))

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
    except json.JSONDecodeError:
        return("Failed to decode JSON from response. Response was:" + result)
    except Exception as e:
        return(f"An error occurred: {e}")

    return pretty_result


if __name__ == "__main__":
    load_dotenv()
    genai.configure(api_key = os.getenv("api_token"))
    image_path = r"Facture Janv 2024 (1)_page-0002.jpg"
    json_data = extract_json_from_image(image_path)
    print(json_data)