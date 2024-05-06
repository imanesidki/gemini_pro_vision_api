import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
import json
import os
import uuid
import time

def extract_json_from_image(image_path) :
    try:
        img = PIL.Image.open(image_path)
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
    "temperature": 0,  # no imagination
    "max_output_tokens": 2500,
    }

    model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config)

    response = model.generate_content([instructions, img], stream=False)
    response.resolve()
    if response.parts:
        result = response.text
    else:
        # Handle the case where no valid part is returned
        return(json.dumps({"status": False, "message":" No valid content was generated"}, indent=4))
    print(response)
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
        return(json.dumps({"status": False, "message": "Failed to decode JSON from response. Response was:" + result}, indent=4))
    except Exception as e:
        return(json.dumps({"status": False, "message": f"An error occurred: {e}"}, indent=4))

    return pretty_result


def prepare_finetuning_dataset(subset_name):
    output_folder = "finetuning_dataset"
    # Define image subfolder within output folder
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')
  
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    # Initialize list to hold all JSON data
    all_json_data = []
    counter = 0
    where_to_stop = 59
    for filename in os.listdir(image_subfolder):
        if counter == where_to_stop:
            where_to_stop += 60
            time.sleep(60)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Rename image
            image_path = os.path.join(image_subfolder, filename)
            unique_id = str(uuid.uuid4())  # Generate unique ID for each image
            new_image_name = f"{unique_id}.jpg"  # Rename image with unique ID
            os.rename(image_path, os.path.join(image_subfolder, new_image_name))
            image_path = os.path.join(image_subfolder, new_image_name)
            counter += 1
            gemini_json_response = extract_json_from_image(image_path)
            prompt_instruction = "Extract in json format information from the invoice image."
            # Structure for LLaVA JSON
            json_data = {
                "id": unique_id,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n"+prompt_instruction
                    },
                    {
                        "from": "gpt",
                        "value": gemini_json_response
                    }
                ]
            }
            print(json_data)
            print("====================================================")
            all_json_data.append(json_data)

    # sort `all_json_data` by `id` key
    all_json_data = sorted(all_json_data, key=lambda x: x["id"])

 
    # Save all_json_data to a JSON file to generate dataset for finetuning
    json_output_path = os.path.join(subset_folder, "dataset.json")
    with open(json_output_path, "w") as json_file:
        json.dump(all_json_data, json_file, indent=4)

if __name__ == "__main__":
    load_dotenv()
    # Configure gemini api key
    genai.configure(api_key = os.getenv("api_token"))
    #test on one invoice
    # image_path =  os.path.join(os.getcwd() ,"finetuning_dataset", "images", "dbee80ac-92bb-40c9-93ab-92d65187899e.jpg")
    # print(extract_json_from_image(image_path))
    # Extracting json data from all invoices
    subset_name = "train"
    prepare_finetuning_dataset(subset_name)
