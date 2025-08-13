
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, Part
from urllib.parse import quote
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Initialize the client 
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# The prompt you want to send for each page:
generic_prompt = """You are an absolute expert on Tibetan. Please OCR and extract all the main text from this file, be as accurate as possible. Make use of your specific knowledge of Tibetan to ensure accuracy. Don't use markdown in the output. """

def ocr_image(image_bytes: bytes, model_name: str, mime_type: str = "image/jpeg", prompt: str = generic_prompt) -> str:
    """
    Process a single image using a specified Gemini model with the provided prompt.
    Dynamically adapts to model requirements (e.g., thinking mode).
    """
    
    # Base config that will be modified
    generation_config = GenerateContentConfig(
        max_output_tokens=4000
    )
    # Start by assuming we can use the more efficient setting
    generation_config.thinking_config = ThinkingConfig(thinking_budget=0)

    for attempt in range(5):
        try:
            generation_config.temperature = 1 + (attempt * 0.2)

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=generation_config
            )
            
            if response.text is not None:
                return response.text.strip()
            else:
                tqdm.write(f"[OCR_IMAGE] Attempt {attempt+1} with {model_name}, response.text is None")
                time.sleep(3)

        except Exception as e:
            # If the model requires thinking mode, adapt the config and retry
            if "This model only works in thinking mode" in str(e):
                tqdm.write(f"Model {model_name} requires thinking mode. Adapting config and retrying.")
                # Remove the thinking_config for subsequent retries
                generation_config = GenerateContentConfig(
                    max_output_tokens=4000
                )
                continue # Immediately try again with the new config

            tqdm.write(f"[OCR_IMAGE] Attempt {attempt+1} with {model_name}, error: {e}")
            time.sleep(3)
            
    return "ERROR"

async def do_ocr(file_bytes: bytes, filename: str, content_type: str, model_name: str, instruction: str = ""):
    """
    Main function that processes either PDF or image files (PNG/JPEG/WEBP).
    """
    start_time = time.time()
    
    filename = filename.lower()
    
    # Update the prompt with the additional instruction
    updated_prompt = generic_prompt + f" {instruction}"

    if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        # Handle image file
        mime_type = (
            'image/jpeg' if filename.endswith(('.jpg', '.jpeg'))
            else 'image/png' if filename.endswith('.png')
            else 'image/webp'
        )
        # Offload blocking OCR to a thread so the event loop can make progress
        text = await asyncio.to_thread(ocr_image, file_bytes, model_name, mime_type, updated_prompt)
    else:
        raise ValueError("unhandled file extension")

    # Return response based on text size
    duration = time.time() - start_time
    
    # The service layer's only job is to do the OCR and return the data.
    # The router/endpoint will handle creating the HTTP response.
    return text, duration