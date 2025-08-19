import openai
import time
import json

# Let user set their own API key here
openai.api_key = "YOUR_API_KEY_HERE"

header = {
    'Content-Type': 'application/json',
}


def get_response(prompt, model_name):
    message = [{'role': 'user', 'content': prompt}]
    resp = '*MAX-TRY*'
    while '*MAX-TRY*' in resp:
        resp = get_response_single(message, model_name, wait=10, timeout=60)
    return resp


def get_response_with_message(message, model_name):
    resp = '*MAX-TRY*'
    while '*MAX-TRY*' in resp:
        resp = get_response_single(message, model_name, wait=10, timeout=60)
    return resp


def get_response_single(message, model_name, wait=5, timeout=60):
    """
    Retrieve a single response from OpenAI's API.
    :param message: list of dicts, e.g. [{'role':'user','content':'Hello'}]
    :param model_name: str, name of the model (e.g. "gpt-4o-mini")
    :param wait: int, seconds to wait before retry on error
    :param timeout: int, max seconds before aborting
    :return: str, response content or "*MAX-TRY*" if failed
    """
    start_time = time.time()
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=message,
                timeout=timeout
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            if time.time() - start_time > timeout:
                return "*MAX-TRY*"
            print(f"Error: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
