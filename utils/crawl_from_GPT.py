import openai
import time
import json

header = {
    'Content-Type': 'application/json',
}


def get_response(prompt, model_name):
    message = [{'role': 'user', 'content': prompt}]
    resp = '*MAX-TRY*'
    while '*MAX-TRY*' in resp:
        resp = get_response_venus(message, model_name, wait=10, timeout=60)
    return resp


def get_response_with_message(message, model_name):
    resp = '*MAX-TRY*'
    while '*MAX-TRY*' in resp:
        resp = get_response_venus(message, model_name, wait=10, timeout=60)
    return resp


def get_response_single(xxxxx):
   ## TODO: impliment your only get response function, input message, return the response string.