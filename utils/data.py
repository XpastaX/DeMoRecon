import json
import tiktoken
import requests
from tqdm import tqdm
import re


def load_data(path):
    try:
        return json.load(open(path, 'r'))
    except:
        with open(path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data


def save_data(data, path, jsonl=False):
    if jsonl:
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        json.dump(data, open(path, 'w'), indent=2)


def read_file(path):
    with open(path, 'r') as f:
        txt = f.read()
    return txt


def convert_back(data):
    new_data = []
    for index, sample in enumerate(data):
        if sample['output'] is None:
            print('gan')
        message = [{"role": "system", "content": ""},
                   {"role": "user", "content": sample['instruction']},
                   {"role": "assistant", "content": sample['output']},
                   ]
        new_sample = {
            'dataset': sample['dataset'],
            'id': sample['id'],
            'messages': message,
        }
        new_data.append(new_sample)
    return new_data


def estimate_bill(prompt=None, response=None, model=None, tqdm_disable=False):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sum_token_prompt = 0
    sum_token_response = 0
    if prompt is not None:
        print('Tokenizing prompts')
        for text in tqdm(prompt, disable=tqdm_disable):
            sum_token_prompt += len(tokenizer.encode(text))
    if response is not None:
        print('Tokenizing responses')
        for text in tqdm(response, disable=tqdm_disable):
            sum_token_response += len(tokenizer.encode(text))

    print(f"prompt_token:{sum_token_prompt}|response_token:{sum_token_response}")
    print(model)
    price_list = {
        'gpt-4-1106-preview': [0.01, 0.03],
        'gpt-4': [0.03, 0.06],
        'gpt-4-32k': [0.06, 0.12],
        'gpt-3.5-turbo-1106': [0.001, 0.002],
        'gpt-3.5-turbo-instruct': [0.0015, 0.002],

    }
    bill = {}
    for mod in price_list:
        bill[mod] = sum_token_prompt / 1000 * price_list[mod][0] + sum_token_response / 1000 * price_list[mod][1]
    print("{:^23}: {:^10} {:^10}".format('model_nli', 'USD', 'CYN'))
    if model is None:
        for mod in bill:
            print(
                "{:<23}: {:<10} {:<10}".format(mod, str(round(bill[mod], 2)), round(convert_usd_to_rmb(bill[mod]), 2)))
        return bill
    else:
        print("{:<23}: {:<10} {:<10}".format(model, str(round(bill[model], 2)),
                                             round(convert_usd_to_rmb(bill[model]), 2)))
        return bill[model]


def convert_usd_to_rmb(amount):
    # API endpoint for currency conversion
    api_url = "https://api.exchangerate-api.com/v4/latest/USD"

    try:
        # Sending a request to the API
        response = requests.get(api_url)
        data = response.json()

        # Getting the exchange rate for USD to RMB (CNY)
        exchange_rate = data['rates']['CNY']

        # Calculating the converted amount
        converted_amount = amount * exchange_rate

        return converted_amount
    except Exception as e:
        return str(e), None


def check_ans(resp, cot=False):
    if not cot:
        if 'Answer: Yes' in resp:
            return 1
        elif 'Answer: No' in resp:
            return 0
        flag_yes = False
        flag_no = False
        for yes in ['Yes', 'YES']:
            if yes in resp:
                flag_yes = True
                break
        for no in ['No', 'NO']:
            if no in resp:
                flag_no = True
                break
        if flag_yes and flag_no:
            pred = -1
        elif flag_yes:
            pred = 1
        elif flag_no:
            pred = 0
        else:
            pred = -1

    else:
        if 'answer is yes' in resp.lower():
            pred = 1
        elif 'answer is no' in resp.lower():
            pred = 0
        else:
            pred = -1
    return pred
