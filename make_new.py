import os
from tqdm import tqdm
import json
import argparse
from multiprocessing import Pool
from utils.common import check_dir, print_args, set_seed
from utils.data import read_file, load_data
from utils.crawl_from_GPT import get_response_venus as get_response
import torch

prefix_path_make = 'prefix/change_inst.txt'
prefix_make = read_file(prefix_path_make)


def judge_sample(args):
    sample, model_name, tmp_file = args
    template = prefix_make
    prompt = template.replace('{{  Prompt  }}', sample['instruction'])
    prompt = prompt.replace('{{  Extracted  }}', sample['det'])
    message = [{'role': 'user', 'content': prompt}]
    resp = '\"error\"'
    while '\"error\"' in resp or '*MAX-TRY*' in resp:
        resp = get_response(message, model_name, wait=10, timeout=60)
    sample['new'] = resp
    with open(tmp_file, 'a') as file:
        file.write(json.dumps(sample) + '\n')
    return sample


def judge(data, arg, name):
    tmp_file = arg.tmp_file.replace('[dataset]', f"{name}")
    args_list = [(sample, arg.model_name, tmp_file) for index, sample in enumerate(data)]
    results = []
    with open(tmp_file, 'w') as file:
        pass
    with Pool(arg.num_workers) as pool:
        # Prepare the pool tasks
        tasks = [pool.apply_async(judge_sample, args=(_arg,)) for _arg in args_list]
        # Use tqdm to track progress
        for task in tqdm(tasks, total=len(tasks)):
            result = task.get()
            results.append(result)
    return results


def run(arg):
    # data version, used for naming
    dataset = arg.dataset
    # load data
    if '/' not in dataset:
        path = f'data/{dataset}.json'
    else:
        path = dataset
    data = load_data(path)
    result = judge(data, arg, arg.stage)
    json.dump(result, open(arg.save_path.replace('[dataset]', f"{arg.stage}"), 'w'), indent=2)


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(description='xxxxx')
    parser.add_argument('--dataset', type=str, default='alpaca_all_rand100')
    parser.add_argument('--stage', type=str, default='change')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='gpt-4-0125-preview')
    # parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--tmp_file', type=str, default='tmp')
    arguments = parser.parse_args()
    # start
    if arguments.save_path == 'tmp':
        arguments.save_path = f'result/{arguments.dataset}/predict/{arguments.model_name}_[dataset].json'
    if arguments.tmp_file == 'tmp':
        arguments.tmp_file = f'result/{arguments.dataset}/cache/{arguments.model_name}_[dataset].json'
    check_dir(arguments.save_path)
    check_dir(arguments.tmp_file)
    print_args(arguments)
    run(arguments)
