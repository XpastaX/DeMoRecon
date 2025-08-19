import random
from tqdm import tqdm
import json
import argparse
from multiprocessing import Pool
from utils.common import check_dir, print_args, set_seed
from utils.process import split_examples
from utils.data import read_file, load_data, save_data
from utils.crawl_from_GPT import get_response
import torch
import re
from glob import glob

prefix = read_file('prefix/test/direct_eval.txt') + 'The answer is '


# prefix = read_file('prefix/test/direct_eval.txt') + 'In conclusion, the answer is '

# prefix_a = read_file('prefix/test/direct_eval_analysis.txt')


def get_answer(response, label='Yes'):
    response = response.strip("*")
    if label.lower() == response[:len(label)].lower():
        return 1
    else:
        return 0


def judge_prediction(args):
    sample, model_name, tmp_file = args
    # check lack info
    template = prefix.replace('{{  prompt  }}', sample['instruction'])
    # template_a = prefix_a.replace('{{  prompt  }}', sample['instruction'])
    for name in sample['prediction']:
        if sample['prediction'][name] is None:
            # ana = template_a.replace('{{  response  }}', sample['response'][name])
            prompt = template.replace('{{  response  }}', sample['response'][name])
            # sample['analysis'][name] = get_response(ana, model_name)
            # prompt = prompt.replace('{{  analysis  }}', sample['analysis'][name])
            sample['prediction'][name] = get_response(prompt, model_name)
            print(f"{sample['id']}\t{name}\t{sample['prediction'][name]}")
            sample['judge'][name] = get_answer(sample['prediction'][name])
    with open(tmp_file, 'a') as file:
        file.write(json.dumps(sample) + '\n')
    return sample


def judge(data, arg):
    tmp_file = arg.tmp_file
    args_list = [(data[key], arg.model_name, tmp_file) for index, key in enumerate(data)]
    results = {}

    print(f"Load prediction from {arg.prediction}")
    judge_fcn = judge_prediction
    with Pool(arg.num_workers) as pool:
        # Prepare the pool tasks
        tasks = [pool.apply_async(judge_fcn, args=(_arg,)) for _arg in args_list]
        # Use tqdm to track progress
        for task in tqdm(tasks, total=len(tasks)):
            result = task.get()
            results[result['id']] = result
    return results


def run(arg, data):
    # data version, used for naming
    set_seed(123)
    result = judge(data, arg)
    save_data(result, arg.save_path)
    result = load_data(arg.save_path)
    benchmark = {key: 0 for key in data[next(iter(history))]['prediction']}
    for key in result:
        sample = result[key]
        for name in sample['judge']:
            benchmark[name] += sample['judge'][name]
    acc = []
    for key in benchmark:
        acc.append(f"{key}:{round(benchmark[key]*100/680,2)}")
    acc.sort()
    for item in acc:
        print(item)


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(description='evaluate using GPT')
    parser.add_argument('--prediction', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--model_name', type=str, default='gpt-4-0125-preview')
    parser.add_argument('--save_path', type=str, default='result/direct_eval/result/eval_result_mistral.json')
    parser.add_argument('--tmp_file', type=str, default='result/direct_eval/cache/eval_result_mistral.json')
    arguments = parser.parse_args()
    check_dir(arguments.save_path)
    check_dir(arguments.tmp_file)
    check = ['id', 'instruction', 'response', 'disentangle', 'response_compare']
    #  load predictions
    try:
        history = load_data(arguments.save_path)
        print('Previous results loaded')
    except FileNotFoundError as e:
        print('No history result')
        history = {}
        test = load_data('data/test/direct_eval_680.json')
        for sample in test:
            history[sample['id']] = {
                'id': sample['id'],
                'instruction': sample['instruction'],
                'response': {
                    # 'gpt-4': sample['response'],
                    # 'gpt-4_comp': sample['response_compare'],
                },
                'prediction': {
                    # 'gpt-4': None,
                    # 'gpt-4_comp': None,
                },
                'judge': {
                    # 'gpt-4': None,
                    # 'gpt-4_comp': None,
                },
                'analysis': {
                    # 'gpt-4': None,
                    # 'gpt-4_comp': None,
                }
            }

    file_list = glob('prediction/mistral2/direct_eval/*.json')
    name_list = [item.split('/')[-1][:-5] for item in file_list]
    force_key = []
    for name, path in zip(name_list, file_list):
        pred = load_data(path)
        model_name = name
        if name in history[next(iter(history))]['prediction'] and name not in force_key:
            continue

        sample = pred[0]
        key = list(sample.keys())[-1]
        assert key not in check
        print(key, sample[key])

        for sample in pred:
            if model_name not in sample:
                key = list(sample.keys())[-1]
                assert key not in check
                resp = sample[key]
            else:
                resp = sample[model_name]
            history[sample['id']]['response'][name] = resp
            history[sample['id']]['prediction'][name] = None
            history[sample['id']]['judge'][name] = None
            history[sample['id']]['analysis'][name] = None
    print_args(arguments)
    run(arguments, history)
