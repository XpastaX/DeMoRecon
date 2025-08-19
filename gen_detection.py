import json
import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from multiprocessing import Pool
import subprocess

B_INST, E_INST = "[INST] ", " [/INST] "

COT = "Answer with 'Yes' or 'No', then gives step by step analysis to proof your answer"
DIRECT = "Answer with 'Yes' or 'No' only, do not explain"
REP = "{{  Method  }}"


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
        json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def format_prompt(txt, template, tokenizer=None):
    if template == 'llama':
        return B_INST + txt + E_INST
    elif template == 'mistral':
        return "[INST] " + txt + " [/INST]"
        # messages = [
        #     {"role": "user", "content": txt}
        # ]
        # texts = tokenizer.apply_chat_template(
        #     messages,
        #     # chat_template=TEMPLATE,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     padding=False,
        #     max_length=8192,
        #     truncation=True)
        # return texts
    elif template == 'qwen':
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": txt}
        ]
        texts = tokenizer.apply_chat_template(
            messages,
            # chat_template=TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
            padding=False,
            max_length=8192,
            truncation=True)
        return texts
    elif template in ['baichuan', 'orion']:
        messages = [{"role": "user", "content": txt}, ]
        return messages
    else:
        return txt


def get_model(load_8bit: bool = False, base_model: str = "bigcode/starcoder", device=None, args=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )
    if 'qwen' in base_model:
        print('Modifying tokenizer for qwen model')
        tokenizer.bos_token_id = tokenizer.encode(
            text='<|im_start|>',
            add_special_tokens=False
        )[0]
        tokenizer.eos_token_id = tokenizer.encode(
            text='<|im_end|>',
            add_special_tokens=False
        )[0]
        print(tokenizer.bos_token_id, tokenizer.eos_token_id)
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    return tokenizer, model


def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def make_prompt(sample):
    instruction = sample['instruction'].replace(REP, DIRECT) + ' The answer is'
    return instruction


def run(args):
    NUM_GPU = args.num_gpus
    NUM_WORKER = args.num_workers
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # load data
    data = load_data(args.data_path)

    num_samples = len(data)
    print("Number of samples: {}".format(num_samples))

    # chunks = list(chunkify(task_data, len(task_data) // NUM_GPU))
    chunks = [data[i::NUM_WORKER] for i in range(NUM_WORKER)]
    # GPU_index = [str(item) for item in list(range(NUM_GPU))]
    GPU_index = args.gpu_id
    GPUs = [GPU_index[i::NUM_WORKER] for i in range(NUM_WORKER)]
    print(f"Seperate Data into {len(chunks)} chunks")
    chunk_data = [(args, index, chunk, GPUs[index]) for index, chunk in enumerate(chunks)]
    pool = Pool(NUM_WORKER)
    results = pool.map(process_chunk, chunk_data)
    pool.close()
    pool.join()

    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    save_data(flat_results, args.save_path)


def encode(tokenizer, prompt):
    input_ids = tokenizer.encode(
        text=prompt,
        add_special_tokens=False
    )
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    model_inputs = torch.tensor([input_ids])
    return model_inputs


@torch.no_grad()
def process_chunk(chunk_data):
    args, index, chunk, GPUs = chunk_data
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(GPUs)
    print(f"Worker:{index}|Sample:{len(chunk)}|GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")
    tokenizer, model = get_model(base_model=args.model_path, device=index, args=args)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        # max_length=max(args.max_len, 8192),
        max_new_tokens=3,
        do_sample=False,
        repetition_penalty=1.0,
    )

    result = []

    for sample in tqdm(chunk, disable=False):
        sample['model_name'] = args.model_name
        prompt = make_prompt(sample)
        # regularize the answer
        prompt = format_prompt(prompt, args.template, tokenizer)  # + 'The answer is '
        if args.template in ['baichuan', 'orion']:
            gen_seqs = model.chat(tokenizer, prompt)
            if type(gen_seqs) != list:
                gen_seqs = [gen_seqs]
        else:
            encoding = encode(tokenizer, prompt)[:int(args.max_len / 2)]
            gen_tokens = model.generate(
                input_ids=encoding.to('cuda'),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config
            )
            if gen_tokens is not None:
                gen_tokens = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(encoding, gen_tokens)
                ]
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

        if gen_seqs is not None:
            if type(gen_seqs) == list:
                gen_seq = gen_seqs[-1]
            else:
                gen_seq = gen_seqs
            if E_INST in gen_seq:
                completion_seq = gen_seq.split(E_INST)[-1]
            elif "### Response:" in gen_seq:
                completion_seq = gen_seq.split("### Response:")[-1]
            else:
                completion_seq = gen_seq
            sample['response'] = completion_seq
        result.append(sample)
    return result


def get_num_gpus():
    try:
        n = len(
            subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n


def update_args(args):
    args.model_name = args.model_path.split('/')[-1]
    if args.gpu_id == 'default':
        args.num_gpus = get_num_gpus()
        args.gpu_id = [str(i) for i in range(args.num_gpus)]
    else:
        args.gpu_id = args.gpu_id.split(',')
        args.num_gpus = len(args.gpu_id)
    if args.num_workers == -1:
        args.num_workers = args.num_gpus
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help="")
    parser.add_argument('--data_path', type=str, default='', help="")
    parser.add_argument('--save_path', type=str, default='', help="")
    parser.add_argument('--template', type=str, default='llama', help="")
    parser.add_argument('--gpu_id', type=str, default='default', help="")
    parser.add_argument('--num_workers', type=int, default=-1, help="")
    parser.add_argument('--max_len', type=int, default=4096, help="")

    args = parser.parse_args()
    args = update_args(args)

    run(args)
