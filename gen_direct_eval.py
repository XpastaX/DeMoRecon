import argparse
import pprint
import os
from tqdm import tqdm
import torch
from transformers import GenerationConfig
from multiprocessing import Pool
from gen_detection import save_data, load_data, format_prompt, get_model, update_args, encode

B_INST, E_INST = "[INST] ", " [/INST]"
prefix = "A chat between a curious user and an artificial intelligence assistant. " \
         "The assistant gives helpful, detailed, and polite answers to the user's questions. " \
         "USER: {{  Prompt  }} ASSISTANT: "
prefix_new = "{{  Prompt  }}\n\n### Response:"


def make_prompt(sample):
    instruction = sample['instruction']
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

    chunks = [data[i::NUM_WORKER] for i in range(NUM_WORKER)]
    GPU_index = args.gpu_id
    print(f'Using GPU:{GPU_index}')
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


@torch.no_grad()
def process_chunk(chunk_data):
    args, index, chunk, GPUs = chunk_data
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(GPUs)
    print(f"Worker:{index}|Sample:{len(chunk)}|GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")
    tokenizer, model = get_model(base_model=args.model_path, device=index, args=args)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        # max_length=max(args.max_len, 8192),
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.0,
    )

    result = []

    for sample in tqdm(chunk, disable=False):
        prompt = make_prompt(sample)
        prompt = format_prompt(prompt, args.template, tokenizer)
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
            sample[args.model_name] = completion_seq
        result.append(sample)
    return result


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
