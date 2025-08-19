from utils.data import load_data, save_data
import random
from utils.common import set_seed


def collect_sample(path='data/WizardLM_143k.json', save_path='data/WizardLM_alpaca.json'):
    set_seed(123)
    data = load_data(path)
    random.shuffle(data)
    processed = []
    counter = 0
    for item in data:
        if 'alpaca' not in item['idx']: continue
        instruction = item['conversations'][0]['value']
        size = len(instruction.split(' '))
        # if 10 < size < 200:
        sample = {'id': item['idx'], 'instruction': instruction}
        processed.append(sample)
        counter += 1
        # if counter == 10000: break
    print(counter)
    save_data(processed, save_path)


def process_extracted(path='result/WizardLM_alpaca/cache/gpt-4-0125-preview_det_fact.json',
                      save_path='data/alpaca_all.json'):
    data = load_data(path)
    error = 0
    new = []
    for sample in data:
        det = sample['det']
        instruction = sample['instruction']
        try:
            cut = det.index('**Extract Instructions:**')
        except Exception as e:
            print(instruction)
            print(det)
            error += 1
            continue
        fact_chunk = det[:cut]
        inst_chunk = det[cut:]

        fact_counter = 0
        for piece in fact_chunk.split('\n'):
            if len(piece.strip()) == 0 or "**Extract" in piece: continue
            fact_counter += 1
        sample['num_fact'] = fact_counter

        inst_counter = 0
        for piece in inst_chunk.split('\n'):
            if len(piece.strip()) == 0 or "**Extract" in piece: continue
            inst_counter += 1
        sample['num_inst'] = inst_counter
        new.append(sample)
    print(f"Missing:{error}")
    save_data(new, save_path)


class check_num_inst:
    def __init__(self, target_num_inst):
        self.target_num_inst = target_num_inst

    def __call__(self, sample):
        if sample['num_inst'] != self.target_num_inst:
            return False
        else:
            return True


def random_sample(num_samples, path='data/alpaca_all.json', save_path='data/alpaca_all_rand100.json',
                  checker=check_num_inst(4)):
    set_seed(123)
    data = load_data(path)
    selected = []
    for item in data:
        if checker(item):
            selected.append(item)
    rand = random.sample(selected, num_samples)
    save_data(rand, save_path)


def split_examples(sample, new):
    results = new.split('**\n')[1:]
    filtered = []
    for row in results:
        if '\n\n**' in row:
            row = row[:row.index('\n\n**')]
        filtered.append(row)
    sample['parallel'] = []
    if len(filtered) % 2 != 0:
        print(filtered)
        sample['check'] = 'wrong_new_prompt'
        return sample
    for idx in range(int(len(filtered) / 2)):
        inst = filtered[2 * idx]
        p = filtered[2 * idx + 1]
        if 'INVALID' in p:
            continue
        change = f"**Modified Instructions:**\n{inst}"
        sample['parallel'].append({'change': change, 'instruction': p})
    return sample
