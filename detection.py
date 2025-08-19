import argparse
from utils.common import set_seed, check_dir
from utils.data import load_data, save_data
from glob import glob
import os
from sklearn.metrics import balanced_accuracy_score as score


def get_answer(resp, label):
    resp = resp.lower().strip().strip('- ').strip('*').replace('assistant\n', '')
    # if len(resp)>10:
    #     return not label
    ans = 'yes' if label else 'no'
    pred = label if ans == resp[:len(ans)] else not label
    # if not pred:
    #     print(f'======{ans}========')
    #     print(resp)
    return pred


def run(args):
    # initialize prediction
    try:
        prediction = load_data(args.save_path)
    except FileNotFoundError:
        print(f"Prediction file not found: {args.save_path}")
        prediction = load_data(args.data_path)
        for sample in prediction:
            sample['result'] = {}
        prediction = {sample['id']: sample for sample in prediction}

    pred_file = glob(args.prediction_path + '*.json')
    for path in pred_file:
        pred = load_data(path)
        for sample in pred:
            sample['model_name'] = path.split('/')[-1][:-5]
        for sample in pred:
            prediction[sample['id']]['result'][sample['model_name']] = \
                {'prediction': get_answer(sample['response'], prediction[sample['id']]['label']),
                 'response': sample['response']}
    save_data(prediction, args.save_path)
    cal_acc(prediction, args)


def cal_acc(prediction,args):
    keys = list(prediction.keys())
    models = list(prediction[keys[0]]['result'].keys())
    models.sort()
    size = len(prediction)
    acc = {m: 0 for m in models}
    inference = {m: [] for m in models}
    label = []
    for key in prediction:
        pred = prediction[key]['result']
        label.append(prediction[key]['label'])
        for m in pred:
            inference[m].append(pred[m]['prediction'])
    for m in acc:
        s = score(label, inference[m])
        acc[m] = round(s*100, 2)
        print(f"{m}:{acc[m]}")
    save_data(acc, os.path.join(args.prediction_path, 'result/acc.json'))


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(description='evaluate using GPT')
    parser.add_argument('--data_path', type=str, default="data/test/detection.json", help='detection_test_path')
    parser.add_argument('--prediction_path', type=str, default="prediction/mistral/detection/", help='parent path of all LLMs')
    args = parser.parse_args()
    # store all prediction of all LLMs
    args.save_path = os.path.join(args.prediction_path, 'result/prediction_mistral_v1.json')
    check_dir(args.save_path)
    run(args)
