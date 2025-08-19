from utils.data import load_data, read_file
from utils.cost import estimate_bill

path = 'data/train/train1k6.json'
prefix_det = read_file('prefix/fea_det.txt')
prefix_change = read_file('prefix/change_inst.txt')
prefix_compare = read_file('prefix/get_new_response.txt')
data = load_data(path)

model_name =  'gpt-4-0125-preview'

input_det_all = []
output_det_all = []

# det 15.37 USD

input_all = []
output_all = []

for sample in data:
    seed_inst = sample['instruction']
    input_det = prefix_det.replace("{{  Prompt  }}", seed_inst)
    output_det = sample['disentangle']
    input_det_all.append(input_det)
    output_det_all.append(output_det)



data = load_data('result/obsolete/alpaca_all/cache/gpt-4-0125-preview_2000.json')

input_change_all = []
output_change_all = []

input_response_all = []
output_response_all = []
input_compare_all = []
output_compare_all = []

for sample in data:
    if sample['check'] == 'pass':

        input_all.append(sample['instruction'])
        output_all.append(sample['response'])
        inst = sample['instruction']
        input_change = prefix_det.replace("{{  Prompt  }}", inst).replace("{{  Extracted  }}", sample['det'])
        input_compare = prefix_compare.replace("{{  Original Prompt  }}", sample['instruction']).replace("{{  Original Response  }}", sample['response'])
        input_change_all.append(input_change)
        output_change = ''
        for i, item in enumerate(sample['parallel']):
            output_change += f'**Modified Instructions {i}:**\n'
            output_change += item['change']+'\n\n'
            output_change += f'**Revised Prompt {i}:**\n'
            output_change += item['instruction']+'\n\n'
            input_response_all.append(item['instruction'])
            input_compare_all.append(input_compare.replace("{{  New Prompt  }}", item['instruction']))
            output_response_all.append(item['response'])
            output_compare_all.append(item['response_compare'])
        output_change_all.append(output_change)



estimate_bill(input_det_all,output_det_all, model_name)

estimate_bill(input_change_all, input_change_all, model_name)


estimate_bill(input_all, output_all,model_name)

estimate_bill(input_response_all, output_response_all, model_name)

estimate_bill(input_compare_all, output_compare_all, model_name)




