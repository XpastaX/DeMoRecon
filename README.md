# DeMoRecon
Code &amp; Data for CIKM 2025 paper: Enhancing and Assessing Instruction-Following with Fine-Grained Instruction Variants 
This code is only for data generation and result evaluation. For training, please use Lamma-Factory: https://github.com/hiyouga/LLaMA-Factory

## Result Evaluation

**gen_direct_eval.py**: generate LLM response for direct_eval  task

**direct_eval.py**: use GPT model to evaluate the result of direct_eval  task

**gen_detection.py**: generate LLM response for detection  task

**detection.py**: evaluate the response of LLMs for detection task

## Data Generation

**extract.py**: extract base elements of each instruction

**gen_response.py**: generate the responses for input instructions

**make_new.py**: modify base elements and generate new prompt variants

## Model CheckPoints:
https://huggingface.co/Pasta009/models

