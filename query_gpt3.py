import argparse
import json
import math
import os
from copy import copy
from datetime import datetime
from string import Template as StringTemplate

import yaml

from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
from promptsource.templates import DatasetTemplates, Template

import requests
import time
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

configs = {
    'income': {'prompts': [StringTemplate('${note}')]},
    'car': {'prompts': [StringTemplate('${note}')]},
    'heart': {'prompts': [StringTemplate('${note}')]},
    'diabetes': {'prompts': [StringTemplate('${note}')]},
    'blood': {'prompts': [StringTemplate('${note}')]},
    'bank': {'prompts': [StringTemplate('${note}')]},
    'creditg': {'prompts': [StringTemplate('${note}')]},
    'calhousing': {'prompts': [StringTemplate('${note}')]},
    'jungle': {'prompts': [StringTemplate('${note}')]},
}
public_tasks = ['income', 'car', 'heart', 'diabetes', 'blood', 'bank', 'creditg', 'calhousing', 'jungle']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=999999)
    args = parser.parse_args()

    return args


def unpack_example(example, task):
    return example


def post_request(example, model, yes_no_probability=False):
    # Remove newline and escape double quotes to prevent ERROR: Your request contained invalid JSON: Expecting ',' delimiter
    text = example['prompt']
    text = json.dumps(text)[1:-1]  # Remove additional quotes for JSON string

    print('-' * 80)
    print(text.replace('\\n', '\n'))

    if model == 'gpt3':
        url = "https://api.openai.com/v1/engines/text-davinci-002/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + OPENAI_API_KEY}
        data = '{"prompt": "' + text + '", "temperature": 0, "max_tokens": 1, "logprobs": 50}'
        response_json = requests.post(url, headers=headers, data=data.encode('utf-8')).json()

        if 'error' in response_json:
            if not response_json['error']['message'].startswith("Rate limit reached"):  # Ignore rate limit error.
                raise Exception('ERROR: ' + response_json['error']['message'] + ' ' + text)

        if yes_no_probability:
            logprobs = response_json["choices"][0]["logprobs"]["top_logprobs"][0]
            yes_prob = 0 if ' Yes' not in logprobs.keys() else math.exp(logprobs[' Yes'])
            no_prob = 0 if ' No' not in logprobs.keys() else math.exp(logprobs[' No'])
            print(f"Yes probability {yes_prob / (yes_prob + no_prob)}")
            if yes_prob == 0 and no_prob == 0:
                return 0.5
            return yes_prob / (yes_prob + no_prob)

            # For car dataset: Unacceptable ||| Acceptable ||| Good ||| Very good'
            # logprobs = response_json["choices"][0]["logprobs"]["top_logprobs"][0]
            # unacceptable_prob = 0 if ' ' not in logprobs.keys() else math.exp(logprobs[' Un'])
            # acceptable_prob = 0 if ' ' not in logprobs.keys() else math.exp(logprobs[' Accept'])
            # good_prob = 0 if ' ' not in logprobs.keys() else math.exp(logprobs[' Good'])
            # verygood_prob = 0 if ' ' not in logprobs.keys() else math.exp(logprobs[' Very'])
            # print(f"Car probs {unacceptable_prob}, {acceptable_prob}, {good_prob}, {verygood_prob}.")
            # return f"{unacceptable_prob}, {acceptable_prob}, {good_prob}, {verygood_prob}"

        output = response_json["choices"][0]["text"]

    else:
        raise ValueError('Unexpected model')

    print(output)
    print('-' * 80)
    return output


def submit_req(item, model, max_tries=300, sleep_sec=20, yes_no_probability=False):
    for i in range(max_tries):
        try:
            return post_request(item, model, yes_no_probability=yes_no_probability)
        except Exception as e:
            print(e)
            print(f"Request error; retrying in {sleep_sec} sec\n")
            time.sleep(sleep_sec)
    print("RAN OUT OF QUOTA or issues w/ API; quitting")
    return None


# From: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary
def dictify(r, root=True):
    if root:
        return {r.tag: dictify(r, False)}
    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify(x, False))
    return d


def read_dataset(task, input_file):
    # Get dataset as list of entities
    if task in public_tasks:
        # External dataset are not yet shuffled, so do it now
        orig_data = load_from_disk(input_file)

        # Without template
        # input_list = [{'note': x['note'], 'label': x['label']} for x in orig_data]

        # Load template
        yaml_dict = yaml.load(open('/root/TabLLM/templates/templates_' + task + '.yaml', "r"), Loader=yaml.FullLoader)
        prompts = yaml_dict['templates']
        # Return a list of prompts (usually only a single one with dataset_stash[1] name)
        templates_for_custom_tasks = {
            'income': '50000_dollars',
            'car': 'rate_decision',
            'heart': 'heart_disease',
            'diabetes': 'diabetes',
            'creditg': 'creditg',
            'bank': 'bank',
            'blood': 'blood',
            'jungle': 'jungle',
            'calhousing': 'calhousing',
        }
        temp = [t for k, t in prompts.items() if t.get_name() == templates_for_custom_tasks[task]][0]
        input_list = [{'note': temp.apply(x)[0], 'answer': temp.apply(x)[1], 'label': x['label']} for x in orig_data]
    else:
        raise ValueError("Invalid task name")

    dataset = [unpack_example(ex, task) for ex in input_list]
    return dataset


def main():
    time.sleep(0)
    args = parse_args()
    assert args.task in configs.keys()
    config = configs[args.task]
    outputs = pd.DataFrame()

    dataset = read_dataset(args.task, args.input)
    start_time = datetime.now().strftime("-%Y%m%d-%H%M%S")

    for k, example in enumerate(dataset):
        try:
            # if k >= 3:
            #     break
            # Only consider examples in provided range
            if k < args.start_index or k >= args.end_index:
                continue
            print(f"{k}/{len(dataset)} (from {args.start_index} to {args.end_index})")

            # Copy input into outputs
            output = example.copy()

            for i, prompt_temp in enumerate(config['prompts']):
                example['note'] = example['note'].strip()
                prompt = prompt_temp.substitute(**example)
                example['prompt'] = (prompt_temp.substitute(**example)).strip()
                output['prompt' + str(i)] = prompt
                if args.model == 'gpt3':
                    if args.task in public_tasks:
                        out = submit_req(example, args.model, yes_no_probability=False)
                    else:
                        out = submit_req(example, args.model).strip()
                    output['output' + str(i)] = out
                    time.sleep(0)
                outputs = pd.concat([outputs, pd.Series(output).to_frame(1).T], ignore_index=True)

                if args.model == 'gpt3' and k % 50 == 0:
                    # Write temporary results out
                    outputs.to_csv('output/outputs-' + args.task + start_time + '.csv', index=False)

        except Exception as e:
            print("Error occurred: " + str(e))

    if args.model == 'gpt3':
        # Final output
        outputs.to_csv('output/outputs-' + args.task + start_time + '.csv', index=False)


if __name__ == '__main__':
    main()
