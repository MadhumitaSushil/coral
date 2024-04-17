import argparse
import os
import sys
from itertools import zip_longest

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from coral.utils.task_prompts import OncPrompt
from coral.utils.utils import append_to_csv
from coral.utils.dataprocessing import AnnotatedDataset


def load_model(model_name_or_path, local_files_only=True, device_map="auto",
               load_in_8bit=True, **kwargs):
    if not torch.cuda.is_available():
        print("No cuda found")
        load_in_8bit = False

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 local_files_only=local_files_only,
                                                 device_map=device_map,
                                                 load_in_8bit=load_in_8bit,
                                                 **kwargs
                                                 )
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print("Completed model loading")
    return model


def load_tokenizer(model_name_or_path, local_files_only=True, padding_side='left', **kwargs):
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              padding_side=padding_side,
                                              local_files_only=local_files_only, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Completed loading tokenizer")
    return tokenizer


def get_prompts_for_tasks(tasks, section_texts, model_name_or_path):
    prompt_obj = OncPrompt()
    system_msg = prompt_obj.get_system_message(model_name_or_path)
    batch_messages = list()
    for task, section_text in zip(tasks, section_texts):
        user_msg = prompt_obj.get_prompt(task)
        user_msg = section_text + user_msg

        if system_msg is None:
            cur_messages = [
                {
                    'role': 'user',
                    'content': user_msg
                }
            ]
        else:
            cur_messages = [
                        {
                            'role': 'system',
                            'content': system_msg
                         },

                        {
                            'role': 'user',
                            'content': user_msg
                        }
                    ]
        batch_messages.append(cur_messages)
    return batch_messages


def format_hf_chat_template(tokenizer, batch_messages):
    """
    Generate a single tokenized prompt from list of messages for chat-tuned models.

    Messages should be a list of messages, each of which should be in the following format:
    batch_messages = [messages]
    messages = [
            {"role": "system", "content": "Provide the system message content here."},
            {"role": "user", "content": "Provide the user content here."},
            {"role": "assistant", "content": "Provide the assistant content here."},
        ]
    """
    formatted_messages = list()
    if tokenizer.chat_template is None:
        # maybe it is better to not use a template at all and just concatenate with whitespaces?
        tokenizer.chat_template = 'default'

    for message in batch_messages:
        format_msg = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        formatted_messages.append(format_msg)

    tokenized_messages = tokenizer(formatted_messages,
                                   return_token_type_ids=False,  # token_type_ids are not needed for generation?
                                   return_tensors="pt", padding=True)
    return tokenized_messages


def get_model_response(model, tokenizer, messages, max_new_tokens=1024, only_new_text=True, **gen_kwargs):
    """
    model: Instance of Huggingface model
    tokenizer: Instance of Huggingface tokenizer
    messages: list of dictionaries with messages in the format:
                [
                {"role": "system", "content": "Provide the system message content here."},
                {"role": "user", "content": "Provide the user content here."},
                {"role": "assistant", "content": "Provide the assistant content here."},
            ]
    gen_config: dictionary of configuration parameters and values for generating model response.
    """
    tokenized_msgs = format_hf_chat_template(tokenizer, messages)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_msgs.to(device)
    outputs = model.generate(**tokenized_msgs,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             **gen_kwargs)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if only_new_text:
        # Using only the first decoded tokens since we only pass one instance.
        prompt_lens = [len(prompt) for prompt in tokenizer.batch_decode(sequences=tokenized_msgs['input_ids'],
                                                                        skip_special_tokens=True)
                       ]
        decoded = [cur_response[cur_prompt_len:] for cur_response, cur_prompt_len in zip(decoded, prompt_lens)]
    return decoded


def process_text(model, tokenizer, tasks, section_texts):
    # get prompt for task
    batch_messages = get_prompts_for_tasks(tasks, section_texts, model.name_or_path)

    # query model
    model_responses = get_model_response(model, tokenizer, batch_messages)

    return model_responses


def process_batch(batch, model, tokenizer, fout, dir_out):
    tasks = batch['task']
    section_texts = batch['section_text']
    batch['model'] = [model.name_or_path] * len(section_texts)
    batch['output'] = process_text(model, tokenizer, tasks, section_texts)
    append_to_csv(fout, dir_out, batch)


def main(fdata, dir_data, model_name_or_path, fout, dir_out, batch_size):
    ds = AnnotatedDataset(fdata, dir_data, get_labels=False)

    if 'olmo' in model_name_or_path.lower():
        import hf_olmo # Not sure how to improve dynamic loading here. TODO: brainstorm

    # load model
    model = load_model(model_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path)

    # Evaluation
    model.eval()

    if not os.path.exists(os.path.realpath(dir_out)):
        os.makedirs(os.path.realpath(dir_out))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        process_batch(batch, model, tokenizer, fout, dir_out)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero-shot inference with open source models.')
    parser.add_argument('-dir_data', type=str, default='../../data', help='data directory')
    parser.add_argument('-dir_out', type=str, default='../../output', help='output directory')

    parser.add_argument('-fdata', type=str, required=True, help='csv file containing inference data')
    parser.add_argument('-fout', type=str, required=True, help='csv file to store the output in')

    parser.add_argument('-model_name_or_path', type=str, required=True, help='huggingface model name or path')
    parser.add_argument('-batch_size', type=int, required=False, default=16, help='batch size for inference')

    args = parser.parse_args()

    main(args.fdata, args.dir_data, args.model_name_or_path, args.fout, args.dir_out, args.batch_size)
