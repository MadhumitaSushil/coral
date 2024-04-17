import re
import os
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset

from coral import *


class AnnotatedDataset(Dataset):
    def __init__(self, fdata, dir_data, get_labels):
        self.df = read_data(fdata, dir_data, get_labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        return row


def read_data(fdata, dir_data, get_annots=False):
    print("Loading data")
    df = pd.read_csv(os.path.join(os.path.realpath(dir_data), fdata))
    df = df.rename(columns={'inference_subtype': 'task'})
    if not get_annots:
        df = df.drop(columns=['annotation_set'])

    df.drop_duplicates(inplace=True)
    print("Completed data loading")
    return df


def _escape_quote(cur_output):
    pattern = r"[A-Za-z]'([A-Za-z])"
    cur_output = re.sub(pattern, r"\'\1", cur_output)
    return cur_output


def _extract_patterns(input_string):
    pattern = r"[A-Za-z]+\(.+\{.+\}\)"
    matches = re.findall(pattern, input_string)
    return matches


def parse_output(output, task):
    parsed_outputs = list()
    n_parse_errors = 0
    for cur_out in _extract_patterns(output):
        cur_out = _escape_quote(cur_out)
        cur_out = cur_out.strip()
        try:
            cur_out = eval(cur_out)
        except Exception as e:
            # print("Could not parse the following:", e, "\n", cur_out, " Original output: ", output)
            n_parse_errors += 1
            continue

        # sometimes the regex returns maximal match instead of minimal, which is inferred as a tuple
        if type(cur_out) == tuple:
            cur_out = [eval(item) if type(item) == str else item for item in cur_out]
            parsed_outputs.extend(cur_out)
        else:
            parsed_outputs.append(cur_out)

    if len(parsed_outputs) == 0:
        parsed_outputs.append(task_to_default_tuple_dict[task])
    return parsed_outputs, n_parse_errors


def format_tuple_annots_for_eval(tuple_list, eval_type='relation'):
    assert eval_type in ['relation', 'entity'], "Please enter a valid eval_type (relation|entity)."

    subtask2vals = defaultdict(set)

    for cur_tuple in tuple_list:
        if type(cur_tuple) == CancerDiagnosis:
            # print("skipping first cancer diagnosis")
            continue
        tuple_entity_type = cur_tuple._fields[0]
        tuple_entity_val = cur_tuple[0]

        if eval_type == 'entity':
            if len(tuple_entity_val):
                 subtask2vals[tuple_entity_type].add(tuple_entity_val)
        elif eval_type == 'relation':
            for i, (cur_subent_type, cur_subent_val) in enumerate(cur_tuple._asdict().items()):
                if not i or cur_subent_type == 'AdditionalTesting':
                    # print("Skipping additional testing")
                    continue
                if len(cur_subent_val) == 0:
                    cur_subent_val = 'unknown'
                if type(cur_subent_val) == dict:
                    cur_subent_val = set(cur_subent_val.values())
                if type(cur_subent_val) == set or type(cur_subent_val) == list:
                    for val in cur_subent_val:
                        subtask2vals[tuple_entity_type + ' ' + cur_subent_type].add(tuple_entity_val + ' ' + val)
                else:
                    subtask2vals[tuple_entity_type + ' ' + cur_subent_type].add(tuple_entity_val + ' ' + cur_subent_val)

    if not len(subtask2vals):
        print("No outputs found for tuples: ", tuple_list)

    return subtask2vals