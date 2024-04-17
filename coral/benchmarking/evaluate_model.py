import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from itertools import product

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from coral import *
from coral.utils.dataprocessing import read_data, parse_output, format_tuple_annots_for_eval
from coral.utils.metrics import Metrics


def get_annots(fdata, dir_data):
    df = read_data(fdata, dir_data, get_annots=True)
    proc_annots = list()
    for row in df.itertuples(index=False):
        annots = [eval(cur_annot) for cur_annot in row.annotation_set.strip().split('\n')]
        if not len(annots):
            annots = [task_to_default_tuple_dict[row.task]]
        proc_annots.append(annots)
    df['proc_annots'] = proc_annots
    df.to_csv(os.path.join(dir_data, 'proc_' + fdata), index=False)
    print("Annots", df.shape, df.columns)
    return df


def get_outputs(fout, dir_out):
    df = pd.read_csv(os.path.join(os.path.realpath(dir_out), fout))
    df.drop_duplicates(inplace=True)
    proc_outputs = list()
    n_parsed_outputs, total_parse_errors = 0, 0
    for row in df.itertuples(index=False):
        parsed_output, n_errors = parse_output(row.output, row.task)
        proc_outputs.append(parsed_output)
        total_parse_errors += n_errors
        n_parsed_outputs += len(parsed_output)
    print("Total number of parsed outputs: ", n_parsed_outputs)
    print("Total number of parsing errors: ", total_parse_errors)
    assert len(proc_outputs) == df.shape[0], "Number of processed outputs more than original outputs"
    df['proc_outputs'] = proc_outputs
    df.to_csv(os.path.join(dir_out, 'parsed_'+fout), index=False)
    print("Outputs:", df.shape, df.columns)
    return df


def evaluate(df_data, df_out, f_inst_score, f_agg_scores, f_final_scores, dir_score,
             eval_type='relation'):
    metrics = Metrics(tokenizer='default')
    scores = list()

    # one output row contains one section of a note
    for (doc_idx, section_name) in product(df_out['doc_idx'].unique(),
                                           df_out['section_name'].unique()
                                           ):
        for model in df_out['model'].unique():
            for task in df_out['task'].unique():
                # get outputs for the instance
                outputs = df_out[(df_out['doc_idx'] == doc_idx) &
                                    (df_out['section_name'] == section_name) &
                                    (df_out['model'] == model) &
                                    (df_out['task'] == task)
                                 ]
                # get annots for the instance
                annots = df_data[(df_data['doc_idx'] == doc_idx) &
                                 (df_data['section_name'] == section_name) &
                                 (df_data['task'] == task)
                                 ]

                if outputs.shape[0] == 0:
                    outputs = [task_to_default_tuple_dict[task]]
                else:
                    outputs = outputs['proc_outputs'].item()

                if annots.shape[0] == 0:
                    annots = [task_to_default_tuple_dict[task]]
                else:
                    annots = annots['proc_annots'].item()

                annots = format_tuple_annots_for_eval(annots, eval_type=eval_type)
                outputs = format_tuple_annots_for_eval(outputs, eval_type=eval_type)
                if not len(outputs):
                    outputs = format_tuple_annots_for_eval([task_to_default_tuple_dict[task]], eval_type=eval_type)

                # outputs is a dictionary {relation_task: [all outputs of that relation type]}
                for cur_subtask in outputs.keys():
                    cur_outs = [item.lower() for item in outputs[cur_subtask]]
                    cur_annots = [item.lower() for item in annots[cur_subtask]]

                    # computing mean bleu and rouge for multi-set output
                    bleus = metrics.compute_multiset_bleus(cur_outs, cur_annots, max_n=4, smooth=True)
                    rouges = metrics.compute_multiset_rouges(cur_outs, cur_annots, rouge_types=['rouge1'])

                    em_p, em_r, em_f1 = metrics.compute_em_over_multiset_prec_recall_f1(cur_outs, cur_annots)

                    cur_scores = {
                        'doc_idx': doc_idx,
                        'section_name': section_name,
                        'task': task,
                        'subrelation': cur_subtask,
                        'model': model,
                        'bleu4': bleus,
                        'rouge1': rouges['rouge1'],
                        'em_prec': em_p,
                        'em_recall': em_r,
                        'em_f1': em_f1,
                    }

                    scores.append(cur_scores)

    scores = pd.DataFrame(scores)
    print(scores)
    scores.to_csv(os.path.join(dir_score, f_inst_score), index=False)
    aggregate_scores(f_inst_score, f_agg_scores, f_final_scores, dir_score, eval_type)


def aggregate_scores(f_inst_scores, f_agg_scores, f_final_scores, dir_out, eval_type):
    df = pd.read_csv(os.path.join(dir_out, f_inst_scores))
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    agg_df = df.groupby(['task', 'model', 'subrelation']).agg(mean_bleu4=('bleu4', 'mean'),
                             mean_rouge1=('rouge1', 'mean'),
                             mean_em_prec=('em_prec', 'mean'),
                             mean_em_recall=('em_recall', 'mean'),
                             mean_em_f1=('em_f1', 'mean'),
                             )

    agg_df.to_csv(os.path.join(dir_out, f_agg_scores), index=True)
    agg_df = agg_df.reset_index()
    agg_df = _reorganize_scores_df(agg_df, eval_type)

    agg_df.to_csv(os.path.join(dir_out, f_final_scores), index=False)


def _reorganize_scores_df(scores_df, eval_type):
    def _modify_med_rel(row):
        if '_med_' in row['task']:
            prefix = row['task'].split('_')[0]
            return f'{prefix.title()}{row["subrelation"]}'
        else:
            return row['subrelation']

    def _modify_symptom_rel(row, eval_type):
        if 'symptoms' in row['task']:
            prefix = ''.join([subword.capitalize() for subword in row['task'].split('_')])
            if eval_type == 'relation':
                return f'{prefix} {row["subrelation"].split()[1]}'
            elif eval_type == 'entity':
                return f'{prefix}'
        else:
            return row['subrelation']

    # Apply the function to each row in the DataFrame
    scores_df['subrelation'] = scores_df.apply(_modify_med_rel, axis=1)
    scores_df['subrelation'] = scores_df.apply(lambda x: _modify_symptom_rel(x, eval_type), axis=1)

    new_df = list()
    metrics = [col for col in scores_df.columns if 'mean' in col]
    for cur_rel in scores_df['subrelation'].unique():

        cur_dict = dict()
        cur_dict['Relation'] = cur_rel
        for metric in metrics:
            for model in scores_df['model'].unique():
                cur_df = scores_df[(scores_df['subrelation'] == cur_rel) &
                                   (scores_df['model'] == model)
                                   ]
                model = os.path.basename(model.rstrip('/'))
                cur_dict[metric[5:].upper()+'_'+model.upper()] = round(cur_df[metric].item(), 2)
        new_df.append(cur_dict)

    new_df = pd.DataFrame(new_df)

    # Rename fields for plotting
    new_df['Relation'].replace('PrescribedMedicationName PotentialAdvEvent',
                               'PrescribedMedicationName PotentialAdverseEvent', inplace=True)
    new_df['Relation'].replace('PrescribedMedicationName ConfirmedAdvEvent',
                               'PrescribedMedicationName ConfirmedAdverseEvent', inplace=True)
    new_df['Relation'].replace('FutureMedicationName PotentialAdvEvent',
                                'FutureMedicationName PotentialAdverseEvent', inplace=True)

    return new_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero-shot inference with open source models.')
    parser.add_argument('-dir_data', type=str, default='../../data', help='data directory')
    parser.add_argument('-dir_out', type=str, default='../../output', help='output directory')

    parser.add_argument('-fdata', type=str, default='onc_pn_ie_data.csv', help='csv file containing inference data')
    parser.add_argument('-fout', type=str, default='output_onc_pn_ie.csv', help='csv file to store the output in')

    parser.add_argument('-fscores_inst', type=str, default='rel_instance_level_scores.csv',
                        help='csv file to store instance level scores')
    parser.add_argument('-fscores_agg', type=str, default='rel_agg_scores.csv',
                        help='csv file to store aggregated scores')
    parser.add_argument('-fscores_reformatted', type=str, default='rel_reformatted_agg_scores.csv',
                        help='csv file to store reformatted aggregated scores')

    args = parser.parse_args()

    df_data = get_annots(args.fdata, args.dir_data)
    df_out = get_outputs(args.fout, args.dir_out)

    evaluate(df_data, df_out,
             args.fscores_inst,
             args.fscores_agg,
             args.fscores_reformatted,
             args.dir_out, eval_type='relation')

