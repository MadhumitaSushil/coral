from collections import defaultdict

import numpy as np

import evaluate
import transformers


class Metrics:

    def __init__(self, tokenizer='default', is_f1_scorer=True, is_em_acc_scorer=True,
                 is_bleu_scorer=True, is_rouge_scorer=True):
        if tokenizer.lower() == 'gpt':
            self.tokenizer = transformers.OpenAIGPTTokenizerFast.from_pretrained("openai-gpt")
        elif tokenizer.lower() == 'default':
            self.tokenizer = None
        else:
            print("Supported tokenizers: (gpt|default). Using default.")
            self.tokenizer = None

        if is_bleu_scorer:
            self.bleu = evaluate.load("bleu")

        if is_rouge_scorer:
            self.rouge = evaluate.load('rouge')

        if is_em_acc_scorer:
            self.em_acc = evaluate.load('exact_match')

        if is_f1_scorer:
            self.f1_func = evaluate.load("f1")

    def compute_bleu_score(self, preds, references, max_n=1, smooth=False, **kwargs):
        """
        :param preds: list of all predictions
        :param references: list of list of all references for all predictions.
        :return: results
        """

        if self.tokenizer is not None:
            results = self.bleu.compute(predictions=preds, references=references, max_order=max_n, smooth=smooth,
                                   tokenizer=self.tokenizer.tokenize, **kwargs
                                   )
        else:
            results = self.bleu.compute(predictions=preds, references=references, max_order=max_n, smooth=smooth,
                                   **kwargs)

        return results

    def compute_rouge_score(self, preds, references, rouge_types=None, **kwargs):
        """
        :param preds:  list of all predictions
        :param references: list of all references for all predictions (list of list)
        :param rouge_types: A list of rouge types to calculate
        :param kwargs: any additional supported parameters for metric computation
        :return: rouge score results
        """
        if self.tokenizer is not None:
            results = self.rouge.compute(predictions=preds, references=references, rouge_types=rouge_types,
                                    tokenizer=self.tokenizer.tokenize, **kwargs
                                    )
        else:
            results = self.rouge.compute(predictions=preds, references=references, rouge_types=rouge_types,
                                    use_stemmer=True, **kwargs
                                    )
        return results

    def compute_em_accuracy(self, preds, references, **kwargs):
        """
        Exact match accuracy across set of values
        :param preds: list of all predictions
        :param references: list of list of all references for all predictions.
        :return: results

        """
        return self.em_acc.compute(predictions=preds, references=references, **kwargs)

    def compute_em_F1(self, preds, references, pos_label=None, average="macro", **kwargs):
        """
        Macro or micro F1 score for classification values
        """
        f1_score = self.f1_func.compute(references=references, predictions=preds, average=average, pos_label=pos_label,
                                   **kwargs)

        return f1_score

    def compute_multiset_bleus(self, cur_outs, cur_annots, max_n, smooth):
        bleus = list()
        for cur_out in cur_outs:
            bleu = self.compute_bleu_score(preds=[cur_out], references=[cur_annots],
                                           max_n=max_n, smooth=smooth
                                           )['bleu']
            bleus.append(bleu)
        bleus = np.mean(bleus)
        return bleus

    def compute_multiset_rouges(self, cur_outs, cur_annots, rouge_types):
        rouge_scores = defaultdict(list)
        for cur_annot in cur_annots:
            cur_rouges = defaultdict(list)
            for cur_out in cur_outs:
                rouges = self.compute_rouge_score(preds=[cur_out], references=[cur_annot],
                                                  rouge_types=rouge_types
                                                  )
                for rouge_type in rouge_types:
                    cur_rouges[rouge_type].append(rouges[rouge_type])
            for rouge_type in rouge_types:
                rouge_scores[rouge_type].append(np.max(cur_rouges[rouge_type]))

        return {k: np.mean(v) for k, v in rouge_scores.items()}

    def compute_em_over_multiset_prec_recall_f1(self, model_outputs, annots):
        model_outputs = set(model_outputs)
        annots = set(annots)

        TP = len(model_outputs.intersection(annots))
        FP = len(model_outputs - annots)
        FN = len(annots - model_outputs)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        if precision == 0. and recall == 0.:
            f1_score = 0.
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score
