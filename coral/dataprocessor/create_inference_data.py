import argparse
import csv
import os

import pandas as pd

from coral import *
from coral.dataprocessor.annots_for_inference import InferenceAnnots
from coral.dataprocessor.brat import Collection

class Annotation:
    def __init__(self, doc_idx, section_name, section_text, inf_subtype, annotations):
        self.doc_idx = doc_idx
        self.section_name = section_name
        self.section_text = section_text
        self.inf_subtype = inf_subtype
        self.annotations = annotations

    def serialize(self, fannot, dir_annot):
        headers = ['doc_idx', 'section_name', 'section_text', 'inference_subtype',
                   'annotation_set']

        data = [str(self.doc_idx), self.section_name, self.section_text, self.inf_subtype,
                self.annotations]

        with open(os.path.join(dir_annot, fannot), 'a') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if f.tell() == 0:
                # write header to empty file
                csvwriter.writerow(headers)
            csvwriter.writerow(data)


class AnnotationColl:

    def __init__(self, annot_lst):
        self.annotation_coll = annot_lst

        self.processed = {str(cur_annot.doc_idx) + cur_annot.section_name
                          + cur_annot.inf_subtype
                          for cur_annot in self.annotation_coll}

    @classmethod
    def read_existing_coll(cls, fname_coll, dir_coll):
        df = pd.read_csv(os.path.join(dir_coll, fname_coll), quotechar='"')
        df.drop_duplicates(inplace=True)
        annot_lst = list()

        for row in df.itertuples(index=False):
            annots = [eval(cur_annot) for cur_annot in row.annotation_set.strip().split('\n')]

            cur_annot = Annotation(row.doc_idx, row.section_name, row.section_text,
                                   row.inference_subtype, annots)

            annot_lst.append(cur_annot)

        return cls(annot_lst)

    def is_processed(self, example):
        if example in self.processed:
            return True
        else:
            return False

    def get_annot_from_coll(self, doc_idx, section_name, inf_subtype):
        matching_annots = list()
        for annot in self.annotation_coll:
            if (annot.doc_idx == doc_idx and annot.section_name == section_name and
                    annot.inf_subtype == inf_subtype):
                matching_annots.append(annot)

        return matching_annots


class OncInfoExtr:
    def __init__(self, annot_data_dir,
                 sections_to_infer=('hpi', 'a&p'),
                 fdata='onc_pn_ie_data.csv',
                 data_dir='../data/',
                 ):

        self.collection = Collection.read_collection(annot_data_dir)
        self.sections_to_infer = sections_to_infer

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(data_dir)

        self.fname_ie_data = fdata

    def get_elements_for_inference(self, entities, attributes, relations):
        inference_annots = InferenceAnnots(entities, attributes, relations)
        advanced_inference_tuple_dict = inference_annots.get_tuples_for_advanced_inference()

        return advanced_inference_tuple_dict

    def serialize_annotated_ie_dataset(self):
        if os.path.exists(os.path.join(self.data_dir, self.fname_ie_data)):
            existing_coll = AnnotationColl.read_existing_coll(self.fname_ie_data, self.data_dir)
        else:
            existing_coll = None

        for doc_idx, doc in self.collection.documents.items():
            for section in self.sections_to_infer:
                cur_text, cur_ents, cur_atts, cur_rels = self.collection.get_annots_by_section_name(doc, section)
                cur_text = cur_text.replace('', ' ')
                adv_inf_tuples = self.get_elements_for_inference(cur_ents, cur_atts, cur_rels)

                for cur_inf_type, tuple in adv_inf_tuples.items():
                    str_tuple = ""
                    for cur_item in tuple:
                        str_tuple += str(cur_item) + '\n'

                    if existing_coll and existing_coll.is_processed(
                            doc_idx + section + cur_inf_type):
                        print("Skipping duplicate data entry")
                        continue

                    annot_obj = Annotation(doc_idx, section, cur_text, cur_inf_type, str_tuple)
                    # serialize the input data
                    annot_obj.serialize(self.fname_ie_data, self.data_dir)


def main(annot_data_dir, fdata, data_dir):
    ie_extractor = OncInfoExtr(
        annot_data_dir=annot_data_dir,
        fdata=fdata,
        data_dir=data_dir
    )

    ie_extractor.serialize_annotated_ie_dataset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero-shot inference with open source models.')
    parser.add_argument('-annot_data_dir', type=str, default='../data/annotated', help='output directory')

    parser.add_argument('-fdata', type=str, default='data_onc_pn_ie.csv', help='csv file containing inference data')
    parser.add_argument('-dir_data', type=str, default='../../data/', help='data directory')

    args = parser.parse_args()

    main(args.annot_data_dir, args.fdata, args.dir_data)
