import csv
import os
from itertools import zip_longest


def _writerow(file_pointer, csvwriter, headers, values):
    if file_pointer.tell() == 0:
        # write header to empty file
        csvwriter.writerow(headers)
    if isinstance(list(values)[0], list):
        for cur_vals in zip_longest(*values):
            csvwriter.writerow(cur_vals)
    else:
        csvwriter.writerow(values)


def append_to_csv(fout, dir_out, data):
    """
    fout: output filename,
    dir_out: directory name
    data: dictionary of {header: content}, or list of such dictionaries, or dictionary of lists with multiple values,
        each containing data to serialize
    """
    with open(os.path.join(os.path.realpath(dir_out), fout), 'a') as f:
        csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if type(data) == list:
            for cur_data in data:
                _writerow(f, csvwriter, cur_data.keys(), cur_data.values())
        else:
            _writerow(f, csvwriter, data.keys(), data.values())
