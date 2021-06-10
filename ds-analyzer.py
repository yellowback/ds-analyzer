#!/usr/bin/env python
import argparse
import datasets
datasets.logging.set_verbosity_warning()
from transformers import AutoTokenizer
import numpy as np
import os
import json

def analyze(column, ds_tokenized, ds_unk):
    target_list = np.array(ds_tokenized)
    unk_list = np.array(ds_unk)
    return {
        "mean": round(target_list.mean(),1),
        "std": round(target_list.std(),1),
        "median": round(np.median(target_list),1),
        "unk_mean": round(unk_list.mean(),3)
    }

def get_all_columns(ds):
    return ','.join([k for k,v in ds.features.items() if v.dtype == 'string'])

def load_ds(file):
    _, ext = os.path.splitext(file)
    ext = ext[1:]
    ds_args = { 'data_files': file }
    if ext == 'jsonl':
        ext = 'json'
    elif ext == 'tsv':
        ext = 'csv'
        ds_args['delimiter'] = '\t'

    ds = datasets.load_dataset(ext, **ds_args)
    return ds['train']

def ds_analyze(filename, columns, tokenizer_name):
    ds = load_ds(filename)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if columns == '':
        columns = get_all_columns(ds)

    result = {}
    for column in columns.split(','):
        def decode(example):
            encoded = tokenizer.encode(example[column], add_special_tokens=False)
            return {'tokenized_len': len(encoded), 'unk_count' : encoded.count(tokenizer.unk_token_id)}
        ds_tokenized = ds.map(decode)
        result[column] = analyze(column, ds_tokenized['tokenized_len'], ds_tokenized['unk_count'])
    return result

def main(args):
    analyzed = ds_analyze(args.target_file, args.columns, args.tokenizer_name)
    print(json.dumps(analyzed,ensure_ascii=False, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_file', help='target file(csv,tsv,json)')
    parser.add_argument('--columns', default='')
    parser.add_argument('--tokenizer_name', default='google/mt5-base', required=True)
    args = parser.parse_args()

    main(args)
    

