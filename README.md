# ds-analyzer

Load csv/tsv/jsonl dataset and tokenize each line of the column and then calc mean/std/median tokens.

## Requirements

- huggingface transformers
- huggingface datasets

```
$ pip install transformers datasets
```

## Usage

```
$ python ds-analyzer \
    --tokenizer_name google/mt5-base \
	mydataset.csv
```

OUTPUT: 
```
{
  "label": {
    "mean": 20.3,
    "std": 6.4,
    "median": 20.0,
    "unk_mean": 0.0
  },
  "text": {
    "mean": 588.6,
    "std": 349.1,
    "median": 504.5,
    "unk_mean": 0.003
  }
}
```
