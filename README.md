## Overview

Coming Soon

arxiv link - coming soon.

## Dependencies and set-up

| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.7     | `conda create --name measeval python=3.7` and `conda activate measeval` |
| PyTorch, cudatoolkit    | 1.4.0, 10.1   | `conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 4.2.2 | `pip install transformers==4.2.2` |
| Scikit-learn | 0.23.2 | `pip install scikit-learn==0.23.2` |
| scipy        | 1.5.0  | `pip install scipy==1.5.0` |
| nltk    | 0.5.1  | `pip install nltk` |
| pandas        | 1.2.2      | `pip install pandas` |
| pandasql        | 0.7.3     | `pip install pandasql` |
| vladiate        | 0.0.23      | `pip install vladiate` |
| numpy        | 1.18.1      | - |


## Instructions

### Setting up the Code-base

0. download this repository and name the folder as 'measeval'

1. `cd measeval`

2. Clone the dataset and other utils - `git clone https://github.com/harperco/MeasEval`

3. Set up the dependencies:
    - `conda create --name measeval python=3.8`
    - `conda activate measeval`
    - `conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch`
    - `pip install transformers==4.2.2`
    - `pip install scikit-learn==0.23.2`
    - `pip install scipy==1.5.0`
    - `pip install nltk`
    - `pip install pandas`
    - `pip install pandasql`
    - `pip install vladiate`

4. Set the paths: change the lines 10, 328 in task2/label.py, 15, 217 in task1/train.py, 16, 287 in task3/train.py and 6 in combine_3.py to set your basepaths.

### Part1: 

`python3 task1/train.py --batch=16 --lr=3e-5 --n_epochs=5 --device=cuda --bert="bert-base-cased"`

You may change the following parameters:
- Batch-size: `--batch`
- Number of Epochs: `--n_epochs`
- Learning rate: `--lr`
- Bert-type: `--bert`
- Device: `--device` cuda or cpu

Output Folder: `task1/output/$(bert-type)`
- `model.pt`: Model weights 
- `params.json`: Hyperparams
- `test_spans.json`: Test set quantity span predicts
- `train_spans.json`: Train set quantity span predicts
- `trial_spans.json`: Trial set quantity span predicts


### Part2:

`python3 task2/label.py --quant_path="task1/output/bert-base-cased/"`

Input: output from part 1

You may change the following parameters:
- `--quant_path`: path to your output folder from part1

Output Folder: `task2/output/$(bert-type)`
- `test_labels.json`: Test set quantity span, unit and modifier predictions
- `train_labels.json`: Train set quantity span, unit and modifier predictions
- `trial_labels.json`: Trial set quantity span, unit and modifier predictions


## Part3:

`python3 task3/train.py --batch=16 --lr=3e-5 --n_epochs=10 --device=cuda --bert="bert-base-cased" --me --mp --qual --test_label_folder="task2/output/bert-base-cased"`

Input: output from part 2

You may change the following parameters:
- Batch-size: `--batch`
- Number of Epochs: `--n_epochs`
- Learning rate: `--lr`
- Bert-type: `--bert`
- Device: `--device` cuda or cpu
- Multi-task learning: if you want to do ME, then include `--me` flag; if you want to do MP, then include `--mp` flag; if you want to do Qual, then include `--qual` flag. You can use any combination of these with at least one of these three flags provided.
- `--test_label_folder`: path to your output folder from part2


Output Folder: `task3/output/$(bert-type)`
- `model.pt`: Model weights 
- `params.json`: Hyperparams
- `test_spans_me.json`: Test set quantity span predicts me
- `train_spans_me.json`: Train set quantity span predicts me
- `trial_spans_me.json`: Trial set quantity span predicts me
- `test_spans_mp.json`: Test set quantity span predicts mp
- `train_spans_mp.json`: Train set quantity span predicts mp
- `trial_spans_mp.json`: Trial set quantity span predicts mp
- `test_spans_qual.json`: Test set quantity span predicts qual
- `train_spans_qual.json`: Train set quantity span predicts qual
- `trial_spans_qual.json`: Trial set quantity span predicts me




### Combine and Bring to desired format:

1. Combine:
`python3 combine_3.py --me task3/output/bert-base-cased/test_spans_me.json --mp task3/output/bert-base-cased/test_spans_mp.json --qual task3/output/bert-base-cased/test_spans_qual.json --t2_path task2/output/bert-base-cased/test_labels.json`

You may change the following parameters:
- ME testset prediction path: `--me`
- MP testset prediction path: `--mp`
- Qual testset prediction path: `--qual`
- Task2 testset prediction path: `--t2`


Output - `final_labels.json`

2. Convert json to tsv:
`python3 json_to_tsv.py`

Output - predictions inside `tsv` folder

3. Run their evaluation script:
`python measeval-eval.py -i ./ -s tsv/ -g MeasEval/data/eval/tsv/ -m class`

OR 

`python measeval-eval.py -i ./ -s tsv/ -g MeasEval/data/eval/tsv/`

Output isn't stored, only printed.


## Trained Models

| Model          | Part1 | Part3 |
| -------------- | ----- | ----- |
| Biomed Roberta | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/biomed_roberta1.pt) | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/biomed_roberta3.pt) |
| Bert large     | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/bert_large1.pt) | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/bert_large3.pt) |
| Bert base      | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/bert_base1.pt) | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/bert_base3.pt) |
| SciBert        | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/scibert1.pt) | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/scibert3.pt) |
| BioBert        | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/biobert1.pt) | [link](https://github.com/Ayushk4/SE-T8/releases/download/v0.0.0/biobert3.pt) |


## Performance

| Model          | Quant     | ME        | MP        | Qual      | Unit     | Mod  | HQ        | HP        | Overall   |
| -------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | 
| BERT-base      | 0.828     | 0.338     | 0.277     | 0.072     | 0.765     | **0.465** | 0.310     | 0.174     | 0.402     |
| BERT-large     | 0.705     | 0.343     | 0.296     | 0.081     | 0.755     | 0.442     | 0.325     | 0.207     | 0.392     |
| RoBERTa-BioMed | 0.812     | 0.384     | 0.365     | 0.104     | 0.804     | 0.434     | 0.383     | 0.238     | 0.440     |
| SciBERT        | 0.809     | 0.382     | 0.324     | 0.072     | **0.811** | 0.435     | 0.354     | 0.230     | 0.433     |
| BioBERT        | **0.844** | **0.407** | **0.365** | **0.111** | 0.796     | **0.465** | **0.400** | **0.269** | **0.456** |


## Miscellanous

- You may contact us by opening an issue on this repo. Please allow 2-3 days of time to address the issue.

- The evaluation script was borrowed from this baseline [here](https://github.com/harperco/MeasEval)

- License: MIT
