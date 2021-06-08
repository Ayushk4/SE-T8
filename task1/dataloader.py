import torch
import torch.nn as nn
from transformers import AutoTokenizer
try:
    from params import params
except:
    print("Params not loaded")
    class Params:
        def __init__(self):
            self.bert_type = "bert-base-cased"
            self.device = "cuda"
            self.batch_size = 16
            self.task = "QUANT"
    params = Params()

import random
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import re

typemap = {"Quantity": "QUANT",
           "MeasuredEntity": "ME", 
           "MeasuredProperty": "MP", 
           "Qualifier": "QUAL"
        }


# Dataloader
class MEDataset:
    def __init__(self, text_path, label_path):
        self.text_path = text_path
        self.label_path = label_path

        # Load all text
        self.textset = self.get_doc_ids()

        # Load all annotations
        files_with_label = [file_name[:-4] for file_name in os.listdir(label_path)]
        all_files_with_or_without_label = [file_name[:-4] for file_name in os.listdir(text_path)]
        files_without_label = list(set(all_files_with_or_without_label).difference(set(files_with_label)))
        print(len(files_with_label), len(files_without_label), len(all_files_with_or_without_label))
        self.all_data = self.load_dataset(files_with_label, files_without_label)

        # Preprocess - sentence splitting, normalizing numbers
        self.tokenized_data = self.tokenize_split_data(self.all_data)
        self.all_annotated_split_data = self.annotation_map(self.tokenized_data, self.all_data)
        print("Loaded and processed data")

        # Load Tokenizer and Batch
        self.bert_tok = AutoTokenizer.from_pretrained(params.bert_type, use_fast=True)
        print("Loaded tokenizer")
        self.batched_dataset = self.batch_dataset(self.all_annotated_split_data,
                                                  shuffle=True if "train" in text_path else False
                                                )


    def load_dataset(self, files_with_label, files_without_label):

        alldata = []

        # Load annotations from files with label
        for fn_no_ext in files_with_label:
            fn = fn_no_ext + ".tsv"
            entities = {"QUANT": [], "ME": [], "MP": [], "QUAL": []}
            with open(self.label_path+fn) as annotfile:
                text = self.textset[fn[:-4]]
                next(annotfile)
                annots = annotfile.read().splitlines()
                for a in annots:
                    annot = a.split("\t")
                    atype = typemap[annot[2]]
                    start = int(annot[3])
                    stop = int(annot[4])
                    # This is where we toss out the overlaps:
                    overlap = False
                    for ent in entities[atype]:
                        if ((start >= ent[0] and start <= ent[1]) or (stop >= ent[0] and stop <= ent[1]) or
                            (ent[0] >= start and ent[0] <= stop) or (ent[1] >= start and ent[1] <= stop)):
                            overlap = True
                    if overlap == False:    
                        entities[atype].append((start, stop, atype))
                alldata.append((text,
                                {"QUANT": entities["QUANT"],
                                 "ME": entities["ME"],
                                 "MP": entities["MP"],
                                 "QUAL": entities["QUAL"]
                                },
                                (fn,)
                            ))

        # Load annotations from files without label
        for fn in files_without_label:
            text = self.textset[fn]
            alldata.append((text,
                            {"QUANT": [],
                             "ME": [],
                             "MP": [],
                             "QUAL": []
                        },
                        (fn,)
                    ))
        return alldata

    def get_doc_ids(self):
        # Get doc ids and their text
        textset = {}
        for fn in os.listdir(self.text_path):
            with open(self.text_path+fn) as textfile:
                text = textfile.read()
                textset[fn[:-4]] = text

        return textset

    def tokenize_split_data(self, all_data):
        # Sentence Tokenize the dataset
        processed_data = []

        cnt_toks = {"figs.": 0, "fig.": 0, "et al.": 0,
                    "ref.": 0, "eq.": 0, "e.g.": 0,
                    "i.e.": 0, "nos.": 0, "no.": 0,
                    "spp.": 0
                    }
        regex_end_checker = [".*[a-zA-Z]figs\.$", 
                            ".*[a-zA-Z]fig\.$",
                            ".*[a-zA-Z]et al\.$",
                            ".*[a-zA-Z]ref\.$",
                            ".*[a-zA-Z]eq\.$",
                            ".*[a-zA-Z]e\.g\.$",
                            ".*[a-zA-Z]i\.e\.$",
                            ".*[a-zA-Z]nos\.$",
                            ".*[a-zA-Z]no\.$",
                            ".*[a-zA-Z]spp\.$",
                            # figs., fig., et al., Ref., Eq., e.g., i.e., Nos., No., spp.
                        ]

        assert len(cnt_toks) == len(regex_end_checker)

        # list of sentences
        # for every tokenized sentence obtained
            # check if ends with "fig or Figs or et al."
            # keep track of the index where the this sentence starts and end,

        all_tokenized_data = []
        for doc in all_data:
            flag = False
            sentences = sent_tokenize(doc[0])

            fixed_sentence_tokens = []
            curr_len = 0
            for s in sentences:
                if flag == True:
                    assert s[0] != ' '
                    white_length = doc[0][curr_len:].find(s[0])

                    prev_len = len(fixed_sentence_tokens[-1])
                    fixed_sentence_tokens[-1] = fixed_sentence_tokens[-1] + (" "*white_length) + s

                    assert fixed_sentence_tokens[-1][prev_len+white_length] == doc[0][curr_len+white_length], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    tmp_this_sent_len = white_length + len(s)
                    assert fixed_sentence_tokens[-1][-1] == doc[0][curr_len+tmp_this_sent_len-1], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    curr_len += tmp_this_sent_len
                else:
                    if len(fixed_sentence_tokens) != 0:
                        assert s[0] != ' '
                        white_length = doc[0][curr_len:].find(s[0])
                        fixed_sentence_tokens.append( (" "*white_length) + s )
                    else:
                        fixed_sentence_tokens.append(s)
                    assert fixed_sentence_tokens[-1][0] == doc[0][curr_len], (fixed_sentence_tokens, doc[0], curr_len, tmp_this_sent_len)
                    tmp_this_sent_len = len(fixed_sentence_tokens[-1])
                    assert fixed_sentence_tokens[-1][-1] == doc[0][curr_len+tmp_this_sent_len-1], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    curr_len += tmp_this_sent_len

                lower_cased_s = fixed_sentence_tokens[-1].lower()
                flag = False
                for i, k in enumerate(cnt_toks):
                    this_regex_pattern = regex_end_checker[i]
                    if lower_cased_s.endswith(k) and re.match(this_regex_pattern, lower_cased_s) == None:
                        cnt_toks[k] += 1
                        flag = True
                        break

            all_tokenized_data.append(fixed_sentence_tokens)      
        print("Fixed sentence splitting:", cnt_toks)
        return all_tokenized_data

    def annotation_map(self, all_tokenized_data, all_data):
        """
            Inputs:
                all_tokenized_data: list of [list of sentences]
                all_data: list of Tuple[doc, annotations, (doc_id,)]

            Outputs:
                all_annotated_split_data: 
                        [
                            Dict{'doc_id': doc_id,
                                'sentences': [list of string]
                                'offsets': [list of int]
                                'annotations': [list of Dict{"QUANT": [],
                                                            "ME": [],
                                                            "MP": [],
                                                            "QUAL": []
                                                        }
                                                ],
                    
                                ... Repeat for the second doc
                ]
        """

        # in the next loop
            # Replace all numbers by zero.
            # check if any of the annotations are used for this falls in between the start and end. Make sure no overlap
            # add offset as well.

        normalize = lambda x: re.sub(r'\d', '0', x)
        all_annotated_split_data = []
        
        for doc, sent_splits in zip(all_data, all_tokenized_data):

            this_offsets = []

            prev_end = 0
            for s in sent_splits:
                this_offsets.append([prev_end, prev_end+len(s)])
                prev_end += len(s)
            
            this_annotations = []
            for s, offset in zip(sent_splits, this_offsets):
                this_sent_ann = {}
                for k,v in doc[1].items():
                    this_key_annotation_sentence = []

                    for ann in v:
                        if offset[0] <= ann[0] and ann[1] < offset[1]:
                            this_key_annotation_sentence.append((ann[0]-offset[0], ann[1]-offset[0]))

                    this_sent_ann[k] = this_key_annotation_sentence
                
                this_annotations.append(this_sent_ann)

            all_annotated_split_data.append({'doc_id': doc[-1][0],
                        'sentences': [normalize(ss) for ss in sent_splits],
                        'offsets': this_offsets,
                        'annotations': this_annotations
                    })
            # print(all_annotated_split_data[-1])
            assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['sentences'])
            assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['annotations'])
        
        return all_annotated_split_data

    def batch_dataset(self, sentence_wise_data, shuffle=False):
        """
            Inputs:
                all_annotated_split_data: 
                        [
                            Dict{'doc_id': doc_id,
                                'sentences': [list of string]
                                'offsets': [list of int]
                                'annotations': [list of Dict{"QUANT": [],
                                                            "ME": [],
                                                            "MP": [],
                                                            "QUAL": []
                                                        }
                                                ],
                                }
                                ... Similar dict for the second doc and so on.
                ]
            Outputs:
                Batched data: [doc_ixs, sent_offsets, sentences, labels, pad_masks]
        """
        # First flatten and shuffle
        # Then Batch
        flattened = [[doc['doc_id'], doc['sentences'][i], doc['offsets'][i], doc['annotations'][i]]
                     for doc in sentence_wise_data for i in range(len(doc['sentences']))]

        print(f'Flattened {len(sentence_wise_data)} docs into {len(flattened)} data points',
                "\nSome examples:", flattened[:2])
        if shuffle:
            random.shuffle(flattened)

        cls_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[CLS]'))[0]
        sep_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[SEP]'))[0]
        pad_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[PAD]'))[0]

        
        dataset = []
        idx = 0
        num_data = len(flattened)
        while idx < num_data:
            batch_doc_ids = []
            batch_sent_offsets = []
            batch_raw_text = []
            batch_raw_labels = []

            for single_docid, single_sentence, single_offset, single_annotations in \
                        flattened[idx:min(idx+params.batch_size, num_data)]:

                batch_doc_ids.append(single_docid)
                batch_sent_offsets.append(single_offset)
                batch_raw_text.append(single_sentence)
                batch_raw_labels.append(single_annotations)


            batched_dict = self.bert_tok.batch_encode_plus(batch_raw_text,
                                                        return_offsets_mapping=True,
                                                        padding=True)


            batch_tokens = torch.LongTensor(batched_dict['input_ids']).to(params.device)
            batch_maxlen = batch_tokens.shape[-1]
            pad_masks = torch.LongTensor(batched_dict['attention_mask']).to(params.device)
            batch_offset_mapping = batched_dict['offset_mapping']

            # Create sequence labels using token offsets
            batch_labels = []
            for single_token_offs, single_anns in zip(batch_offset_mapping, batch_raw_labels):
                anns = single_anns[params.task]
                single_labels = []
                i = 0
                for off in single_token_offs:
                    if off == (0,0):
                        single_labels.append(0)
                    elif type(anns) == list and i < len(anns):
                        if off[1] < anns[i][0]:
                            single_labels.append(0)
                        elif off[0] > anns[i][1]:
                            i += 1
                            single_labels.append(0)
                        else:
                            single_labels.append(1)
                    else:
                        single_labels.append(0)
                batch_labels.append(single_labels)

            batch_labels = torch.LongTensor(batch_labels).to(params.device)

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            assert batch_tokens.size() == torch.Size([b, batch_maxlen])
            assert batch_labels.size() == torch.Size([b, batch_maxlen])
            assert pad_masks.size() == torch.Size([b, batch_maxlen])

            dataset.append((batch_tokens, batch_labels, pad_masks, batch_doc_ids, batch_sent_offsets, batch_offset_mapping))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        return dataset

# Test dataset, similar to above dataset except simpler
class METestDataset:
    def __init__(self, text_path):
        self.text_path = text_path

        # Load all text
        self.textset = self.get_doc_ids()

        # Load all annotations
        files_without_label = [file_name[:-4] for file_name in os.listdir(text_path)]
        print(len(files_without_label))
        self.all_data = self.load_dataset(files_without_label)

        # Preprocess - sentence splitting, normalizing numbers
        self.tokenized_data = self.tokenize_split_data(self.all_data)
        self.all_annotated_split_data = self.annotation_map(self.tokenized_data, self.all_data)
        print("Loaded and processed data")

        # Load Tokenizer and Batch
        self.bert_tok = AutoTokenizer.from_pretrained(params.bert_type, use_fast=True)
        print("Loaded tokenizer")
        self.batched_dataset = self.batch_dataset(self.all_annotated_split_data,
                                                  shuffle=True if "train" in text_path else False
                                                )


    def load_dataset(self, files_without_label):

        alldata = []

        # Load annotations from files without label
        for fn in files_without_label:
            text = self.textset[fn]
            alldata.append((text,
                            {"QUANT": [],
                             "ME": [],
                             "MP": [],
                             "QUAL": []
                        },
                        (fn,)
                    ))
        return alldata

    def get_doc_ids(self):
        textset = {}
        for fn in os.listdir(self.text_path):
            with open(self.text_path+fn) as textfile:
                text = textfile.read()
                textset[fn[:-4]] = text

        return textset

    def tokenize_split_data(self, all_data):
        processed_data = []

        cnt_toks = {"figs.": 0, "fig.": 0, "et al.": 0,
                    "ref.": 0, "eq.": 0, "e.g.": 0,
                    "i.e.": 0, "nos.": 0, "no.": 0,
                    "spp.": 0
                    }
        regex_end_checker = [".*[a-zA-Z]figs\.$", 
                            ".*[a-zA-Z]fig\.$",
                            ".*[a-zA-Z]et al\.$",
                            ".*[a-zA-Z]ref\.$",
                            ".*[a-zA-Z]eq\.$",
                            ".*[a-zA-Z]e\.g\.$",
                            ".*[a-zA-Z]i\.e\.$",
                            ".*[a-zA-Z]nos\.$",
                            ".*[a-zA-Z]no\.$",
                            ".*[a-zA-Z]spp\.$",
                            # figs., fig., et al., Ref., Eq., e.g., i.e., Nos., No., spp.
                        ]

        assert len(cnt_toks) == len(regex_end_checker)

        # list of sentences
        # for every tokenized sentence obtained
            # check if ends with "fig or Figs or et al."
            # keep track of the index where the this sentence starts and end,

        all_tokenized_data = []
        for doc in all_data:
            flag = False
            sentences = sent_tokenize(doc[0])

            fixed_sentence_tokens = []
            curr_len = 0
            for s in sentences:
                if flag == True:
                    assert s[0] != ' '
                    white_length = doc[0][curr_len:].find(s[0])

                    prev_len = len(fixed_sentence_tokens[-1])
                    fixed_sentence_tokens[-1] = fixed_sentence_tokens[-1] + (" "*white_length) + s

                    assert fixed_sentence_tokens[-1][prev_len+white_length] == doc[0][curr_len+white_length], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    tmp_this_sent_len = white_length + len(s)
                    assert fixed_sentence_tokens[-1][-1] == doc[0][curr_len+tmp_this_sent_len-1], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    curr_len += tmp_this_sent_len
                else:
                    if len(fixed_sentence_tokens) != 0:
                        assert s[0] != ' '
                        white_length = doc[0][curr_len:].find(s[0])
                        fixed_sentence_tokens.append( (" "*white_length) + s )
                    else:
                        fixed_sentence_tokens.append(s)
                    assert fixed_sentence_tokens[-1][0] == doc[0][curr_len], (fixed_sentence_tokens, doc[0], curr_len, tmp_this_sent_len)
                    tmp_this_sent_len = len(fixed_sentence_tokens[-1])
                    assert fixed_sentence_tokens[-1][-1] == doc[0][curr_len+tmp_this_sent_len-1], (fixed_sentence_tokens[-1], doc[0], curr_len, tmp_this_sent_len)
                    curr_len += tmp_this_sent_len

                lower_cased_s = fixed_sentence_tokens[-1].lower()
                flag = False
                for i, k in enumerate(cnt_toks):
                    this_regex_pattern = regex_end_checker[i]
                    if lower_cased_s.endswith(k) and re.match(this_regex_pattern, lower_cased_s) == None:
                        cnt_toks[k] += 1
                        flag = True
                        break

            all_tokenized_data.append(fixed_sentence_tokens)      
        print("Fixed sentence splitting:", cnt_toks)
        return all_tokenized_data

    def annotation_map(self, all_tokenized_data, all_data):
        """
            Inputs:
                all_tokenized_data: list of [list of sentences]
                all_data: list of Tuple[doc, annotations, (doc_id,)]

            Outputs:
                all_annotated_split_data: 
                        [
                            Dict{'doc_id': doc_id,
                                'sentences': [list of string]
                                'offsets': [list of int]
                                'annotations': [list of Dict{"QUANT": [],
                                                            "ME": [],
                                                            "MP": [],
                                                            "QUAL": []
                                                        }
                                                ],
                    
                                ... Repeat for the second doc
                ]
        """

        # in the next loop
            # Replace all numbers by zero.
            # check if any of the annotations are used for this falls in between the start and end. Make sure no overlap
            # add offset as well.

        normalize = lambda x: re.sub(r'\d', '0', x)
        all_annotated_split_data = []
        
        for doc, sent_splits in zip(all_data, all_tokenized_data):

            this_offsets = []

            prev_end = 0
            for s in sent_splits:
                this_offsets.append([prev_end, prev_end+len(s)])
                prev_end += len(s)
            
            this_annotations = []
            for s, offset in zip(sent_splits, this_offsets):
                this_sent_ann = {}
                for k,v in doc[1].items():
                    this_key_annotation_sentence = []

                    for ann in v:
                        if offset[0] <= ann[0] and ann[1] < offset[1]:
                            this_key_annotation_sentence.append((ann[0]-offset[0], ann[1]-offset[0]))

                    this_sent_ann[k] = this_key_annotation_sentence
                
                this_annotations.append(this_sent_ann)

            all_annotated_split_data.append({'doc_id': doc[-1][0],
                        'sentences': [normalize(ss) for ss in sent_splits],
                        'offsets': this_offsets,
                        'annotations': this_annotations
                    })
            # print(all_annotated_split_data[-1])
            assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['sentences'])
            assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['annotations'])
        
        return all_annotated_split_data

    def batch_dataset(self, sentence_wise_data, shuffle=False):
        # First flatten and shuffle
        # Then Batch
        flattened = [[doc['doc_id'], doc['sentences'][i], doc['offsets'][i], doc['annotations'][i]]
                     for doc in sentence_wise_data for i in range(len(doc['sentences']))]

        print(f'Flattened {len(sentence_wise_data)} docs into {len(flattened)} data points',
                "\nSome examples:", flattened[:2])

        cls_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[CLS]'))[0]
        sep_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[SEP]'))[0]
        pad_token_idx =self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize('[PAD]'))[0]
 
        dataset = []
        idx = 0
        num_data = len(flattened)
        while idx < num_data:
            batch_doc_ids = []
            batch_sent_offsets = []
            batch_tokens = []
            batch_raw_text = []
            batch_token_offset = []

            for single_docid, single_sentence, single_offset, single_annotations in \
                        flattened[idx:min(idx+params.batch_size, num_data)]:

                batch_doc_ids.append(single_docid)
                batch_sent_offsets.append(single_offset)
                batch_raw_text.append(single_sentence)

            batched_dict = self.bert_tok.batch_encode_plus(batch_raw_text,
                                                        return_offsets_mapping=True,
                                                        padding=True,
                                                        # return_tensors="pt",
                                                        # return_token_type_ids=True
                                                    )

            texts = torch.LongTensor(batched_dict['input_ids']).to(params.device)
            batch_maxlen = texts.shape[-1]
            pad_masks = torch.LongTensor(batched_dict['attention_mask']).to(params.device)
            batch_offset_mapping = batched_dict['offset_mapping']

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            assert texts.size() == torch.Size([b, batch_maxlen])
            assert pad_masks.size() == torch.Size([b, batch_maxlen])

            dataset.append((texts, batch_raw_text, pad_masks, batch_doc_ids, batch_sent_offsets, batch_offset_mapping))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        return dataset
