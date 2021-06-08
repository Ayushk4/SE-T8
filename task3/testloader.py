import torch
import torch.nn as nn
from transformers import AutoTokenizer
try:
    from params import params
except:
    print("Params not loaded")
    class Params:
        def __init__(self):
            self.bert_type = "allenai/biomed_roberta_base"
            self.device = "cpu"
            self.batch_size = 16
    params = Params()

import random
import os
import json
import re


class METestDataset:
    def __init__(self, text_path, quant_label_path):
        self.text_path = text_path
        self.quant_label_path = quant_label_path

        # Load all text
        self.textset = self.get_doc_ids()

        # Load all annotations
        all_files_with_text = list(self.textset.keys())
        all_quant_labels = json.load(open(self.quant_label_path))
        sorted(list(all_quant_labels.keys())) == sorted(all_files_with_text)
        print(len(all_files_with_text), len(all_quant_labels))
        
        # Also preprocess - normalizing numbers
        self.all_data = self.load_dataset(all_files_with_text, all_quant_labels)
        print("Loaded and processed data")

        # Load Tokenizer and Batch
        self.bert_tok = AutoTokenizer.from_pretrained(params.bert_type, use_fast=True)
        print("Loaded tokenizer")
        # Add new tokens in tokenizer
        new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>"]}
        self.bert_tok.add_special_tokens(new_special_tokens_dict)

        self.batched_dataset = self.batch_dataset(self.all_data)

    def load_dataset(self, all_files_with_text, all_quant_labels):

        alldata = []
        normalize = lambda x: re.sub(r'\d', '0', x)

        # Load annotations from files with label
        for file_id in all_files_with_text:
            doc_text = self.textset[file_id]

            for quants in all_quant_labels[file_id]:
                this_quant_sent = doc_text[quants[8]:quants[9]]
                this_quant_offsets_from_sent_start = [quants[3] - quants[8], quants[4] - quants[8]]
                alldata.append([normalize(this_quant_sent), quants, this_quant_offsets_from_sent_start])

        return alldata

    def get_doc_ids(self):
        textset = {}
        for fn in os.listdir(self.text_path):
            with open(self.text_path+fn) as textfile:
                text = textfile.read()
                textset[fn[:-4]] = text

        return textset


    def batch_dataset(self, sentence_mapped_data):
        """
            Inputs:
                    [ [sent, list of quant1_details from task2, quant_offset],
                      [sent, list of quant2_details from task2],
                      .....

                    ]

            Outputs:
                Batched data: [doc_ixs, sent_offsets, sentences, token_offsets, pad_masks]
        """
        flattened = sentence_mapped_data
        print(f'Flattened doc - {len(sentence_mapped_data)}',
                "\nSome examples:", flattened[:2]
            )

        cls_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize(self.bert_tok.cls_token))[0]
        sep_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize(self.bert_tok.sep_token))[0]
        pad_token_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize(self.bert_tok.pad_token))[0]

        special_st_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize("<E>"))
        special_end_idx = self.bert_tok.convert_tokens_to_ids(self.bert_tok.tokenize("</E>"))
        assert len(special_st_idx) == 1
        assert len(special_end_idx) == 1
        special_st_idx, special_end_idx = special_st_idx[0], special_end_idx[0]

        print("CLS, SEP, PAD, <E>, </E> tokens are:", cls_token_idx, sep_token_idx, pad_token_idx, special_st_idx, special_end_idx)

        dataset = []
        idx = 0
        num_data = len(flattened)

        while idx < num_data:
            batch_quant_data = []
            batch_raw_text = []
            batch_raw_labels = []

            for single_sentence, quant_data, quant_offsets_from_sent_start in \
                        flattened[idx:min(idx+params.batch_size, num_data)]:
                offset = quant_offsets_from_sent_start
                single_raw_anns = [[offset[0], offset[1], 'QUANT']]

                single_raw_text = single_sentence

                batch_quant_data.append(quant_data)
                batch_raw_text.append(single_raw_text)
                batch_raw_labels.append(single_raw_anns)

            batched_dict = self.bert_tok.batch_encode_plus(batch_raw_text,
                                return_offsets_mapping=True, padding=True)


            # Create sequence labels using token offsets
            no_special_batch_tokens = batched_dict['input_ids']
            no_special_batch_pads = batched_dict['attention_mask']
            no_special_batch_offset_mapping = batched_dict['offset_mapping']
            
            batch_tokens = []
            batch_pads = []
            batch_offset_mapping = []
            batch_labels = {"QUANT": []}

            # Add <E> and </E>
            for single_sent_tokens, single_sent_pad_masks, single_token_offs, single_sent_anns in \
                    zip(no_special_batch_tokens, no_special_batch_pads, no_special_batch_offset_mapping, batch_raw_labels):

                this_tokens = []
                this_pads = []
                this_offset_mapping = []
                this_labels = {"QUANT": []}

                quant_label, entity_label, property_label, qual_label = None, None, None, None
                for ann in single_sent_anns:
                    if ann[2] == "QUANT":
                        quant_label = [ann[0], ann[1]]
                    else:
                        raise ann

                quant_happening = 0
                for i, off in enumerate(single_token_offs):
                    if off == (0,0):
                        if quant_happening == 1: 
                            # Append for </E>, if the code reaches here then quant label has ended
                            this_tokens.append(special_end_idx)
                            this_pads.append(1)
                            this_offset_mapping.append((0,0))
                            this_labels["QUANT"].append(1)
                        quant_happening = 0

                        this_tokens.append(single_sent_tokens[i])
                        this_pads.append(single_sent_pad_masks[i])
                        this_offset_mapping.append(off)
                        this_labels["QUANT"].append(0)
                    else:
                        if off[1] < quant_label[0] or off[0] > quant_label[1]:
                            if quant_happening == 1: 
                                # Append for </E>, if the code reaches here then quant label has ended
                                this_tokens.append(special_end_idx)
                                this_pads.append(1)
                                this_offset_mapping.append((0,0))
                                this_labels["QUANT"].append(1)
                            quant_happening = 0
                        else:
                            if quant_happening == 0:
                                # Append for <E>, if the code reaches here then quant label has started
                                this_tokens.append(special_st_idx)
                                this_pads.append(1)
                                this_offset_mapping.append((0,0))
                                this_labels["QUANT"].append(1)
                            quant_happening = 1

                        # Check for all_quant and append
                        this_tokens.append(single_sent_tokens[i])
                        this_pads.append(single_sent_pad_masks[i])
                        this_offset_mapping.append(off)

                        if off[1] >= quant_label[0] and off[0] <= quant_label[1]:
                            this_labels["QUANT"].append(1)
                        else:
                            this_labels["QUANT"].append(0)

                batch_tokens.append(this_tokens)
                batch_pads.append(this_pads)
                batch_offset_mapping.append(this_offset_mapping)
                batch_labels["QUANT"].append(this_labels["QUANT"])

            max_batch_len = max([len(bt) for bt in batch_tokens])
            min_batch_len = min([len(bt) for bt in batch_tokens])
            assert max_batch_len == min_batch_len or max_batch_len == min_batch_len + 2

            token_pad_function_maxlen = lambda x, maxlen: x if len(x) == maxlen else x + [pad_token_idx, pad_token_idx]
            mask_pad_function_maxlen = lambda x, maxlen: x if len(x) == maxlen else x + [0, 0]
            label_pad_function_maxlen = lambda x, maxlen: x if len(x) == maxlen else x + [0, 0]

            batch_tokens = [token_pad_function_maxlen(sentence, max_batch_len) for sentence in batch_tokens]
            batch_tokens = torch.LongTensor(batch_tokens).to(params.device)

            batch_pads = [mask_pad_function_maxlen(sentence, max_batch_len) for sentence in batch_pads]
            pad_masks = torch.LongTensor(batch_pads).to(params.device)

            batch_maxlen = batch_tokens.shape[-1]
            for k in batch_labels:
                padded_labels = [label_pad_function_maxlen(sentence, max_batch_len) for sentence in batch_labels[k]]
                batch_labels[k] = padded_labels
            batch_ann_labels = {k: torch.LongTensor(batch_labels[k]).to(params.device)
                                    for k in batch_labels
                                }

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            assert batch_tokens.size() == torch.Size([b, batch_maxlen])
            for vvvvv in batch_ann_labels.values():
                assert vvvvv.size() == torch.Size([b, batch_maxlen])
            assert pad_masks.size() == torch.Size([b, batch_maxlen])

            dataset.append((batch_tokens, batch_ann_labels, pad_masks, batch_offset_mapping, batch_quant_data))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        return dataset
