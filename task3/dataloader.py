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

        # Map ME, MP, Qual to the Quant.
        self.mapped_data = self.map_dataset(self.all_data)

        # Preprocess - sentence splitting, normalizing numbers
        self.tokenized_data = self.tokenize_split_data(self.all_data)
        self.all_sentence_mapped_data = self.sentence_map(self.tokenized_data, self.mapped_data)
        print("Loaded and processed data")

        # Load Tokenizer and Batch
        self.bert_tok = AutoTokenizer.from_pretrained(params.bert_type)
        print("Loaded tokenizer")
        # Add new tokens in tokenizer
        new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>"]}
        self.bert_tok.add_special_tokens(new_special_tokens_dict)
        print("LEN VOCAB:", len(self.bert_tok))
        self.batched_dataset = self.batch_dataset(self.all_sentence_mapped_data,
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
                    other = None if annot[-1] == '' else json.loads(annot[-1])
                    ann_id = annot[1]
                    # This is where we toss out the overlaps:
                    overlap = False
                    # for ent in entities[atype]:
                    #     if ((start >= ent[0] and start <= ent[1]) or (stop >= ent[0] and stop <= ent[1]) or
                    #         (ent[0] >= start and ent[0] <= stop) or (ent[1] >= start and ent[1] <= stop)):
                    #         overlap = True
                    if overlap == False:    
                        entities[atype].append((start, stop, atype, ann_id, other))
                alldata.append((text,
                                {"QUANT": entities["QUANT"],
                                 "ME": entities["ME"],
                                 "MP": entities["MP"],
                                 "QUAL": entities["QUAL"]
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

    def map_dataset(self, all_data):
        '''
        Output:
            [
                [text,
                 List[
                        q1_start, q1_stop,
                        q1_atype, q1_ann_id,
                        [list of max length 3, containing corresponding ME, MP, QUAL]
                 ],
                 (filename,
                 )
                ]
            ]
        '''
        mapped_dataset = []
        for doc in all_data:
            this_quant_data = []
            for single_quant in doc[1]['QUANT']:
                this_ann_id = single_quant[3]
                this_quant_props = []
                for me in doc[1]['ME']:
                    if this_ann_id == me[-2]:
                        this_quant_props.append(me)
                for mp in doc[1]['MP']:
                    if this_ann_id == mp[-2]:
                        this_quant_props.append(mp)
                for qual in doc[1]['QUAL']:
                    if this_ann_id == qual[-2]:
                        this_quant_props.append(qual)

                this_quant_data.append([single_quant[0], single_quant[1],
                                        single_quant[2], single_quant[3],
                                        single_quant[4], this_quant_props
                                    ])

            mapped_dataset.append((doc[0], this_quant_data, doc[-1]))
        return mapped_dataset

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

    def sentence_map(self, all_tokenized_data, mapped_data):
        """
            Inputs:
                all_tokenized_data: list of [list of sentences]
                mapped_data: list of Tuple[doc, quant-wise-annotations, (doc_id,)]

            Outputs:
                all_annotated_split_data: 
                        [
                            Dict{'doc_id': doc_id,
                                'sentences': [list of string]
                                'offsets': [list of int]
                                'annotations': [ list of [sent_ix, sent_offset, offset_adjusted annotations for a single quant]
                                                 ... same for other quants
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
        
        for doc, sent_splits in zip(mapped_data, all_tokenized_data):

            this_offsets = []

            prev_end = 0
            for s in sent_splits:
                this_offsets.append([prev_end, prev_end+len(s)])
                prev_end += len(s)
            
            this_annotations = [] # Each element of the format: (sent_ix, anns)

            for ann in doc[1]:
                if len(ann[-1]) == 0:
                    lowest_char_ix = ann[0]
                    highest_char_ix = ann[1]
                else:
                    lowest_char_ix = min(ann[0], min(rel_ann[0] for rel_ann in ann[-1]))
                    highest_char_ix = max(ann[1], max(rel_ann[1] for rel_ann in ann[-1]))
                
                iii = 0
                # Calc low ix.
                min_sent_ix = -1
                while iii < len(this_offsets):
                    if lowest_char_ix < this_offsets[iii][1]:
                        min_sent_ix = iii
                        break
                    iii += 1

                # Calc high sent ix
                max_sent_ix = -1
                while iii < len(this_offsets):
                    if highest_char_ix <= this_offsets[iii][1]:
                        max_sent_ix = iii
                        break
                    iii += 1
                assert min_sent_ix != -1
                assert max_sent_ix != -1

                if min_sent_ix != max_sent_ix:
                    # skip, since it will be difficult during test time.
                    pass
                else:
                    # Append ann
                    this_ann_off = this_offsets[min_sent_ix][0]
                    offset_modified_ann = [ann[0]-this_ann_off, ann[1]-this_ann_off, ann[2], ann[3], ann[4],
                                            [(rel_ann[0]-this_ann_off, rel_ann[1]-this_ann_off, rel_ann[2], rel_ann[3], rel_ann[4])
                                                for rel_ann in ann[5]
                                            ]
                                        ]
                    this_annotations.append([min_sent_ix, this_ann_off, offset_modified_ann])

            all_annotated_split_data.append({'doc_id': doc[-1][0],
                        'sentences': [normalize(ss) for ss in sent_splits],
                        'offsets': this_offsets,
                        'annotations': this_annotations
                    })
            # print(all_annotated_split_data[-1])
            for ta in this_annotations:
                assert ta[0] >= 0 and ta[0] < len(sent_splits)
            assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['sentences'])
            # assert len(all_annotated_split_data[-1]['offsets']) == len(all_annotated_split_data[-1]['annotations'])
        
        return all_annotated_split_data

    def batch_dataset(self, sentence_mapped_data, shuffle=False):
        """
            Inputs:
                sentence_mapped_data: 
                        [
                            Dict{'doc_id': doc_id,
                                'sentences': [list of string]
                                'offsets': [list of int]
                                'annotations': [ list of [sent_ix, sent_offset, offset_adjusted annotations for a single quant]
                                                 ... same for other quants
                                                ],
                    
                                ... Repeat for the second doc
                ]
            Outputs:
                Batched data: [doc_ixs, sent_offsets, sentences, labels, pad_masks]
        """
        # First flatten and remove overlapping labels and then shuffle
        # Then Batch
        flattened = [[doc['doc_id'], doc['sentences'][doc['annotations'][i][0]],
                      doc['annotations'][i][1], doc['annotations'][i]]
                    for doc in sentence_mapped_data for i in range(len(doc['annotations']))
                ]

        print(f'Flattened {len(sentence_mapped_data)} docs into {len(flattened)} data points',
                "\nSome examples:", flattened[:2])

        def has_overlap(ann, datapt):
            no_overlap = True
            sss = sorted([ann[:2]] + [list(ff[:2]) for ff in ann[-1]], key = lambda x: x[0])
            for i in range(len(sss)-1):
                if sss[i][1] > sss[i+1][0]:
                    no_overlap = False
                    print("\n\nWARNING: Discarding due to overlap", datapt, '\n\n')
                    break
            return no_overlap

        flattened = [f for f in flattened if has_overlap(f[-1][-1], f)]
        print("Length after discarding overlaps:", len(flattened))
        
        if shuffle:
            random.shuffle(flattened)

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
            batch_doc_ids = []
            batch_sent_offsets = []
            batch_raw_text = []
            batch_raw_labels = []

            for single_docid, single_sentence, single_offset, single_annotations_with_offset_sentence_idx in \
                        flattened[idx:min(idx+params.batch_size, num_data)]:
                
                single_annotations = single_annotations_with_offset_sentence_idx[-1]
                single_raw_anns = [single_annotations[:3]] + [list(ff[:3]) for ff in single_annotations[-1]]

                # quant_st, quant_end = (single_raw_anns[0][0], single_raw_anns[0][1])
                single_raw_text = single_sentence#[:quant_st] + " <E> " + \
                            #single_sentence[quant_st:quant_end] + " </E> " + single_sentence[quant_end:]
                

                batch_doc_ids.append(single_docid)
                batch_sent_offsets.append(single_offset)
                batch_raw_text.append(single_raw_text)
                batch_raw_labels.append(single_raw_anns)

            batched_dict = self.bert_tok.batch_encode_plus(batch_raw_text,
                                                        return_offsets_mapping=True,
                                                        padding=True)


            # Create sequence labels using token offsets
            no_special_batch_tokens = batched_dict['input_ids']
            no_special_batch_pads = batched_dict['attention_mask']
            no_special_batch_offset_mapping = batched_dict['offset_mapping']
            
            batch_tokens = []
            batch_pads = []
            batch_offset_mapping = []
            batch_labels = {"QUANT": [], "ME": [], "MP": [], "QUAL": []}

            # Add <E> and </E>
            for single_sent_tokens, single_sent_pad_masks, single_token_offs, single_sent_anns in \
                    zip(no_special_batch_tokens, no_special_batch_pads, no_special_batch_offset_mapping, batch_raw_labels):

                this_tokens = []
                this_pads = []
                this_offset_mapping = []
                this_labels = {"QUANT": [], "ME": [], "MP": [], "QUAL": []}

                quant_label, entity_label, property_label, qual_label = None, None, None, None
                for ann in single_sent_anns:
                    if ann[2] == "QUANT":
                        quant_label = [ann[0], ann[1]]
                    elif ann[2] == "ME":
                        entity_label = [ann[0], ann[1]]
                    elif ann[2] == "MP":
                        property_label = [ann[0], ann[1]]
                    elif ann[2] == "QUAL":
                        qual_label = [ann[0], ann[1]]
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
                            this_labels["MP"].append(0)
                            this_labels["ME"].append(0)
                            this_labels["QUAL"].append(0)
                        quant_happening = 0

                        this_tokens.append(single_sent_tokens[i])
                        this_pads.append(single_sent_pad_masks[i])
                        this_offset_mapping.append(off)
                        this_labels["QUANT"].append(0)
                        this_labels["MP"].append(0)
                        this_labels["ME"].append(0)
                        this_labels["QUAL"].append(0)
                    else:
                        if off[1] < quant_label[0] or off[0] > quant_label[1]:
                            if quant_happening == 1: 
                                # Append for </E>, if the code reaches here then quant label has ended
                                this_tokens.append(special_end_idx)
                                this_pads.append(1)
                                this_offset_mapping.append((0,0))
                                this_labels["QUANT"].append(1)
                                this_labels["MP"].append(0)
                                this_labels["ME"].append(0)
                                this_labels["QUAL"].append(0)
                            quant_happening = 0
                        else:
                            if quant_happening == 0:
                                # Append for <E>, if the code reaches here then quant label has started
                                this_tokens.append(special_st_idx)
                                this_pads.append(1)
                                this_offset_mapping.append((0,0))
                                this_labels["QUANT"].append(1)
                                this_labels["MP"].append(0)
                                this_labels["ME"].append(0)
                                this_labels["QUAL"].append(0)
                            quant_happening = 1

                        # Check for all_quant and append
                        this_tokens.append(single_sent_tokens[i])
                        this_pads.append(single_sent_pad_masks[i])
                        this_offset_mapping.append(off)

                        if off[1] >= quant_label[0] and off[0] <= quant_label[1]:
                            this_labels["QUANT"].append(1)
                        else:
                            this_labels["QUANT"].append(0)

                        if entity_label != None and off[1] >= entity_label[0] and off[0] <= entity_label[1]:
                            this_labels["ME"].append(1)
                        else:
                            this_labels["ME"].append(0)

                        if property_label != None and off[1] >= property_label[0] and off[0] <= property_label[1]:
                            this_labels["MP"].append(1)
                        else:
                            this_labels["MP"].append(0)

                        if qual_label != None and off[1] >= qual_label[0] and off[0] <= qual_label[1]:
                            this_labels["QUAL"].append(1)
                        else:
                            this_labels["QUAL"].append(0)

                batch_tokens.append(this_tokens)
                batch_pads.append(this_pads)
                batch_offset_mapping.append(this_offset_mapping)
                batch_labels["QUANT"].append(this_labels["QUANT"])
                batch_labels["MP"].append(this_labels["MP"])
                batch_labels["ME"].append(this_labels["ME"])
                batch_labels["QUAL"].append(this_labels["QUAL"])

            batch_tokens = torch.LongTensor(batch_tokens).to(params.device)
            pad_masks = torch.LongTensor(batch_pads).to(params.device)

            batch_maxlen = batch_tokens.shape[-1]
            batch_ann_labels = {k: torch.LongTensor(batch_labels[k]).to(params.device)
                                    for k in batch_labels
                                }

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            assert batch_tokens.size() == torch.Size([b, batch_maxlen])
            for vvvvv in batch_ann_labels.values():
                assert vvvvv.size() == torch.Size([b, batch_maxlen])
            assert pad_masks.size() == torch.Size([b, batch_maxlen])

            dataset.append((batch_tokens, batch_ann_labels, pad_masks, batch_doc_ids, batch_sent_offsets, batch_offset_mapping, batch_raw_text))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        return dataset
