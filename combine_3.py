import json
import argparse
import os

# combine the outputs from part2 and part3 predictions for ME, MP, Qual

parser = argparse.ArgumentParser()
basepath = "/home/bt1/17CS10029/measeval/MeasEval/data/"

eval_doc_path = basepath + "eval/text/"
parser.add_argument("--text_path", type=str, default=eval_doc_path)
parser.add_argument("--t2_path", type=str, required=True)
parser.add_argument("--me_path", type=str, required=True)
parser.add_argument("--mp_path", type=str, required=True)
parser.add_argument("--qual_path", type=str, required=True)
parser.add_argument("--save_path", type=str, default='final_labels.json')

params = parser.parse_args()

all_text = {f[:-4]: open(params.text_path + f, encoding="utf8").read() for f in os.listdir(params.text_path)}
j_t2 = json.load(open(params.t2_path))
j_me = json.load(open(params.me_path))
j_mp = json.load(open(params.mp_path))
j_qual = json.load(open(params.qual_path))


all_doc_labels = {docid: {q[1]: [q] for q in j_t2[docid]} for docid in all_text.keys()}
"""
DocId1: {1: [quant, me, mp, qual],
        2: [quant, me],
        ...
        }
        ...
"""

for ii, labl in enumerate(j_me):
    """
    [["S0022459611006116-987", 1, "Quantity", 119, 130, 1, "1323\u20131423 K", "{'unit': 'K'}", 0, 206], ["me", [15, 18]]]
    
    desired_format for each new entry:
        docId, annotSet, annotType, startOffset, endOffset, annotId, text, other
    """
    docId = labl[0][0]
    annotSet = labl[0][1]
    annotType = "MeasuredEntity"
    startOffset = labl[0][-2] + labl[1][-1][0]
    endOffset = labl[0][-2] + labl[1][-1][1]
    if int(startOffset) >= int(endOffset):
        continue
    annotId = "T1-"+str(ii)
    text = all_text[docId][startOffset:endOffset]
    other = {}

    all_doc_labels[docId][annotSet].append([docId, annotSet, annotType, startOffset, endOffset, annotId, text, other])

for ii, labl in enumerate(j_mp):
    docId = labl[0][0]
    annotSet = labl[0][1]
    annotType = "MeasuredProperty"
    startOffset = labl[0][-2] + labl[1][-1][0]
    endOffset = labl[0][-2] + labl[1][-1][1]
    if int(startOffset) >= int(endOffset):
        continue
    annotId = "T2-"+str(ii)
    text = all_text[docId][startOffset:endOffset]
    other = {}

    all_doc_labels[docId][annotSet].append([docId, annotSet, annotType, startOffset, endOffset, annotId, text, other])

for ii, labl in enumerate(j_qual):
    docId = labl[0][0]
    annotSet = labl[0][1]
    annotType = "Qualifier"
    startOffset = labl[0][-2] + labl[1][-1][0]
    endOffset = labl[0][-2] + labl[1][-1][1]
    if int(startOffset) >= int(endOffset):
        continue
    annotId = "T3-"+str(ii)
    text = all_text[docId][startOffset:endOffset]
    other = {}

    all_doc_labels[docId][annotSet].append([docId, annotSet, annotType, startOffset, endOffset, annotId, text, other])

# Iterate over each doc and each annotSet
    # If Qual/MP detect and not detected ME, then remove Qual/MP
    # Map the property

final_labels = {}

for docid in all_doc_labels:
    final_labels[docid] = []
    for annotset in all_doc_labels[docid].values():
        if len(annotset) > 4:
            print(annotset)
        print(len(annotset))
        this_anns = {"MeasuredProperty": None, 
                    "MeasuredEntity": None,
                    "Qualifier": None,
                    "Quantity": None
                    }
        for a in annotset:
            this_anns[a[2]] = a
        
        if this_anns['MeasuredEntity'] == None:
            this_anns["MeasuredProperty"] = None
            this_anns["Qualifier"] = None
        
        if this_anns["Qualifier"] != None:
            this_anns["Qualifier"][-1] = {"Qualifies": this_anns['MeasuredEntity'][5]}

        if this_anns["MeasuredProperty"] != None:
            this_anns["MeasuredProperty"][-1] = {"HasQuantity": str(this_anns['Quantity'][5])}
            this_anns['MeasuredEntity'][-1] = {"HasProperty": this_anns['MeasuredProperty'][5]}
        else:
            if this_anns['MeasuredEntity'] != None:
                this_anns['MeasuredEntity'][-1] = {"HasQuantity": str(this_anns['Quantity'][5])}

        final_labels[docid].append(this_anns)

json.dump(final_labels, open(params.save_path, 'w+'))
