# A few imports and set up our paths
import itertools
import numpy as np
import random
import os
import re
from os import path
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd

# Extracted keywords and units for faster processing   
list_units = ['varves','hour','s−1','g mol−1','mg/l','kgs-1','beach materials','kW','g/m3','scale heights','μs','×','cm−1','mm2','mM'
        'Mbps','mbar','hr','passages','participants','μbar','day','m2 g−1','mg','oN','μg/m','kg','GPa','μM','women'
        'stems/ha','nbar','vertices','U/ml','m thick','month','ppt','W m−2','centimeters','second','mg/ml','mWm−2','°C','μg/L','wt.%'
        'times','months','ppq','M','km','mBar','cm2/Vs','Ma','H','point','monomer units','mA/cm2','nT','cm3 g−1','m'
        '°','metre','°N','bp','v/v','Mt','cores','Whitehall II participants','Rs','m/s','mdeg','Torr','Å/s','occasions','clones'
        'men','%','kHz','MPa','mH','MW','orders of magnitude','year-old','m s−1','UT','hours','h','μL','weeks','discrete particles'
        '% per year','wt%','cm−3','kV','fold','lines of code','days','nJ','years','km/h','week','km s−1','ppm','mmol/L','year'
        'Rp','elderly participants','g cm−3','bit','AU','MeV','mm per side','° latitude','mm3','yrs','pairs per mm','K/min',
        'Mg ha−1 year−1','wt. %','K''men','%','kHz','MPa','mH','MW','orders of magnitude','year-old','m s−1','UT','hours','h','μL','weeks','discrete particles'
        'byr','kR','s','mA g− 1','°S','min','mbsl','m−2','°C/min','W/m2','minutes','percentage points per year','nm','vol%','degrees'
        'mg cm− 2','pH','keV','mA','ka','employees','fold per passage','mm','RRh','eV','ms','μg','kg s−1','w/w','item'
        'μg/ml','μm','p0','g/L','horizons','V','Hz','SDG vertices','cm','items','percent','mg/L','ppm by mass','‰','g'
        'Saturn radii RS'
    ]


            
set_units = set(list_units)
print(set_units)

# IsApproximate IsCount IsRange IsList IsMean IsMedian IsMeanHasSD IsMeanHasTolerance IsRangeHasTolerance HasTolerance                 

revlist = sorted(list(set_units), reverse = True, key = len) # List of units
# print(revlist)

# we set non alphabet as the word boundary in our regex
listcounts = ['[^a-z]half[^a-z]', '[^a-z]quarter[^a-z]', '[^a-z]one[^a-z]', '[^a-z]two[^a-z]', '[^a-z]three[^a-z]', '[^a-z]four[^a-z]', '[^a-z]five[^a-z]', '[^a-z]six[^a-z]', '[^a-z]seven[^a-z]', '[^a-z]eight[^a-z]', '[^a-z]nine[^a-z]', '[^a-z]ten[^a-z]', 
              '[^a-z]eleven[^a-z]', '[^a-z]twelve[^a-z]', '[^a-z]thirteen[^a-z]','[^a-z]fourteen[^a-z]','[^a-z]fifteen[^a-z]','[^a-z]sixteen[^a-z]','[^a-z]seventeen[^a-z]','[^a-z]eighteen[^a-z]','[^a-z]nineteen[^a-z]','[^a-z]twenty[^a-z]',
              '[^a-z]thirty[^a-z]','[^a-z]forty[^a-z]','[^a-z]fifty[^a-z]','[^a-z]sixty[^a-z]','[^a-z]seventy[^a-z]','[^a-z]eighty[^a-z]','[^a-z]ninety[^a-z]','[^a-z]hundred[^a-z]',
              '[^a-z]thousand[^a-z]','[^a-z]million[^a-z]','[^a-z]billion[^a-z]','[^a-z]trillion[^a-z]']

listcounts2 = []
listcounts3 = []
listcounts4 = []
for item in listcounts:
    temp = "^"+item[6:]
    listcounts2.append(temp)

# print(listcounts2)

for item in listcounts:
    temp = item[0:len(item)-6] + "$"
    listcounts3.append(temp)

# print(listcounts3)

for item in listcounts:
    temp = "^"+item[6:len(item)-6]+"$"
    listcounts4.append(temp)

# print(listcounts4)

listcounts.extend(listcounts2)
listcounts.extend(listcounts3)
listcounts.extend(listcounts4)
# print(listcounts)

listapproximate = ['∼','~', 'about', 'around', 'close to', 'the order of','approximately', 'nominally', 'near', 'roughly', 'almost', 'approximation',
'≈', 'circa']

listmean = ['average', 'mean']

listmedian = ['median']

listrange = ['to', 'from', 'below', 'beyond', 'above', 'between', 'up to', '<', '>', 'upper',
'greater', 'lesser', 'bigger', 'smaller', 'more than', 'less than', '≥', '≤', 'within',
'throughout', 'at least', 'or more', 'or less', 'past', 'higher', 'almost', 'high', 'before', 'after',
'over', 'under', 'range', 'ranging', 'top', 'at most', 'down to', '⩽', '⩾', '≳', 'as much as','±']

listhyphen = ['−', '-']
    
listlist = ['or', 'and']



'''
We set any non alphabet Character as the word boundary.
Since scientific document can have non-unicode

''' 


def findmodifier(sent, start_off, end_off):
    Numpresent = False
    IsUnit = False
    IsApproximate = False
    IsCount = True
    IsRange = False
    IsList = False 
    Mean = False
    IsMean = False
    IsMedian = False
    Tolerance = False
    Range = False
    IsMeanHasSD = False
    IsMeanHasTolerance = False
    IsRangeHasTolerance = False
    HasTolerance = False
    Unit = ""
    
    # We find if plus-minus exists
    
    
    t_sent = sent[start_off:end_off]
    print("Span is: ", t_sent)
    
    if re.search("[^a-z][0-9]|^[0-9]", t_sent.lower()) is not None:
        Numpresent = True
    else:
        for item in listcounts:
            if re.search(item, t_sent.lower()) is not None:
                Numpresent = True
    
    if Numpresent is False:
        IsCount = False
        return ["null", IsApproximate, IsCount, IsRange, IsList, IsMean, IsMedian, IsMeanHasSD, IsMeanHasTolerance, IsRangeHasTolerance, HasTolerance]
        
    for unit in revlist:
        if re.search("[^a-zA-Z]"+unit+"[^a-zA-Z]"+'|'+"[^a-zA-Z]"+unit+"$", sent[start_off:end_off]) is not None:
            Unit+=unit
            IsUnit = True
            IsCount = False
            break

    if '±' in t_sent:
        IsCount = False
        Tolerance = True

    for item in listmean:
        tmp = re.search("[^a-z]"+item+"[^a-z]"+'|'+"[^a-z]"+item+"$" +'|'+"^"+item+"[^a-z]", sent[start_off:end_off].lower())
        if tmp is not None:
            Mean = True


    if re.search("[^a-z]"+"sd"+"[^a-z]"+"|"+"[^a-z]"+"sd"+"$", sent[start_off-20:end_off].lower()) is True:
        IsMeanHasSD = True

    for item in listrange:
        tmp = re.search("[^a-z]"+item+"[^a-z]"+'|'+"[^a-z]"+item+"$" +'|'+"^"+item+"[^a-z]", t_sent.lower())# sent[max(0,start_off-10):end_off].lower())
        if tmp is not None:
            Range = True

    tmp = re.search("[−|-]", t_sent)
    if tmp is not None:
        tmp1 = re.search("[0-9]+", sent[max(0,tmp.start()+start_off-10):tmp.start()+start_off])
        if tmp1 is not None:
            Range = True
    
    if re.search('[^a-z]'+'median'+'[^a-z]'+'|'+'^'+'median'+'[^a-z]'+'|'+'[^a-z]'+'median'+'$', t_sent.lower()):
        IsMedian = True
    
    tmp = re.search("[^a-z]and[^a-z]|[^a-z]or[^a-z]", t_sent.lower())
    if tmp is not None:
        tmp1 = re.search("[0-9]+", sent[start_off+tmp.end():end_off])
        tmp2 = re.search("[0-9]+", sent[max(0,start_off-1):(tmp.start()+start_off)])
        if tmp1 is not None and tmp2 is not None:
            if Range is not True:
                IsList = True
            IsCount = False
        
    
    for item in listapproximate:
        tmp = re.search("[^a-z]"+item+"[^a-z]"+'|'+"[^a-z]"+item+"$" +'|'+"^"+item+"[^a-z]", sent[max(0,start_off-30):end_off].lower())
        if tmp is not None:
            IsApproximate = True
        
    if Range and not Tolerance:
        IsRange = True
    if IsMean and Tolerance and not IsMeanHasSD:
        IsMeanHasTolerance = True
    if Tolerance and Range:
        IsRangeHasTolerance = True
    if Tolerance and not IsMeanHasSD and not IsMeanHasTolerance and not IsRangeHasTolerance:
        HasTolerance = True
        
    if IsUnit is True:
        return [Unit, IsApproximate, IsCount, IsRange, IsList, IsMean, IsMedian, IsMeanHasSD, IsMeanHasTolerance, IsRangeHasTolerance, HasTolerance]
    else:
        return ["null", IsApproximate, IsCount, IsRange, IsList, IsMean, IsMedian, IsMeanHasSD, IsMeanHasTolerance, IsRangeHasTolerance, HasTolerance]


def convert_output_to_dict(preds):
    label_dict = {}
    if preds[0] != "null":
        label_dict['unit'] = preds[0]
    
    mods = []

    mods_order_ = ["IsApproximate", "IsCount", "IsRange", "IsList", "IsMean", "IsMedian",
                    "IsMeanHasSD", "IsMeanHasTolerance", "IsRangeHasTolerance","HasTolerance"
                ]

    for i, bool_value in enumerate(preds):
        if i == 0:
            pass
        else:
            if bool_value:
                mods.append(mods_order_[i-1])

    if len(mods) > 0:
        label_dict["mods"] = mods

    if len(label_dict) == 0:
        return ""

    return str(label_dict)

from params import params
import json

def label_and_dump(text_path, quant_path, save_path, trial_or_train=False):
    all_text = {f[:-4]: open(text_path + f, encoding="utf8").read() for f in os.listdir(text_path)}


    if trial_or_train:
        trim_tsv = lambda x: x[:-4] if x[-4:] == ".tsv" else x
        quant_labels = {trim_tsv(k):v for k, v in json.load(open(quant_path)).items()}
    else:
        quant_labels = json.load(open(quant_path))

    '''
    Quant_labels:
    Dict{
            Doc1_id: {sent1_offset: [list of quant offsets for sent1],
                    sent2_offset: [list of quant offsets for sent2],
                        ....
                    }
            Doc2_id ....
            ....
    }


    Output:
    Dict{
            Doc1_id: [[docId, annotSet, annotType, startOffset, endOffset, annotId, text, other, sentence_start, sentence_end]
                        repeat same for second quant identified
                        ....
                    ]
            Doc2_id ....
            ....
    }
    '''
    assert set(sorted(all_text.keys())) == set(sorted(quant_labels.keys())), (quant_labels.keys(), "   |||||    ", all_text.keys())

    task2_output = {}
    for docid in quant_labels.keys():
        doc_txt = all_text[docid]
        ann_id = 1
        this_doc_labels = []
        this_sents_offs = sorted([int(x) for x in list(quant_labels[docid].keys())])
        for i, sent_offset in enumerate(this_sents_offs):
            this_sentence_start = sent_offset
            this_sentence_end = this_sents_offs[i+1] if (i +1) < len(this_sents_offs) else len(doc_txt)
            sent = doc_txt[this_sentence_start:this_sentence_end]

            sent_offset_str = str(sent_offset)
            single_sent_quant_offs = quant_labels[docid][sent_offset_str]

            for single_quant_offs in single_sent_quant_offs:
                others_val = convert_output_to_dict(findmodifier(sent, single_quant_offs[0], single_quant_offs[1]))

                this_offset = (single_quant_offs[0] + sent_offset, single_quant_offs[1] + sent_offset)
                this_doc_labels.append([docid, ann_id, "Quantity",
                                        this_offset[0], this_offset[1], ann_id,
                                        doc_txt[this_offset[0]:this_offset[1]],
                                        others_val, this_sentence_start, this_sentence_end
                                        ])
                ann_id += 1

        task2_output[docid] = this_doc_labels

    json.dump(task2_output, open(save_path, 'w+'), indent=2)

basepath = "/home/bt1/17CS10029/measeval/MeasEval/data/"

train_doc_path = basepath + "train/text/"
trial_doc_path = basepath + "trial/txt/"
test_doc_path = basepath + "eval/text/"
bert_type = (params.quant_path[:-1]).split('/')[-1]
print("Bert type:", bert_type)
folder_name = "/home/bt1/17CS10029/measeval/task2/output/" + bert_type.replace('/', '_') #datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
if os.path.isdir(folder_name):
    os.system("rm -rf " + folder_name)
os.mkdir(folder_name)

print("\n\n\n====================================================")
print("====================== Train =======================")
print("====================================================\n\n\n")


label_and_dump(train_doc_path, params.quant_path + "train_spans.json",
                os.path.join(folder_name, "train_labels.json"), True
            )

print("\n\n\n====================================================")
print("======================= Trial ======================")
print("====================================================\n\n\n")

label_and_dump(trial_doc_path, params.quant_path + "trial_spans.json",
                os.path.join(folder_name, "trial_labels.json"), True
            )

print("\n\n\n====================================================")
print("======================= Test =======================")
print("====================================================\n\n\n")

label_and_dump(test_doc_path, params.quant_path + "test_spans.json",
                os.path.join(folder_name + "/test_labels.json"), False
            )
