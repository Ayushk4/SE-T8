import json
import os

combined = json.load(open('final_labels.json'))

os.system("rm -rf tsv")
os.mkdir("tsv")

def to_str(vals_list):
    strs = []
    for vl in vals_list[:-1]:
        if type(vl) == str:
            strs.append(vl.encode('utf8'))
        elif type(vl) == int:
            strs.append(str(vl).encode('utf8'))
    
    if type(vals_list[-1]) == dict:
        strs.append(json.dumps(vals_list[-1], ensure_ascii=False).encode('utf8'))
    elif vals_list[-1] == "":
        strs.append("".encode('utf8'))
    else:
        dict_format = json.loads(vals_list[-1].replace("'", '"'))
        dict_format = dict(sorted(dict_format.items(), key = lambda x: x[0]))
        strs.append(json.dumps(dict_format, ensure_ascii=False).encode('utf8'))
    print(strs)
    return strs

for k, v in combined.items():
    print(k)
    lines = [b"docId\tannotSet\tannotType\tstartOffset\tendOffset\tannotId\ttext\tother"]
    for annset in v:
        for kk, vv in annset.items():
            if vv != None:
                lines.append(b"\t".join(to_str(vv[:8])))
    open("tsv/" + k + ".tsv", 'wb').write((b"\n".join(lines)))

