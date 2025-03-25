import os
import json
processed_lines = []
with open('sp_bpe_vietnamese.vocab', 'r',encoding="utf-8") as f:
    lines = f.readlines()
    
    for line in lines:
        line=line.split('\t')
        line[-1]=abs(int(line[-1].split('\n')[0]))
        line[0]=line[0].replace('▁','')
        processed_lines.append(line)
    
processed_lines=[d[0] for d in processed_lines]

processed_lines=list(set(processed_lines))
processed_lines.remove("<pad>")
processed_lines.remove("<s>")
processed_lines.remove("</s>")
processed_lines.remove( "<unk>")
processed_lines.remove( "")

# print(processed_lines)
p=[]
for d in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ':
    if(d not in processed_lines):
        p.append(d)
processed_lines.extend(p)
processed_lines.insert(0,"|")
processed_lines.insert(0,"<unk>")
processed_lines.insert(0,"</s>")
processed_lines.insert(0, "<s>")
processed_lines.insert(0,"<pad>")
vocab_json = {}
for i in range(len(processed_lines)):
    vocab_json[processed_lines[i]] = i
with open('vocab.json', 'w',encoding='utf-8') as outfile:
    json.dump(vocab_json, outfile,ensure_ascii=False)