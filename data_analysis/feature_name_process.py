# coding:utf-8


data = open("./feature.conf","r").readlines()

result = []
print(len(data))
for line in data:
    tmp = line.split(":")
    tmp = tmp[1].split(",")
    feat = tmp[0].strip()
    if(len(feat)<=6):  continue
    if ("waimai" in feat) or ("WAIMAI" in feat): continue
    result.append(feat)
print(len(result))
output = open("./feature_output.conf",'w')

count = 1
for out in result:
    output.write(out+",")
    if(count%8==0): output.write("\n")
    count = count+1

output.flush()
output.close()

