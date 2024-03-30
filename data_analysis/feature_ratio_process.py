# coding:utf-8

import pandas as pd


user_feat = pd.read_csv("user_feat_ratio.txt",delimiter="\\s+",header=None,names=["feat_name","ratio"])
item_feat = pd.read_csv("mix_feat_ratio_v2.txt",delimiter="\\s+",header=None,names=["feat_name","ratio"])

user_feat.sort_values(by=["feat_name","ratio"],ascending=False,inplace=True)
item_feat.sort_values(by=["feat_name","ratio"],ascending=False,inplace=True)
user_feat.to_excel("use_feat_ratio.xls",index=False)
item_feat.to_excel("item_feat_ratio.xls",index=False)

print(user_feat)
