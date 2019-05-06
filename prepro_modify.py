#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random
from nltk.corpus import stopwords
from nltk.corpus import stopwords


#sentenceとquestionで共通するノンストップワードが一つもない場合False
def check_overlap(sentence,question,stop_words):

    for w in question.split():
        if sentence.find(w)!=-1 and w not in stop_words:
            return True
    return False

def answer_find(context_text,answer_start,answer_end):

    #context=sent_tokenize(context_text)
    context=re.split("\.\s|\?\s|\!\s",context_text)
    start_p=0

    for i,sentence in enumerate(context):
        start_p=context_text.find(sentence,start_p)
        end_p=start_p+len(sentence)+1

        if start_p<=answer_start<end_p:
            sentence_start_id=i
        if start_p<answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        #findで処理する
        start_p+=len(sentence)


    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=context_text[sentence_start_id:sentence_end_id+1]
    if sentence_start_id!=sentence_end_id:
        print(context_text)
        print(context)
        print(answer_start,answer_end)
        print(answer_sentence)
        print()


    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#単語が連続して現れている部分は削除する
def overlap_rm(sentence):
    #print(sentence)
    sentence=sentence.split()
    #print(sentence)
    #print()
    rm_index=[]
    for i in range(len(sentence)-1):
        if sentence[i+1]==sentence[i]:
            rm_index.append(i+1)
    new_sentence=[sentence[i] for i in range(len(sentence)) if i not in rm_index]
    return " ".join(new_sentence)


def data_process(input_path,interro_path,modify_path1,modify_path2="",train=False):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(interro_path,"r") as f:
        interro_data=json.load(f)

    modify_data=[]
    with open(modify_path1,"r") as f:
        for line in f:
            modify_data.append(line.rstrip())
    if modify_path2!="":
        with open(modify_path2,"r") as f:
            for line in f:
                modify_data.append(line.rstrip())

    print(len(modify_data))
    use_interro=True

    questions=[]
    answers=[]
    sentences=[]
    interros=[]
    non_interros=[]
    stop_words = stopwords.words('english')
    all_count=0
    modify_count=0

    new_data={"data":[],
                "version":"1.1"}
    for topic in tqdm(data["data"]):
        new_topic={"title":topic["title"],
                    "paragraphs":[]}
        print(new_topic["title"])
        topic=topic["paragraphs"]

        for paragraph in topic:
            new_paragraph={"context":paragraph["context"],
                            "qas":[]}
            context_text=paragraph["context"].lower()

            for qas in paragraph["qas"]:
                sentence_text=interro_data[all_count]["sentence_text"]
                question_text=interro_data[all_count]["question_text"]
                answer_text=interro_data[all_count]["answer_text"]
                interro=interro_data[all_count]["interro"]
                non_interro=interro_data[all_count]["non_interro"]
                all_count+=1

                if True:
                    #テキストとノンストップワードが一つも重複してないものは除去
                    if check_overlap(sentence_text,question_text,stop_words)==False:
                        continue

                if True:
                    #疑問詞がないものは削除
                    if interro=="":
                        continue

                if interro[-1]=="?":
                    interro=interro[:-2]

                if use_interro:
                    sentence_text=" ".join([sentence_text,"<SEP>",interro])

                modify_question=modify_data[modify_count]
                modify_count+=1

                qas["modify_question"]=False
                new_paragraph["qas"].append(qas)

                new_qas=qas.copy()
                new_qas["modify_question"]=True
                new_qas["id"]=new_qas["id"]+"-modify_question"
                new_qas["question"]=modify_question
                new_paragraph["qas"].append(new_qas)

        new_topic["paragraphs"].append(new_paragraph)
    new_data["data"].append(new_topic)


    if use_interro==False:
        setting="modify-normal"
    else:
        setting="modify-interro"

    if train:
        with open("data/squad-train-{}.json".format(setting),"w")as f:
            json.dump(new_data,f,indent=4)
    else:
        with open("data/squad-dev-{}.json".format(setting),"w")as f:
            json.dump(new_data,f,indent=4)



if __name__ == "__main__":
    random.seed(0)

    data_process(input_path="data/squad-dev-v1.1.json",
                interro_path="data/squad-data-dev.json",
                modify_path1="data/squad-pred-val-interro.txt",
                modify_path2="data/squad-pred-test-interro.txt",
                train=False
                )

    data_process(input_path="data/squad-train-v1.1.json",
                interro_path="data/squad-data-train.json",
                modify_path1="data/squad-pred-train-interro.txt",
                train=True
                )
