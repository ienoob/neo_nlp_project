#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
from nlp_applications.data_loader import load_json_line_data


train_path = "D:\data\篇章级事件抽取\\duee_fin_train.json\\duee_fin_train.json"
dev_path = "D:\data\篇章级事件抽取\\duee_fin_dev.json\\duee_fin_dev.json"
new_data_path = "D:\data\\tj_event\\trz_events.csv"

train_data = load_json_line_data(train_path)
dev_data = load_json_line_data(dev_path)
role_set = set()
trigger_set = set()

def f_data(input_data):
    documents = []
    # sub_trigger_set = set()
    for i, sub_train_data in enumerate(input_data):
        text = sub_train_data["text"]
        title = sub_train_data["title"]
        doc_id = sub_train_data["id"]

        event_list = []
        for sub_event in sub_train_data.get("event_list", []):
            if sub_event["event_type"] == "企业融资":
                event_list.append(sub_event)
                trigger_set.add(sub_event["trigger"])

        if doc_id == "08a973d986c7a908fbd5978d6d1e3101":
            event = event_list[0]
            event["arguments"] = [event["arguments"][0], event["arguments"][1], event["arguments"][2], event["arguments"][3], event["arguments"][5]]
        elif doc_id == "873bcc469a2645217190caf41d6724e9":
            # for event in event_list:
            #     print(event)
            event = event_list[0]
            if event["arguments"][2]["role"] == "融资金额" and event["arguments"][2]["argument"] == "D2":
                event["arguments"][2]["role"] = "融资轮次"
        elif doc_id == "0d0244f296b1ad39ba61d9ea83f16652":
            for event in event_list:
                if event["arguments"][1]["role"] == "融资金额" and event["arguments"][1]["argument"] == "滨江集团":
                    event["arguments"][1]["role"] = "被投资方"
        elif doc_id == "6763552149b294fb9a9d7a7b7d194b2b":
            pass
        elif doc_id == "67c1d0bd53896bab5883033316dafff6":
            pass
        elif doc_id == "7a94f60cc3ab446d5978253c8f1c5107":
            pass
        elif doc_id == "16b4d6acad53c77737826d9ed0f84b3b":
            pass
        elif doc_id == "7ef64759c646aa99ae059da69ea393f4":
            pass
        elif doc_id == "c43a5c10d1e67d66fe58c29481faabdf":
            for event in event_list:
                if event["arguments"][2]["role"] == "被投资方" and event["arguments"][2]["argument"] == "pre-A":
                    event["arguments"][2]["role"] = "融资轮次"
            #     if event["arguments"][]
        elif doc_id == "5d991612b4d838d0d00fdcb3e7cf76f5":
            for event in event_list:
                if event["arguments"][3]["role"] == "被投资方" and event["arguments"][3]["argument"] == "中迪资管":
                    event["arguments"] = [event["arguments"][0], event["arguments"][1], event["arguments"][2], event["arguments"][4]]
        elif doc_id == "4e6267d8cab035fd98ec982eea87456c":
            for event in event_list:
                if event["arguments"][0]["role"] == "事件时间" and event["arguments"][0]["argument"] == "8月10日":
                    event["arguments"] = event["arguments"][1:]
        elif doc_id == "d8e7f9df00f4660a08b6836df0a8db02":
            for event in event_list:
                event["arguments"].append({'role': '领投方', 'argument': '知春资本'})

        elif doc_id in ["d27baf35b60ae366eb599202119d2e0b", "3360200be21c9c4cc63e011d42f2da13"]:
            # 不属于投融资
            continue
        elif doc_id == "71a7fedbb252e62c1a3921934b461dc8":
            for event in event_list:
                if event["arguments"][3]["role"] == "被投资方" and event["arguments"][3]["argument"] == "美团龙珠资本":
                    event["arguments"][3]["role"] = "投资方"
        elif doc_id == "fbe032ab57538a286e47a07c6ede52a0":
            for event in event_list:
                if event["arguments"][4]["role"] == "领投方" and event["arguments"][4]["argument"] == "Tsingyuan Ventures":
                    event["arguments"][4]["argument"] = "清源资本Tsingyuan Ventures"
        elif doc_id == "b50d5ea8cc299912b980a82b11b6808e":
            for event in event_list:
                if event["arguments"][2]["role"] == "融资轮次" and event["arguments"][2]["argument"] == "A轮":
                    event["arguments"][2]["argument"] = "A"
        elif doc_id == "36ac5ccca8e1b41fcb2b2c902675295b":
            for event in event_list:
                if event["arguments"][0]["role"] == "披露时间" and event["arguments"][0]["argument"] == "10月23日":
                    event["arguments"].append({'role': '领投方', 'argument': '辉立资本'})
        elif doc_id == "6e304042818d76f9cf11d569c5f30af1":
            for event in event_list:
                if event["arguments"][0]["role"] == "披露时间" and event["arguments"][0]["argument"] == "8月16日":
                    event["arguments"].append({'role': '投资方', 'argument': '王兴'})
        elif doc_id == "dfcc2a1868a6518892e54805679d269d":
            for event in event_list:
                if event["arguments"][-1]["role"] == "领投方" and event["arguments"][-1]["argument"] == "中迪资管":
                    event["arguments"].append({'role': '投资方', 'argument': '中迪资管'})
        elif doc_id == "6010316d9462a27577629bdb55f20042":
            for event in event_list:
                if event["arguments"][-2]["role"] == "领投方" and event["arguments"][-2]["argument"] == "掌阅科技":
                    event["arguments"].append({'role': '投资方', 'argument': '掌阅科技'})
        elif doc_id == "dc8cece5bf40f8ee1a17ce953b59bf75":
            for event in event_list:
                if event["arguments"][-1]["role"] == "投资方" and event["arguments"][-1]["argument"] == "InterVes":
                    event["arguments"][-1]["argument"] = "InterVest"
        elif doc_id == "d29524343c2438300b0fac39d782f1ec":
            event_list.append({
                'trigger': '融资',
                'event_type': '企业融资',
                'arguments': [{'role': '被投资方', 'argument': 'MiNA Therapeutics'},
                              {'role': '融资金额', 'argument': '3000万美元'},
                              {'role': '融资轮次', 'argument': 'A'},
                              {'role': '事件时间', 'argument': '今天'}]})
        elif doc_id == "da8f29a5ce27036464fbd06ac3628c8b":
            for event in event_list:
                if event["arguments"][-2]["role"] == "投资方" and event["arguments"][-2]["argument"] == "Himalaya\nCapital":
                    event["arguments"][-2]["argument"] = "Himalaya \nCapital"

        elif doc_id == "2c5b55e9e9a8f2135add6ffbaa0234f3":
            for event in event_list:
                if event["arguments"][-2]["role"] == "融资轮次" and event["arguments"][-2]["argument"] == "B轮":
                    event["arguments"][-2]["argument"] = "B"
        elif doc_id == "3086ef3440902372a0aa9e10df87ebec":
            for event in event_list:
                if event["arguments"][-1]["role"] == "融资轮次" and event["arguments"][-1]["argument"] == "B轮":
                    event["arguments"][-1]["argument"] = "B"
        elif doc_id == "89aa1f32d1da97646bcdde5980c72872":
            for event in event_list:
                if event["arguments"][-2]["role"] == "被投资方" and event["arguments"][-2]["argument"] == "网易云音":
                    event["arguments"][-2]["argument"] = "网易云音乐"
        elif doc_id == "2899d388065682dedcfd9f0acf072b25":
            for event in event_list:
                if event["arguments"][2]["role"] == "融资轮次" and event["arguments"][2]["argument"] == "C轮":
                    event["arguments"][2]["argument"] = "C"
        elif doc_id == "9d6aba07c224f2a32365178d18c13ca2":
            for event in event_list:
                if event["arguments"][0]["role"] == "投资方" and event["arguments"][0]["argument"] == "李星星":
                    event["arguments"].append({'role': '被投资方', 'argument': '北京好车多多科技有限公司'})
        elif doc_id == "89adb5cdde9d6a30cecff2369a363185":
            for event in event_list:
                if event["arguments"][-5]["role"] == "投资方" and event["arguments"][-5]["argument"] == "中国联塑(":
                    event["arguments"][-5]["argument"] = "中国联塑"
        elif doc_id == "0d20305ccd5d6bebee4b0492c10abf3d":
            for event in event_list:
                if event["arguments"][3]["role"] == "融资轮次" and event["arguments"][3]["argument"] == "天使轮":
                    event["arguments"][3]["argument"] = "天使"
        elif doc_id == "8b4d4bb3a17730e1ccfc39b6d191446f":
            for event in event_list:
                if event["arguments"][0]["role"] == "融资金额" and event["arguments"][0]["argument"] == "10亿美元":
                    event["arguments"].append({'role': '被投资方', 'argument': '华大智造'})
        elif doc_id == "d6ec8cdb1f6a164a5de0e1f130e98f1b":
            for event in event_list:
                if event["arguments"][2]["role"] == "融资轮次" and event["arguments"][2]["argument"] == "B轮":
                    event["arguments"][2]["argument"] = "B"
        elif doc_id == "29ac43ffcbab28253ae2fe5aaaeb3044":
            for event in event_list:
                if event["arguments"][-1]["role"] == "被投资方" and event["arguments"][-1]["argument"] == "Aakash":
                    event["arguments"].append({'role': '估值', 'argument': '超过5亿美元'})
        elif doc_id == "d821587e3a4fe87bf3723c865901c6df":
            for event in event_list:
                if event["arguments"][1]["role"] == "被投资方" and event["arguments"][1]["argument"] == "小度科技":
                    event["arguments"].append({'role': '估值', 'argument': '约200亿元'})
        elif doc_id == "bede494f35ba0c06e6799e39faf391be":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "唯医骨科":
                    event["arguments"].append({'role': '财务顾问', 'argument': '华兴资本'})
        elif doc_id == "6c65f6c82e774a701750c0dc3930a253":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "中羽运动网":
                    event["arguments"].append({'role': '投资方', 'argument': '易知出国'})
        elif doc_id == "8870f8a62d58b7ab2099257cb84f137b":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "中科微针（北京）科技有限公司":
                    event["arguments"].append({'role': '融资轮次', 'argument': '首'})
        elif doc_id == "63c44af3bfee2caeb2c23b86fcc3b439":
            for event in event_list:
                if event["arguments"][0]["role"] == "投资方" and event["arguments"][0]["argument"] == "宁沪高速":
                    event["arguments"][0]["role"] = "被投资方"
        elif doc_id == "a37757b498f115049ea10577986512cf":
            for event in event_list:
                if event["arguments"][1]["role"] == "被投资方" and event["arguments"][1]["argument"] == "ETF":
                    event["arguments"][1]["argument"] = "科技龙头ETF"
        elif doc_id == "872e67890fdb4a54e773c087eb4a19ce":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "圆通速递国际":
                    event["arguments"][0]["argument"] = "圆通速递"
        elif doc_id == "dbd77e28251bd2fe1b8e31fa8603af07":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "金融壹账通":
                    event["arguments"].append({'role': '估值', 'argument': '75亿美元'})
        elif doc_id == "cd4d16cf1f24b3da104a87820a0a7720":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "玄合医疗":
                    event["arguments"][0]["argument"] = "玄合医疗科技（上海）有限公司"
        elif doc_id == "ba1c75a2267197675667cceafb1115a9":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "江小白":
                    event["arguments"].append({'role': '估值', 'argument': '超130亿人民币'})
        elif doc_id == "86124517e70120db2fa3dbd7a1048b90":
            print(event_list)
            event_list = [
                {'trigger': '融资', 'event_type': '企业融资', 'arguments': [
                    {'role': '被投资方', 'argument': 'Bakkt'},
                    {'role': '融资轮次', 'argument': 'B'},
                    {'role':  '投资方', 'argument': 'M12'},
                    {'role': '投资方', 'argument': 'PayU'},
                    {'role': '投资方', 'argument': '波士顿咨询集团'},
                    {'role': '投资方', 'argument': 'Goldfinch Partners'},
                    {'role': '投资方', 'argument': 'CMT Digital'},
                    {'role': '投资方', 'argument': 'Pantera Capital'},
                ]}
            ]
        elif doc_id == "8e6c8701c6945a699b66e0988f4c30af":
            for event in event_list:
                if event["arguments"][0]["role"] == "披露时间" and event["arguments"][0]["argument"] == "4月30日":
                    event["arguments"].append({'role': '被投资方', 'argument': 'ZuBlu'})
        elif doc_id == "5107683d1d5915c08a2b05f1e3e3ed93":
            for event in event_list:
                if event["arguments"][-1]["role"] == "被投资方" and event["arguments"][-1]["argument"] == "Molecular":
                    event["arguments"][-1]["argument"] = "Molecular Assemblies"
        elif doc_id == "9a77eb80a163e4eee91e2369c1a0c074":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "Unacademy":
                    event["arguments"] = [{'role': '被投资方', 'argument': 'Unacademy'}, {'role': '融资金额', 'argument': '5000万美元'}, {'role': '融资轮次', 'argument': 'D'}, {'role': '领投方', 'argument': 'Steadview Capital'}, {'role': '领投方', 'argument': 'Sequoia India'}, {'role': '领投方', 'argument': 'Nexus Venture Partners'}, {'role': '领投方', 'argument': 'Blume Ventures'}, {'role': '投资方', 'argument': 'Gaurav Munjal'}, {'role': '投资方', 'argument': 'Roman Saini'},]
        elif doc_id == "bc24e083a65db9ae242d5088986b2cd4":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "Zenius":
                    event["arguments"].append({'role': '融资轮次', 'argument': '首'})
        elif doc_id == "f5eb599df620b5bc46402a2d8086b7d7":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "云从科技":
                    event["arguments"].append({'role': '估值', 'argument': '200亿元'})
        elif doc_id == "ef89d66dee9130201645146f45a41cf3":
            for event in event_list:
                if event["arguments"][1]["role"] == "融资轮次" and event["arguments"][1]["argument"] == "D":
                    event["arguments"].append({'role': '估值', 'argument': '近350亿元'})
        elif doc_id == "efd256dc607f1fb79fc4b4c6fb9bf3bf":
            for event in event_list:
                if event["arguments"][0]["role"] == "被投资方" and event["arguments"][0]["argument"] == "Udemy Inc.":
                    event["arguments"].append({'role': '估值', 'argument': '超过30亿美元'})
        elif doc_id == "bfb8f91282dfe1f8e8112be2a041c4a0":
            for event in event_list:
                event["arguments"].append({'role': '估值', 'argument': '29.3亿美元'})
        elif doc_id == "fa0f130ca029f2d036e8b3a0d9a01d2d":
            for event in event_list:
                if event["arguments"][0]["role"] == "披露时间" and event["arguments"][0]["argument"] == "6月19日":
                    event["arguments"].append({'role': '估值', 'argument': '18至20亿元人民币'})
        elif doc_id == "42a10cd4d68121bdb9d8aa12980466e5":
            for event in event_list:
                if event["arguments"][2]["role"] == "融资金额" and event["arguments"][2]["argument"] == "2亿美元":
                    event["arguments"].append({'role': '估值', 'argument': '112 亿美元'})
        elif doc_id == "6d3705966c793002a4865b8e7c784534":
            for event in event_list:
                if event["arguments"][-1]["role"] == "被投资方" and event["arguments"][-1]["argument"] == "古早娱乐":
                    event["arguments"].append({'role': '融资金额', 'argument': '千万级'})


        if len(event_list):

            if "估值" in text:
                print(text)
            # print(doc_id)
            # print(text)
            # for event in event_list:
            #     r_event = dict()
            #     for r_arg in event["arguments"]:
            #         if r_arg["role"] in r_event:
            #             if r_arg["role"] == "投资方":
            #                 print(event)
            #                 print(text)
            #                 print(doc_id, "title")
            #             # print(r_arg["role"])
            #             role_set.add(r_arg["role"])
            #         else:
            #             r_event[r_arg["role"]] = r_arg["argument"]

            documents.append({"id": doc_id, "text": text, "title": title, "event": event_list})
            print("===========================================================================")
    return documents

def f_data_v2():
    data_path = "D:\data\\tianjin_dataset\\tj_event\data"

train_documents = f_data(train_data)
dev_documents = f_data(dev_data)

print(trigger_set)

documents = train_documents + dev_documents

print(len(documents))
import re
# with open("finance.json", "w") as f:
#     f.write(json.dumps(documents))

