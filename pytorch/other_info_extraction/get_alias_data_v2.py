#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
import pandas as pd

data_path = "C:\\Users\Herman Yang\Downloads\\alias_res.csv"

data_path_v2 = "alias_res.csv"

df = pd.read_csv(data_path, encoding="utf-8")

df2 = pd.read_csv(data_path_v2)
# print(df2.head(5))

train_data = []
for iv, row in df[:1000].iterrows():
    print(iv)
    print(row["sentence"])

    # print(row["ans"])
    offset_d = dict()
    ii = 0

    if iv not in [11, 28, 36, 38, 50, 59, 60, 63, 68, 70, 80, 85, 91,
                  93, 95, 101, 109, 190, 120, 122, 128, 132, 147, 148,
        156, 159, 161, 164, 185, 187, 188, 189, 191, 192, 193, 194,204, 210,
                  217,224, 238, 249, 255, 259, 261, 262, 266, 267, 288, 294,297,298,303,310,320,324,325,328,334,
                    343, 351, 355, 356, 358, 359,365, 367, 371, 383, 385, 387, 388, 391,397,399,403,
        404, 407, 409, 416, 421,431,434,435,445,450,453,460,467, 470,482,486,490,491,497,
        502,512,527,544,545,555,567,571,574,579,595,598,601,611,613,617,620,622,623,624,633,634,
                  619, 641,656,663,670,671,672,675,679,706,714,736,739,741,765,777,778,789,800,
        808,813,815,816,819,852,859,861,865,866,869,873,876,884,904,906,907,916,917,924,926,946,
        948,962,976,979,
                  989, 990, 997] and row["TRUE"] == 1:
        alias_info = json.loads(df2.iloc[iv]["ans_idx"])
        train_data.append({
            "sentence": row["sentence"],
            "alias": [{"name": item["name"],
                       "alias": item["alias"],
                       "name_idx": [item["name_idx"][0], item["name_idx"][1]+1],
                       "alias_idx": [item["alias_idx"][0], item["alias_idx"][1]+1]} for item in alias_info]
        })
        # print(df2.iloc[iv]["ans_idx"])
        # short_sentence = []
        # for i, s in enumerate(row["sentence"]):
        #     if s in [" "]:
        #         continue
        #     offset_d[ii] = i
        #     ii += 1
        #     short_sentence.append(s)
        # if isinstance(row["ans"], float):
        #     continue
        # # print(row[""])
        # ans_list = [item.split("->") for item in row["ans"].split(":")]
        # print(iv, ans_list)
        # for k, v in ans_list:
        #     print(re.findall(k, "".join(short_sentence)))
        #     print(re.findall(v, "".join(short_sentence)))
    else:
        ans = []
        if iv == 9:
            ans = [{"name": "Sisram Medical Ltd", "alias": "Sisram", "name_re": [0], "alias_re": [1]},
                   {"name": "复锐医疗科技有限公司", "alias": "复锐医疗科技", "name_re": [0], "alias_re": [1]}]
        elif iv == 10:
            ans = [{"name": "Magnificent View Investments Limited", "alias": "Magnificent View", "name_re": [0], "alias_re": [1]},
                   {"name": "上海复星医药（集团）股份有限公司", "alias": "复星医药", "name_re": [0], "alias_re": [1]}]
        elif iv == 12:
            ans = [{"name": "上海复星健康产业控股有限公司", "alias": "复星健控", "name_re": [0],
                    "alias_re": [0]}]
        elif iv == 15:
            ans = [{"name": "复星实业（香港）有限公司", "alias": "复星实业", "name_re": [0],
                    "alias_re": [1]}]
        elif iv == 16:
            ans = [{"name": "海囤（上海）国际贸易有限公司", "alias": "海囤国际", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "上海亲苗科技有限公司", "alias": "亲苗科技", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 18:
            ans = [{"name": "Gland Pharma Limited", "alias": "Gland Pharma", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 20:
            ans = [{"name": "重庆药友制药有限责任公司", "alias": "重庆药友", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "重庆医药（集团）股份有限公司", "alias": "重庆医股", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 37:
            ans = [{"name": "彩生活服务集团有限公司", "alias": "彩生活", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 42:
            ans = [{"name": "福建省现代服务业产业发展基金", "alias": "闽服基金", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "义乌中国小商品城金融控股有限公司", "alias": "商城金控", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 43:
            ans = [{"name": "中电科基金管理有限公司", "alias": "中电基金", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "中电电子信息产业投资基金", "alias": "中电产业基金", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "杭州立思辰安科科技有限公司", "alias": "立思辰安科", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 48:
            ans = [{"name": "内蒙古蒙牛乳业集团股份有限公司", "alias": "蒙牛", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "雅士利国际集团有限公司", "alias": "雅士利", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 51:
            ans = [{"name": "中信资本控股有限公司", "alias": "中信资本", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "麦克英孚控股有限公司", "alias": "麦克英孚", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 53:
            ans = [{"name": "元化智能科技（深圳）有限公司", "alias": "元化智能", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "深圳市创新集团有限公司", "alias": "深创投", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 57:
            ans = [{"name": "欧派家居集团股份有限公司", "alias": "欧派家居", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 89:
            ans = [{"name": "百事食品（中国）有限公司", "alias": "百事食品", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 95:
            ans = [{"name": "Kohlberg Kravis Roberts & Co. L.P.", "alias": "KKR", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 99:
            ans = [{"name": "苏州巨细信息科技", "alias": "巨细", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "广州飞舟信息科技", "alias": "织联网", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 103:
            ans = [{"name": "山东步长制药股份有限公司", "alias": "步长制药", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "步长（香港）控股有限公司", "alias": "步长（香港）", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "西藏丹红企业管理有限公司", "alias": "西藏丹红", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "西藏瑞兴投资咨询有限公司", "alias": "西藏瑞兴", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "西藏广发投资咨询有限公司", "alias": "西藏广发", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "西藏华联商务信息咨询有限公司", "alias": "西藏华联", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 105:
            ans = [{"name": "南京三胞医疗管理有限公司", "alias": "南京三胞医疗", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "徐州三胞医疗管理有限公司", "alias": "徐州三胞医疗", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 116:
            ans = [{"name": "大参林医药集团股份有限公司", "alias": "大参林", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "保定市盛世华兴医药连锁有限公司", "alias": "保定盛世华兴", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 121:
            ans = [{"name": "中国国际金融股份有限公司", "alias": "中金公司", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "安克创新科技股份有限公司", "alias": "安克创新", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 125:
            ans = [{"name": "北京宽街博华投资中心（有限合伙）", "alias": "宽街博华", "name_re": [0],
                    "alias_re": [1]},
                   {"name": "Goldman Sachs Shandong Retail Investment S.à r.l.", "alias": "GS Shandong", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "利群商业集团股份有限公司", "alias": "利群股份", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 141:
            ans = [{"name": "泽星投资有限公司", "alias": "泽星投资", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 149:
            ans = [{"name": "招商局创新投资管理有限责任公司", "alias": "招商创投", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "深圳市招商局创新投资基金中心（有限合伙）", "alias": "招商创新基金", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 150:
            ans = [{"name": "招商局创新投资管理有限责任公司", "alias": "招商创投", "name_re": [0],
                    "alias_re": [0]},
                   {"name": "深圳市招商局创新投资基金中心（有限合伙）", "alias": "招商创新基金", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 164:
            ans = [
                   {"name": "Direct ChassisLink Inc.", "alias": "DCLI", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 167:
            ans = [
                   {"name": "METiS Pharmaceuticals", "alias": "METiS", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 169:
            ans = [
                   {"name": "京东方科技集团股份有限公司", "alias": "京东方", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "北京燕东微电子股份有限公司", "alias": "燕东微", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 177:
            ans = [
                   {"name": "浪莎控股集团有限公司", "alias": "浪莎集团", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 181:
            ans = [
                   {"name": "康恩贝集团有限公司", "alias": "康恩贝集团", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "浙江博康医药投资有限公司", "alias": "博康医药", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 195:
            ans = [
                   {"name": "National Geographic of Azeroth", "alias": "NGA", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 200:
            ans = [
                   {"name": "家家悦控股集团股份有限公司", "alias": "家家悦控股", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 211:
            ans = [
                   {"name": "青岛双星股份有限公司", "alias": "青岛双星", "name_re": [0],
                    "alias_re": [2]}
                   ]

        elif iv == 218:
            ans = [
                   {"name": "北京三夫户外（002780）用品股份有限公司", "alias": "三夫户外", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "北京乐恩嘉业体育发展有限公司", "alias": "乐嘉体育", "name_re": [0],
                     "alias_re": [0]}
                   ]
        elif iv == 220:
            ans = [
                   {"name": "AcFun弹幕视频网", "alias": "A站", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 223:
            ans = [
                   {"name": "北京海科融通支付服务股份有限公司", "alias": "海科融通", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 235:
            ans = [
                   {"name": "虫妈邻里团", "alias": "虫妈", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 249:
            ans = [
                   {"name": "厦门元初食品（需求面积:100-200平方米）股份有限公司", "alias": "元初食品", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 277:
            ans = [
                   {"name": "广州白云电器设备股份有限公司", "alias": "白云电器", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 291:
            ans = [
                   {"name": "康恩贝集团有限公司", "alias": "康恩贝集团", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 297:
            ans = [
                   {"name": "云网万店科技有限公司", "alias": "云网万店", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 298:
            ans = [
                   {"name": "Exegenesis Bio Inc", "alias": "Exegenesis", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 308:
            ans = [
                   {"name": "中山市四维纺织科技有限公司", "alias": "四维科技", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 316:
            ans = [
                   {"name": "宁波中百股份有限公司", "alias": "宁波中百", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "宁波鹏渤投资有限公司", "alias": "鹏渤投资", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 328:
            ans = [
                   {"name": "深国际控股（深圳）有限公司", "alias": "深国际控股", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "深圳市鲲鹏股权投资管理有限公司", "alias": "鲲鹏资本", "name_re": [0],
                    "alias_re": [0]},
                   ]
        elif iv == 337:
            ans = [
                   {"name": "重庆百货大楼股份有限公司", "alias": "重庆百货", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "重庆商社（集团）有限公司", "alias": "商社集团", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "物美科技集团有限公司", "alias": "物美集团", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "步步高投资集团股份有限公司", "alias": "步步高集团", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 338:
            ans = [
                   {"name": "重庆市国有资产监督管理委员会", "alias": "重庆市国资委", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "天津滨海新区物美津融商贸有限公司", "alias": "天津物美", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "深圳步步高智慧零售有限公司", "alias": "步步高零售", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 344:
            ans = [
                   {"name": "大连大杨创世股份有限公司", "alias": "原大杨创世", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "上海圆通蛟龙投资发展（集团）有限公司", "alias": "蛟龙集团", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "上海云锋新创股权投资中心（有限合伙）", "alias": "云锋新创", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 347:
            ans = [
                   {"name": "辰星(天津)自动化设备有限公司", "alias": "辰星", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "诺思(天津)微系统有限责任公司", "alias": "诺思", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 355:
            ans = [
                   {"name": "圆通速递股份有限公司", "alias": "圆通速递", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 371:
            ans = [
                   {"name": "浙江省农民合作经济组织联合会", "alias": "农合联", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "浙江省供销合作社联合社", "alias": "供销社", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "阿里巴巴（中国）软件有限公司", "alias": "阿里巴巴", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 381:
            ans = [
                   {"name": "三胞集团有限公司", "alias": "三胞集团", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 396:
            ans = [
                   {"name": "深圳国中创业投资管理有限公司", "alias": "国中创投", "name_re": [0],
                    "alias_re": [0]},
                    {"name": "深圳国中常荣资产管理有限公司", "alias": "国中常荣", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 416:
            ans = [
                   {"name": "深圳歌力思服饰股份有限公司", "alias": "歌力思", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 419:
            ans = [
                   {"name": "重庆百货大楼股份有限公司", "alias": "重庆百货", "name_re": [0],
                    "alias_re": [1]},
                    {"name": "重庆商社（集团）有限公司", "alias": "商社集团", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 424:
            ans = [
                   {"name": "中国恒大新能源汽车集团有限公司", "alias": "恒大汽车", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 427:
            ans = [
                   {"name": "三胞集团有限公司", "alias": "三胞集团", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 433:
            ans = [
                   {"name": "株式会社万代南梦宫游艺", "alias": "万代南梦宫游艺", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 439:
            ans = [
                   {"name": "日初资本", "alias": "日初资本", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 445:
            ans = [
                   {"name": "虫虫chonny", "alias": "虫虫", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 447:
            ans = [
                   {"name": "上海飞科电器股份有限公司", "alias": "飞科电器", "name_re": [1],
                    "alias_re": [2]}
                   ]
        elif iv == 448:
            ans = [
                   {"name": "上海飞科投资有限公司", "alias": "飞科投资", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 458:
            ans = [
                   {"name": "北京昆仑万维科技股份有限公司", "alias": "昆仑万维", "name_re": [0],
                    "alias_re": [1]},
                {"name": "新余昆诺投资管理有限公司", "alias": "新余昆诺", "name_re": [0],
                 "alias_re": [1]},
                {"name": "北京华宇天宏创业投资管理有限公司", "alias": "华宇天宏", "name_re": [0],
                 "alias_re": [1]}
                   ]
        elif iv == 491:
            ans = [
                   {"name": "Louis Vuitton", "alias": "LV", "name_re": [0],
                    "alias_re": [0]}
                   ]
        elif iv == 498:
            ans = [
                   {"name": "浙商金汇信托股份有限公司", "alias": "浙商金汇信托", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 499:
            ans = [
                   {"name": "厦门国际信托有限公司", "alias": "厦门国际信托", "name_re": [0],
                    "alias_re": [1]}
                   ]
        elif iv == 507:
            ans = [
                   {"name": "浙江省农民合作经济组织联合会", "alias": "农合联", "name_re": [0],
                    "alias_re": [0]},
                {"name": "浙江省供销合作社联合社", "alias": "供销社", "name_re": [0],
                 "alias_re": [0]},
                {"name": "阿里巴巴（中国）软件有限公司", "alias": "阿里巴巴", "name_re": [0],
                 "alias_re": [1]},
                   ]
        elif iv == 508:
            ans = [
                   {"name": "浙江省农民合作经济组织联合会", "alias": "农合联", "name_re": [0],
                    "alias_re": [0]},
                {"name": "供销合作社联合社", "alias": "供销社", "name_re": [0],
                 "alias_re": [0]},
                {"name": "阿里巴巴（中国）有限公司", "alias": "阿里巴巴", "name_re": [0],
                 "alias_re": [1]},
                   ]
        elif iv == 515:
            ans = [
                   {"name": "菜鸟沈阳控股有限公司", "alias": "菜鸟沈阳", "name_re": [0],
                    "alias_re": [1]},
                {"name": "香港青鹬投资管理有限公司", "alias": "青鹬投资", "name_re": [0],
                 "alias_re": [1]}
                   ]
        elif iv == 527:
            ans = [
                {"name": "浙江康恩贝制药股份有限公司", "alias": "康恩贝", "name_re": [0],
                 "alias_re": [1]},
                {"name": "康恩贝集团有限公司", "alias": "康恩贝集团公司", "name_re": [0],
                 "alias_re": [1]},
                {"name": "浙江凤登环保股份有限公司", "alias": "凤登公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "浙江英诺珐医药有限公司", "alias": "英诺珐医药", "name_re": [0],
                 "alias_re": [1]},
                {"name": "浙江康恩贝医药销售有限公司", "alias": "销售公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "浙江珍诚医药在线股份有限公司", "alias": "珍诚医药", "name_re": [0],
                 "alias_re": [1]},
                {"name": "浙江康恩贝健康科技有限公司", "alias": "健康科技公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "浙江宝芝林中药科技有限公司", "alias": "宝芝林公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "浙江康恩贝集团医疗保健品有限公司", "alias": "浙保公司", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 534:
            ans = [
                {"name": "福建丰琪投资有限公司", "alias": "丰琪投资", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 541:
            ans = [
                {"name": "苏宁易购（需求面积:1000-8000平方米）集团股份有限公司", "alias": "苏宁易购或公司", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 552:
            ans = [
                {"name": "赛昉科技有限公司", "alias": "赛昉科技", "name_re": [0],
                 "alias_re": [1]},
                {"name": "深圳市国科瑞华三期股权投资基金合伙企业", "alias": "国科瑞华", "name_re": [0],
                 "alias_re": [1]},
                {"name": "中国互联网投资基金", "alias": "中网投", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 576:
            ans = [
                {"name": "甘肃金控基金管理有限公司", "alias": "金控基金", "name_re": [0],
                 "alias_re": [1]},
                {"name": "甘肃金融控股集团有限公司", "alias": "金控集团", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 599:
            ans = [
                {"name": "青岛啤酒股份有限公司", "alias": "青岛啤酒", "name_re": [0],
                 "alias_re": [1]},
                {"name": "华润雪花啤酒有限公司", "alias": "华润啤酒", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 605:
            ans = [
                {"name": "人福医药集团股份公司", "alias": "人福医药", "name_re": [0],
                 "alias_re": [1]},
                {"name": "武汉当代科技产业集团股份有限公司", "alias": "当代科技", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 608:
            ans = [
                {"name": "贵人鸟股份有限公司", "alias": "贵人鸟", "name_re": [0],
                 "alias_re": [1]},
                {"name": "贵人鸟集团（香港）有限公司", "alias": "贵人鸟集团", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 609:
            ans = [
                {"name": "深圳市鲲鹏股权投资管理有限公司", "alias": "鲲鹏资本", "name_re": [0],
                 "alias_re": [0]},
                {"name": "中国光大控股有限公司", "alias": "光大控股", "name_re": [0],
                 "alias_re": [1]},
                {"name": "深圳市光远投资管理合伙企业", "alias": "光远投资", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 609:
            ans = [
                {"name": "广东新希望新农业股权投资基金管理有限公司", "alias": "广东基金公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "金橡树投资控股（天津）有限公司", "alias": "金橡树公司", "name_re": [0],
                 "alias_re": [0]},
                {"name": "广西投资引导基金有限责任公司", "alias": "广西引导基金", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 619:
            ans = [
                {"name": "维多利亚的秘密", "alias": "维密", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 628:
            ans = [
                {"name": "顺捷投资有限公司", "alias": "顺捷投资", "name_re": [0],
                 "alias_re": [1]},
                {"name": "SF Logistics Development GeneralPartner Limited", "alias": "GP公司", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 629:
            ans = [
                {"name": "顺丰物流开发基金", "alias": "基金", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 639:
            ans = [
                {"name": "西藏信志企业管理咨询有限公司", "alias": "西藏信志", "name_re": [0],
                 "alias_re": [1]},
                {"name": "武汉粹粹餐饮管理有限公司", "alias": "武汉粹粹", "name_re": [0],
                 "alias_re": [1]},
                {"name": "湖北台诚食品科技有限公司", "alias": "湖北台诚", "name_re": [0],
                 "alias_re": [1]},
            ]
        elif iv == 643:
            ans = [
                {"name": "Customer Service by Amazon", "alias": "CSBA", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 648:
            ans = [
                {"name": "跨境通宝电子商务股份有限公司", "alias": "跨境通", "name_re": [0],
                 "alias_re": [1]},
                {"name": "四川金舵投资有限责任公司", "alias": "金舵投资", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 664:
            ans = [
                {"name": "深圳市富匙科技有限公司", "alias": "富匙科技", "name_re": [0],
                 "alias_re": [1]},
                {"name": "农银国际控股有限公司", "alias": "农银国际", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 666:
            ans = [
                {"name": "科兴控股生物技术有限公司", "alias": "科兴控股", "name_re": [0],
                 "alias_re": [1]},
                {"name": "北京科兴中维生物技术有限公司", "alias": "科兴中维", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 688:
            ans = [
                {"name": "复星实业（香港）有限公司", "alias": "复星实业", "name_re": [0],
                 "alias_re": [1]},
                {"name": "NF Unicorn Acquisition L.P.", "alias": "NF", "name_re": [0],
                 "alias_re": [1]},
                {"name": "Healthy Harmony Holdings, L.P.", "alias": "HHH", "name_re": [0],
                 "alias_re": [0]},
                {"name": "Healthy Harmony GP, Inc.", "alias": "Healthy Harmony GP", "name_re": [0],
                 "alias_re": [1]},
            ]
        elif iv == 700:
            ans = [
                {"name": "中国证券监督管理委员会", "alias": "中国证监会", "name_re": [0],
                 "alias_re": [0]},
                {"name": "上海证券交易所", "alias": "上交所", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 723:
            ans = [
                {"name": "广州快决测信息科技有限公司", "alias": "快决测", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 726:
            ans = [
                {"name": "哈药集团有限公司", "alias": "哈药集团", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 729:
            ans = [
                {"name": "深圳瑞享源基金管理有限公司", "alias": "瑞享源基金", "name_re": [0],
                 "alias_re": [1]},
                {"name": "广州越秀产业投资基金管理股份有限公司", "alias": "越秀产业基金", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 739:
            ans = [
                {"name": "安徽国药医疗科技有限公司", "alias": "国医科技", "name_re": [0],
                 "alias_re": [0]},
                {"name": "吉富创业投资股份有限公司", "alias": "吉富创投", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 741:
            ans = [
                {"name": "深圳茂业商厦有限公司", "alias": "茂业商厦", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 760:
            ans = [
                {"name": "创响生物", "alias": "创响", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 800:
            ans = [
                {"name": "南京盈鹏蕙康医疗产业投资合伙企业（有限合伙）", "alias": "盈鹏蕙康", "name_re": [0],
                 "alias_re": [1]},
                {"name": "南京盈鹏资产管理有限公司", "alias": "盈鹏资产", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 845:
            ans = [
                {"name": "综合性血浆制品企业泰邦生物集团", "alias": "泰邦生物", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 849:
            ans = [
                {"name": "福海国盛（天津）股权投资合伙企业（有限合伙）", "alias": "福海国盛", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 853:
            ans = [
                {"name": "利群集团股份有限公司", "alias": "利群集团", "name_re": [0],
                 "alias_re": [1]},
                {"name": "青岛钧泰基金投资有限公司", "alias": "钧泰投资", "name_re": [0],
                 "alias_re": [0]},
                {"name": "青岛利群投资有限公司", "alias": "利群投资", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 859:
            ans = [
                {"name": "Progressive Web Apps", "alias": "PWAs", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 889:
            ans = [
                {"name": "毒App", "alias": "毒", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 901:
            ans = [
                {"name": "重庆百货大楼股份有限公司", "alias": "重庆百货", "name_re": [0],
                 "alias_re": [1]},
                {"name": "重庆商社（集团）有限公司", "alias": "商社集团", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 902:
            ans = [
                {"name": "重庆市国有资产监督管理委员会", "alias": "重庆市国资委", "name_re": [0],
                 "alias_re": [0]},
                {"name": "物美科技集团有限公司", "alias": "物美集团", "name_re": [0],
                 "alias_re": [0]},
                {"name": "天津滨海新区物美津融商贸有限公司", "alias": "天津物美", "name_re": [0],
                 "alias_re": [0]},
                {"name": "步步高投资集团股份有限公司", "alias": "步步高集团", "name_re": [0],
                 "alias_re": [0]},
                {"name": "深圳步步高智慧零售有限公司", "alias": "步步高零售", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 913:
            ans = [
                {"name": "康恩贝集团有限公司", "alias": "康恩贝集团", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 924:
            ans = [
                {"name": "青岛金王应用化学股份有限公司", "alias": "金王", "name_re": [0],
                 "alias_re": [1]},
                {"name": "杭州悠可化妆品有限公司", "alias": "杭州悠可", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 962:
            ans = [
                {"name": "宁波梅山保税港区顾家投资管理有限公司", "alias": "梅山顾家投资", "name_re": [0],
                 "alias_re": [0]}
            ]
        elif iv == 974:
            ans = [
                {"name": "江苏恒瑞医药股份有限公司", "alias": "江苏恒瑞", "name_re": [0],
                 "alias_re": [1]},
                {"name": "上海恒瑞医药有限公司", "alias": "上海恒瑞", "name_re": [0],
                 "alias_re": [1]},
                {"name": "瑞利迪（上海）生物医药有限公司", "alias": "瑞利迪", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 983:
            ans = [
                {"name": "江苏 泰治科技 股份有限公司", "alias": "泰治科技", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 990:
            ans = [
                {"name": "广东省小分子新药创新中心", "alias": "小分子中心", "name_re": [0],
                 "alias_re": [0]},
                {"name": "深圳市新樾生物科技有限公司", "alias": "新樾生物", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 995:
            ans = [
                {"name": "上海实业（集团）有限公司", "alias": "上实集团", "name_re": [0],
                 "alias_re": [1]},
                {"name": "上实国际投资有限公司", "alias": "上实国际", "name_re": [0],
                 "alias_re": [1]}
            ]
        elif iv == 999:
            ans = [
                {"name": "深圳市客一客信息科技有限公司", "alias": "客一客", "name_re": [0],
                 "alias_re": [1]},
                {"name": "中航联创科技有限公司", "alias": "中航联创", "name_re": [0],
                 "alias_re": [1]}
            ]
        ans_new = []
        for item in ans:
            name_location = [al.span(0) for al in re.finditer(item["name"].replace("(", "\(").replace(")", "\)"), row["sentence"])]
            alias_location = [al.span(0) for al in re.finditer(item["alias"].replace("(", "\(").replace(")", "\)"), row["sentence"])]

            name_idx = name_location[item["name_re"][0]]
            print(row["sentence"][name_idx[0]:name_idx[1]])
            it = {"name": item["name"],
                            "alias": item["alias"],
                            "name_idx": name_location[item["name_re"][0]],
                            "alias_idx": alias_location[item["alias_re"][0]]}
            assert it["name_idx"][1] < it["alias_idx"][0]
            # print(it)
            ans_new.append(it)
        train_data.append({
            "sentence": row["sentence"],
            "alias": ans_new
        })



with open("D:\data\self-data\\alias_train_v2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_data))


