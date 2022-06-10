#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
from pytorch.event_extraction.bert_utils import split_cha

with open("finance.json", "r") as f:
    datas = f.read()

with open("finance_add.json", "r") as f:
    data_add = f.read()


documents = json.loads(datas)
add_documents = json.loads(data_add)

documents += add_documents
dt = [

   {
        "idx": 1,
        "id": "01274f6fee64538d9af3a42ad0b660dc",
        "cg": [
            {'role': '被投资方', 'argument': '锐思华创', "index": [0, 1]}
        ]
    },
    {
        "idx": 3,
        "id": "dfcc2a1868a6518892e54805679d269d",
        "cg": [
            {'role': '融资金额', 'argument': '亿元', "index": [0, 1]},
            {'role': '投资方', 'argument': '中迪资管', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '中迪资管', "index": [0]},
            {'role': '被投资方', 'argument': '润哲教育', "index": [0, 1, 2]},

        ]
    },
    {
        "idx": 4,
        "id": "e2dd35c94c477d8c6cfd6b7b205ca484",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]}
        ]
    },
    {
        "idx": 6,
        "id": "6010316d9462a27577629bdb55f20042",
        "cg": [
            {'role': '被投资方', 'argument': '书链', "index": [0, 1]},
            {'role': '投资方', 'argument': '掌阅科技', "index": [0, 1]},
            {'role': '领投方', 'argument': '掌阅科技', "index": [0]}
        ]
    },
    {
        "idx": 8,
        "id": "3e469495f7d7197aa582260386b97018",
        "cg": [
            {'role': '被投资方', 'argument': '信唐普华ITIC', "index": [0, 1]}
        ]
    },
    {
        "idx": 9,
        "id": "29ac43ffcbab28253ae2fe5aaaeb3044",
        "cg": [
            {'role': '被投资方', 'argument': 'Aakash', "index": [0, 1, 2, 3, 6, 7]}
        ]
    },
    {
        "idx": 15,
        "id": "c37fa5e14867391345c36189ffdfad90",
        "cg": [
            {'role': '被投资方', 'argument': '心声医疗', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '信文资本', "index": [0, 2]}
        ]
    },
    {
        "idx": 18,
        "id": "b50d5ea8cc299912b980a82b11b6808e",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '被投资方', 'argument': '西柚健康', "index": [0, 1, 3]}
        ]
    },
    {
        "idx": 20,
        "id": "4a96b2d9f6a4edfaa6911ecd1481f0d5",
        "cg": [
            {'role': '被投资方', 'argument': '趣记忆', "index": [0, 1]}
        ]
    },
    {
        "idx": 23,
        "id": "fa4bf7a9d8d7d5e6d3fda6959ce90cd1",
        "cg": [
            {'role': '被投资方', 'argument': '转转', "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]}
        ]
    },
    {
        "idx": 26,
        "id": "0e7ae4aecafec3028752d219367c53d8",
        "cg": [
            {'role': '被投资方', 'argument': '*ST庞大', "index": [10, 11]}
        ]
    },
    {
        "idx": 28,
        "id": "d29524343c2438300b0fac39d782f1ec",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [1, 18]}
        ],
    },
    {
        "idx": 30,
        "id": "652b32b316947599d66f6bacab9ef850",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [2, 3]},
            {'role': '融资轮次', 'argument': 'B', "index": [0]}
        ]
    },
    {
        "idx": 32,
        "id": "ccfe48ea003c754554d9ec1ce6b25a18",
        "cg": [
            {'role': '被投资方', 'argument': '指真生物', "index": [0, 1]},
            {'role': '领投方', 'argument': '盛宇投资', "index": [0, 2]}
        ]
    },
    {
        "idx": 33,
        "id": "2126d7ce07748e56d1013dfd7029da29",
        "cg": [
            {'role': '被投资方', 'argument': '伊对', "index": [0, 1, 2, 3, 4]},
            {'role': '领投方', 'argument': '小米', "index": [1]},
            {'role': '领投方', 'argument': '云九资本', "index": [1]},
        ]
    },
    {
        "idx": 38,
        "id": "4cb8346b8d52f39a53dec7fc18064ae8",
        "cg": [
            {'role': '被投资方', 'argument': '金康新能源', "index": [0, 1, 3, 4, 5, 6]}
        ]
    },
    {
        "idx": 39,
        "id": "117bb17482a3e82b2ebc3de40dbd6b9d",
        "cg": [
            {'role': '被投资方', 'argument': 'Ace＆Tate', "index": [0, 1]}
        ]
    },
    {
        "idx": 40,
        "id": "5b68c0a1c821698e8ca0085974edec78",
        "cg": [
            {'role': '被投资方', 'argument': '老百姓', "index": [0, 1, 2, 6, 7]}
        ]
    },
    {
        "idx": 43,
        "id": "caa1fdafca6042fdcacb200560f0314f",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [1, 3]},
        ]
    },
    {
        "idx": 44,
        "id": "36ac5ccca8e1b41fcb2b2c902675295b",
        "cg": [
            {'role': '被投资方', 'argument': '长桥证券LongBridge', "index": [0, 1]},
            {'role': '领投方', 'argument': '元璟资本', "index": [0, 2]}
        ]
    },
    {
        "idx": 46,
        "id": "6e304042818d76f9cf11d569c5f30af1",
        "cg": [
            {'role': '被投资方', 'argument': '理想汽车', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 47,
        "id": "0607246dfff18fe5fb95bc884482abdd",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0]},
            {'role': '被投资方', 'argument': '波奇宠物', "index": [0, 1, 2, 3, 4, 5]},
        ]
    },
    {
        "idx": 48,
        "id": "59a6a66fc521896e54d0da6e57fb2220",
        "cg": [
            {'role': '被投资方', 'argument': 'BerryMelon', "index": [0, 1, 2, 4]},
        ]
    },
    {
        "idx": 50,
        "id": "59a6a66fc521896e54d0da6e57fb2220",
        "cg": [
            {'role': '被投资方', 'argument': '同程金服', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 51,
        "id": "dac5dd09143700b39f7c28c33eab47d8",
        "cg": [
            {'role': '被投资方', 'argument': '诺辉健康', "index": [9]},
            {'role': '领投方', 'argument': '君联资本', "index": [0]}
        ]
    },
    {
        "idx": 55,
        "id": "11bc0cf5deebbcebcdac911333e9d28a",
        "cg": [
            {'role': '被投资方', 'argument': 'Authing', "index": [0, 1]},
        ]
    },
    {
        "idx": 56,
        "id": "3aac8257e1df92766b4e3b2be2ab2b64",
        "cg": [
            {'role': '被投资方', 'argument': '达观数据', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '深创投', "index": [0, 2]}
        ]
    },
    {
        "idx": 58,
        "id": "16b4d6acad53c77737826d9ed0f84b3b",
        "cg": [
            {'role': '被投资方', 'argument': '恒润拾', "index": [0, 1, 4]},
            {'role': '领投方', 'argument': '清流资本', "index": [0, 2]}
        ]
    },
    {
        "idx": -1,
        "id": "90806c42bf4be78bbdb5819f566483c2",
        "cg": [
            {'role': '融资金额', 'argument': '5亿元', "index": [1, 3]},
        ]
    },
    {
        "idx": 59,
        "id": "a114e61f7340345039f4ca8f4197559d",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 60,
        "id": "064a3d21c5ba05a1d994bc44304a5893",
        "cg": [
            {'role': '被投资方', 'argument': '非同生物', "index": [1, 2]},
        ]
    },
    {
        "idx": 62,
        "id": "6d3705966c793002a4865b8e7c784534",
        "cg": [
            {'role': '被投资方', 'argument': '古早娱乐', "index": [0, 4]},
        ]
    },
    {
        "idx": 63,
        "id": "873bcc469a2645217190caf41d6724e9",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [2, 3]},
            {'role': '被投资方', 'argument': '高思教育', "index": [2, 3, 4]},
        ]
    },
    {
        "idx": 64,
        "id": "20b9c058e0c92d339c75313379ce2df0",
        "cg": [
            {'role': '被投资方', 'argument': 'WeWork China', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 65,
        "id": "3af902d93ad1158450834828c740a612",
        "cg": [
            {'role': '被投资方', 'argument': '蔚来汽车', "index": [0]},
        ]
    },
    {
        "idx": 66,
        "id": "96fc6aefbd535b06f90604428dbc8804",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [2]},
        ]
    },
    {
        "idx": 67,
        "id": "3cf86c79f698a5becabab9e991903c2c",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
        ]
    },
    {
        "idx": 68,
        "id": "0e4efad5111d5bd45eadf64521120dab",
        "cg": [
            {'role': '被投资方', 'argument': '知乎', "index": [0, 1, 2, 3, 7]},
            {'role': '投资方', 'argument': '百度', "index": [0]},
            {'role': '领投方', 'argument': '快手', "index": [0]},
            {'role': '投资方', 'argument': '快手', "index": [0]}
        ]
    },
    {
        "idx": 69,
        "id": "46f454e2752a0dd821c1c485c040aeed",
        "cg": [
            {'role': '被投资方', 'argument': '康得集团', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 73,
        "id": "29f443b711d94f53495dad529697f5e4",
        "cg": [
            {'role': '被投资方', 'argument': '云圣智能', "index": [0, 1]},
        ]
    },
    {
        "idx": 74,
        "id": "3cc7851d01ee68c2a1be4a8e87014d6e",
        "cg": [
            {'role': '被投资方', 'argument': '禾氏美康', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 77,
        "id": "844ac0acbb13ecbccd4923ac7ad368b1",
        "cg": [
            {'role': '被投资方', 'argument': '居然新零售', "index": [4]},
            {'role': '领投方', 'argument': '加华资本', "index": [6]},
        ]
    },
    {
        "idx": 79,
        "id": "be46ace6cfd0447a9b5278de98bd8c8c",
        "cg": [
            {'role': '被投资方', 'argument': 'Bruush Oral Care Inc', "index": [0]},
        ]
    },
    {
        "idx": 83,
        "id": "da8f29a5ce27036464fbd06ac3628c8b",
        "cg": [
            {'role': '被投资方', 'argument': '比亚迪半导体有限公司', "index": [0]},
        ]
    },
    {
        "idx": 84,
        "id": "9b4e1d553682165fada9ec806323d3a4",
        "cg": [
            {'role': '被投资方', 'argument': '威视佰科', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 89,
        "id": "b2260ea3504b47ff6d7082e4227f88d6",
        "cg": [
            {'role': '被投资方', 'argument': '星源材质', "index": [2]},
        ]
    },
    {
        "idx": 91,
        "id": "945bd7839541e39bff58a794fe85f3a4",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [0, 1, 2]},
            {'role': '被投资方', 'argument': '百奥赛图', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 93,
        "id": "e4b61f6fd89347f745563ea62cd89aca",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
        ]
    },
    {
        "idx": 94,
        "id": "8236f89d800fd39e1bd3f965f04fc378",
        "cg": [
            {'role': '被投资方', 'argument': '艾米森', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 96,
        "id": "08292e6c99447a2e5c93f0209586a845",
        "cg": [
            {'role': '被投资方', 'argument': '拜腾汽车', "index": [1, 2, 3]},
        ]
    },
    {
        "idx": 97,
        "id": "307dc10622afb726db923619399378ca",
        "cg": [
            {'role': '被投资方', 'argument': '普罗格', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 98,
        "id": "5a9340e572fe304582d242b0cad88724",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [12]},
            {'role': '被投资方', 'argument': 'DaDa', "index": [5]},
        ]
    },
    {
        "idx": 99,
        "id": "c0a4e8ddf2ecebd6b44238036e98e3cc",
        "cg": [
            {'role': '被投资方', 'argument': 'Ayenda', "index": [0, 1]},
            {'role': '领投方', 'argument': 'Kaszek Ventures', "index": [0]},
        ]
    },
    {
        "idx": 100,
        "id": "72d9a4bb2a22b3ab0f8b463a8ec91633",
        "cg": [
            {'role': '被投资方', 'argument': '驭能科技', "index": [0, 1, 5]},
        ]
    },
    {
        "idx": 102,
        "id": "9a532a04a4ab30fbac99a7e5fa4e230d",
        "cg": [
            {'role': '被投资方', 'argument': '铂诺商学', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 103,
        "id": "088905e6090b25264850fbf293927c60",
        "cg": [
            {'role': '融资金额', 'argument': '亿元', "index": [1]},
            {'role': '被投资方', 'argument': '易游集团', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 106,
        "id": "872865434e8866ad5fd497f939acfad9",
        "cg": [
            {'role': '被投资方', 'argument': '贝丰科技', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 108,
        "id": "2c5b55e9e9a8f2135add6ffbaa0234f3",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2]},
        ]
    },
    {
        "idx": 109,
        "id": "fd880697024eec42879d58b43ab5dd9d",
        "cg": [
            {'role': '被投资方', 'argument': '宝兰德', "index": [0, 1, 4]},
            {'role': '被投资方', 'argument': '美迪西', "index": [0, 1, 4, 5]},
        ]
    },
    {
        "idx": 110,
        "id": "52a1436ae0344088fba78b5adaeb31d7",
        "cg": [
            {'role': '被投资方', 'argument': '新宁物流', "index": [0, 1]},
            {'role': '投资方', 'argument': '京东', "index": [0, 1]},
        ]
    },
    {
        "idx": 116,
        "id": "9fbab557408277921b7053e6cb71144f",
        "cg": [
            {'role': '被投资方', 'argument': '钉学', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 117,
        "id": "9b5b06c71826b817a119dbfcf00f1a02",
        "cg": [
            {'role': '被投资方', 'argument': '钉学', "index": [0, 2, 3, 5]},
        ]
    },
    {
        "idx": 119,
        "id": "be7870d0d0c6db88e5067dcb75ae1579",
        "cg": [
            {'role': '被投资方', 'argument': 'Farfetch', "index": [9, 10]},
        ]
    },
    {
        "idx": 120,
        "id": "b311b27612ccd9c976aaef9334c69ce3",
        "cg": [
            {'role': '被投资方', 'argument': '立达融医', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 121,
        "id": "9a3dc783c64fa22cf05dbf20056f1be6",
        "cg": [
            {'role': '被投资方', 'argument': '小码大众', "index": [0, 1, 2, 3, 9]},
        ]
    },
    {
        "idx": 122,
        "id": "7a34d7086677f4999af03023702e302f",
        "cg": [
            {'role': '被投资方', 'argument': '诺而为', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '协立投资', "index": [0]},
        ]
    },
    {
        "idx": 123,
        "id": "950c3fe7faa177b0d8a0253b6ab67803",
        "cg": [
            {'role': '被投资方', 'argument': '国信证券', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 125,
        "id": "9b916236af35e0e45ebabef4f382fcb7",
        "cg": [
            {'role': '被投资方', 'argument': '孚日集团股份有限公司', "index": [0, 1, 2, 3, 4, 5, 6]},
        ]
    },
    {
        "idx": 126,
        "id": "c00b7e5685649d80cf236c769b4c402d",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [2]},
            {'role': '领投方', 'argument': '腾讯', "index": [1]},
            {'role': '领投方', 'argument': '博裕资本', "index": [1]},
        ]
    },
    {
        "idx": 130,
        "id": "9e3805ad52fd79b9432510cf7678c956",
        "cg": [
            {'role': '被投资方', 'argument': 'Moka', "index": [0, 1]},
        ]
    },
    {
        "idx": 131,
        "id": "1c60f1e442eb730055c251fe05afcec3",
        "cg": [
            {'role': '被投资方', 'argument': '喜尔康', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '金鼎资本', "index": [0, 2]},
        ]
    },
    {
        "idx": 132,
        "id": "52b282832be31b246c27e8a6b2890204",
        "cg": [
            {'role': '被投资方', 'argument': '中科云创', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 134,
        "id": "d1f7d9cafb32a9579c6a5e99aa2da241",
        "cg": [
            {'role': '被投资方', 'argument': '南京麦迪森生物科技有限公司', "index": [0]},
        ]
    },
    {
        "idx": 135,
        "id": "8cff33664a1b7077809585356dde96fa",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 136,
        "id": "67ca3bd68c542824ba38f92920c8095f",
        "cg": [
            {'role': '被投资方', 'argument': '笔神作文', "index": [0]},
        ]
    },
    {
        "idx": 139,
        "id": "2397d0685d8c5850639a15df30f93a9e",
        "cg": [
            {'role': '被投资方', 'argument': '金杏商务', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 142,
        "id": "41dd21204064682a3671becbc275b0c0",
        "cg": [
            {'role': '被投资方', 'argument': '隆基股份', "index": [0, 1, 2, 3, 4, 6]},
        ]
    },
    {
        "idx": 143,
        "id": "a019d7989530bcd24f5716bcd60072fb",
        "cg": [
            {'role': '被投资方', 'argument': '英雄体育', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 145,
        "id": "e6e97c8a533416e62b82512b7806f26a",
        "cg": [
            {'role': '被投资方', 'argument': 'Evelo Biosciences', "index": [0, 1]},
        ]
    },
    {
        "idx": 146,
        "id": "bede494f35ba0c06e6799e39faf391be",
        "cg": [
            {'role': '被投资方', 'argument': '唯医骨科', "index": [0, 1]},
        ]
    },
    {
        "idx": 150,
        "id": "59c78bdc50c3875c00ed208211cceaa6",
        "cg": [
            {'role': '被投资方', 'argument': '弘量科技', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 154,
        "id": "b6c198ec62e6317a0c3d72a1ad9b51aa",
        "cg": [
            {'role': '被投资方', 'argument': '高川自动化', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 156,
        "id": "80d4f913d827b64a5dc8d3f8207a1935",
        "cg": [
            {'role': '被投资方', 'argument': '合众公司', "index": [0]},
        ]
    },
    {
        "idx": 158,
        "id": "b70e669d434e037ddfd2429b52e4c9d1",
        "cg": [
            {'role': '融资金额', 'argument': '5 亿元', "index": [1]},
        ]
    },
    {
        "idx": 161,
        "id": "ae87f215a844d4a960201853c5e6e34c",
        "cg": [
            {'role': '被投资方', 'argument': '南都电源', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 167,
        "id": "c8ceeec4f373de6f4c4c13198c44be64",
        "cg": [
            {'role': '被投资方', 'argument': '格力地产', "index": [11, 12, 13, 14, 15]},
        ]
    },
    {
        "idx": 168,
        "id": "0d65bc02434151fe5a87d95eacc4863a",
        "cg": [
            {'role': '被投资方', 'argument': '北海康成', "index": [0, 1, 2, 3, 4]},
            {'role': '领投方', 'argument': '药明康德', "index": [3]},
        ]
    },
    {
        "idx": 170,
        "id": "90bbdc5708bd4837aae39238f061c1ff",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
            {'role': '被投资方', 'argument': '北海康成', "index": [0, 1, 2, 5]},
        ]
    },
    {
        "idx": 171,
        "id": "4e6267d8cab035fd98ec982eea87456c",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [0, 1, 3, 4]},
            {'role': '领投方', 'argument': '钟鼎资本', "index": [1]},
            {'role': '领投方', 'argument': '元璟资本', "index": [0]},
        ]
    },
    {
        "idx": 173,
        "id": "922f4e04a11efd1597f831de6c17d4fa",
        "cg": [
            {'role': '被投资方', 'argument': '新宙邦', "index": [10, 11]},
        ]
    },
    {
        "idx": 174,
        "id": "95fa1b954bf0e5083c630584324034d8",
        "cg": [
            {'role': '被投资方', 'argument': '轩田工业', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 175,
        "id": "3afaf0caa400d834977ec55c9f085ae1",
        "cg": [
            {'role': '被投资方', 'argument': '数势科技', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 179,
        "id": "594d461919305e617ec3521588a40c83",
        "cg": [
            {'role': '被投资方', 'argument': '泰山财险', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 180,
        "id": "8374fd294156e46bafac8503cb98df93",
        "cg": [
            {'role': '被投资方', 'argument': '青山纸业', "index": [6]},
        ]
    },
    {
        "idx": 181,
        "id": "78171eb6a236a00a3ac344cc4f1920bd",
        "cg": [
            {'role': '融资金额', 'argument': '1亿美元', "index": [0]},
        ]
    },
    {
        "idx": 182,
        "id": "2ab6d945ec8ee724a66c8b8ceba04d12",
        "cg": [
            {'role': '被投资方', 'argument': '韦拓生物', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 183,
        "id": "f5c23cb103918d473f4d2979858ea5b2",
        "cg": [
            {'role': '被投资方', 'argument': '美凯龙', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 184,
        "id": "c0cb7544871aba1b3be03b6c7361cecc",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
            {'role': '被投资方', 'argument': '聚云位智', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 185,
        "id": "c0cb7544871aba1b3be03b6c7361cecc",
        "cg": [
            {'role': '被投资方', 'argument': 'Arrival', "index": [0, 1]},
        ]
    },
    {
        "idx": 186,
        "id": "01a7ab4313401a68ac31a63d3458624c",
        "cg": [
            {'role': '被投资方', 'argument': '通用五矿医院', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 187,
        "id": "9ae3f3b3cd1e33dc403923231b0a4bcc",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
            {'role': '被投资方', 'argument': '南燕信息', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '千骥资本', "index": [0, 2]},
        ]
    },
    {
        "idx": 189,
        "id": "3e87982ee9f71db45919f34af7fae8c9",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 2]},
        ]
    },
    {
        "idx": 191,
        "id": "0c4cff66fcc7a7cba1e4fc93e796c827",
        "cg": [
            {'role': '被投资方', 'argument': '岁悦生', "index": [0, 1]},
        ]
    },
    {
        "idx": 195,
        "id": "958366af52fe9fa503185af466c0229a",
        "cg": [
            {'role': '被投资方', 'argument': 'Luua噜啊', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 197,
        "id": "4d0cd394a631d4d42e2e2e5f5d6a68b2",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [1, 2]},
            {'role': '被投资方', 'argument': '微龛半导体', "index": [0, 1]},
        ]
    },
    {
        "idx": 203,
        "id": "8cd7b3b079dbae982e26ffbf34469fe7",
        "cg": [
            {'role': '被投资方', 'argument': '汉威科技', "index": [7]},
        ]
    },
    {
        "idx": 204,
        "id": "fbe5b74bcc5ae80b6b20903b4be731ec",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [10]},
            {'role': '领投方', 'argument': '五源资本', "index": [1, 2]},
            {'role': '领投方', 'argument': '经纬中国', "index": [1]},
            {'role': '领投方', 'argument': '云启资本', "index": [1]},
        ]
    },
    {
        "idx": 207,
        "id": "1a8f0ffa711d816396b5f9a7cf6c3e7c",
        "cg": [
            {'role': '被投资方', 'argument': 'Travelio', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 208,
        "id": "5785d4f7b553abb74ca3d04a9953084e",
        "cg": [
            {'role': '被投资方', 'argument': '好好榜样', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 209,
        "id": "cd5d63f2dcb92716a9e2c0f797c8f4ef",
        "cg": [
            {'role': '被投资方', 'argument': '捍宇医疗', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '弘晖资本', "index": [0, 2]},
        ]
    },
    {
        "idx": -1,
        "id": "ac27a8306f9a6e6a077ca495e74550e9",
        "cg": [
            {'role': '被投资方', 'argument': 'Dewey’s Bakery', "index": [0, 1]},
            {'role': '领投方', 'argument': 'Eurazeo', "index": [0]},
            {'role': '投资方', 'argument': 'Eurazeo', "index": [0]},
        ]
    },
    {
        "idx": -1,
        "id": "89aa1f32d1da97646bcdde5980c72872",
        "cg": [
            {'role': '领投方', 'argument': '阿里巴巴', "index": [2]},
            {'role': '投资方', 'argument': '阿里巴巴', "index": [2]},
        ]
    },
    {
        "idx": -1,
        "id": "f1c6996a139a1867c5165f41566eff90",
        "cg": [
            {'role': '投资方', 'argument': '松禾资本', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '松禾资本', "index": [0, 3]},
        ]
    },
    {
        "idx": 211,
        "id": "0b76311d39f85d7189b130b813e9188e",
        "cg": [
            {'role': '被投资方', 'argument': '麦耘财商学院', "index": [0, 1]},
        ]
    },
    {
        "idx": 214,
        "id": "b9861618cdf98b06081f39b8197e41d9",
        "cg": [
            {'role': '被投资方', 'argument': '艺人星球', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 217,
        "id": "65ae61c6bdb29f270995328da4ae4fd9",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [3]},
        ]
    },
    {
        "idx": 219,
        "id": "51daef8833599490018a80b32c6d4a6e",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
            {'role': '被投资方', 'argument': '在途商旅', "index": [0, 1]},
        ]
    },
    {
        "idx": 220,
        "id": "9e74d764fe18f118b97080da580b75b5",
        "cg": [
            {'role': '被投资方', 'argument': '零氪科技', "index": [0, 1, 4]},
        ]
    },
    {
        "idx": 221,
        "id": "91760c289c45f389f15dc42a632814c0",
        "cg": [
            {'role': '被投资方', 'argument': '众安在线', "index": [1, 2, 3, 4]},
        ]
    },
    {
        "idx": 223,
        "id": "4af9d88da38858193f3e8d423c4160cf",
        "cg": [
            {'role': '被投资方', 'argument': '招商证券', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 227,
        "id": "c43a5c10d1e67d66fe58c29481faabdf",
        "cg": [
            {'role': '被投资方', 'argument': '广汽蔚来', "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 229,
        "id": "917932ca1d5a1d849f8c4d2105c836d5",
        "cg": [
            {'role': '被投资方', 'argument': '筷子科技', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 233,
        "id": "71aa4a545edaafb1130a40b081dc77b7",
        "cg": [
            {'role': '被投资方', 'argument': '神州租车', "index": [10, 11]},
            {'role': '投资方', 'argument': '华平投资', "index": [3, 4, 5]},
        ]
    },
    {
        "idx": 234,
        "id": "a37757b498f115049ea10577986512cf",
        "cg": [
            {'role': '被投资方', 'argument': '科技龙头ETF', "index": [0, 1, 3, 4, 5, 6]},
        ]
    },
    {
        "idx": 235,
        "id": "b3719b0f32d8212f042731f81410e133",
        "cg": [
            {'role': '投资方', 'argument': '步长制药', "index": [0, 1, 8]},
        ]
    },
    {
        "idx": 236,
        "id": "3bf5cb98d606b0f5fe7d9012f114ced6",
        "cg": [
            {'role': '被投资方', 'argument': '伟杰信生物', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 239,
        "id": "66055c084f1039fb3eeeca8e4c8c2513",
        "cg": [
            {'role': '被投资方', 'argument': '晖泽光伏', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 242,
        "id": "59ec9f808a305ed7da81fe7939778acf",
        "cg": [
            {'role': '被投资方', 'argument': '云从科技', "index": [0, 1]},
        ]
    },
    {
        "idx": 243,
        "id": "4ee2457ccb15002185a0cd4165af106c",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 246,
        "id": "bb3b201b8e29ec1ae7923cbfcfd1f7de",
        "cg": [
            {'role': '被投资方', 'argument': '荔枝', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 247,
        "id": "e3a1d717582bd09157b8a10aedd7e112",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 248,
        "id": "43e7dcde776751354e8757eabf876a1a",
        "cg": [
            {'role': '被投资方', 'argument': 'Razorpay', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 249,
        "id": "becc2be193db80704f4c42f37547795b",
        "cg": [
            {'role': '被投资方', 'argument': '华安证券', "index": [5, 6]},
        ]
    },
    {
        "idx": 250,
        "id": "d5426974628219eb4bf6e16d5ca52686",
        "cg": [
            {'role': '被投资方', 'argument': '闻医富馨', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 251,
        "id": "24a55c4a06ffad3cf0c82e5b25858c3b",
        "cg": [
            {'role': '被投资方', 'argument': 'Giphy', "index": [4]},
        ]
    },
    {
        "idx": 253,
        "id": "6bb6477cc6178f47990a89e62085a7a0",
        "cg": [
            {'role': '被投资方', 'argument': 'Airbnb', "index": [5]},
        ]
    },
    {
        "idx": 254,
        "id": "23522ac2b4435076a8e85c2168ec3f32",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
            {'role': '被投资方', 'argument': '中科汉天下', "index": [0, 1]},
        ]
    },
    {
        "idx": 256,
        "id": "93ce363f36158184ba65fac5b6a2fb3a",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 257,
        "id": "872e67890fdb4a54e773c087eb4a19ce",
        "cg": [
            {'role': '被投资方', 'argument': '圆通速递', "index": [10]},
        ]
    },
    {
        "idx": 258,
        "id": "e543c3a879f02cd16ff010eff6bcb45b",
        "cg": [
            {'role': '被投资方', 'argument': '一汽轿车', "index": [5]},
        ]
    },
    {
        "idx": 259,
        "id": "7be7a5fb53f6b4d9ac77b9224a159a54",
        "cg": [
            {'role': '被投资方', 'argument': '淘租公', "index": [0, 1]},
        ]
    },
    {
        "idx": 261,
        "id": "2c684b58c6f13614d069d6bceaa03c01",
        "cg": [
            {'role': '被投资方', 'argument': 'Cortica', "index": [4]},
        ]
    },
    {
        "idx": 263,
        "id": "f07f31ba3fffddf25813e6ed85a725b0",
        "cg": [
            {'role': '被投资方', 'argument': '邻汇吧', "index": [0, 1]},
        ]
    },
    {
        "idx": 264,
        "id": "7229a7ebd6b3993e61329e7c3dabad2d",
        "cg": [
            {'role': '被投资方', 'argument': '谊品生鲜', "index": [0, 1]},
        ]
    },
    {
        "idx": 265,
        "id": "dbd77e28251bd2fe1b8e31fa8603af07",
        "cg": [
            {'role': '事件时间', 'argument': '2018年', "index": [1]},
            {'role': '被投资方', 'argument': '金融壹账通', "index": [5]},
        ]
    },
    {
        "idx": 266,
        "id": "9e9828a2ea6b4b70f88b4025dcfbcbf4",
        "cg": [
            {'role': '被投资方', 'argument': '禧涤智能Triooo', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 269,
        "id": "6a9bb6bfd2ba83d9bc79eaa3bc2db6d6",
        "cg": [
            {'role': '融资金额', 'argument': '5亿美元', "index": [2]},
            {'role': '被投资方', 'argument': '瓦拉里斯公司', "index": [2]},
        ]
    },
    {
        "idx": 270,
        "id": "3cd3a7fd6c7274d1311e1968a098ee02",
        "cg": [
            {'role': '被投资方', 'argument': '青岛英诺包装', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 274,
        "id": "fb52607f499da4c340a4ee832ea38a3e",
        "cg": [
            {'role': '被投资方', 'argument': '亦诺微医药', "index": [0, 1]},
        ]
    },
    {
        "idx": 276,
        "id": "6e0dde539acb6e713feab54ceabaf8e6",
        "cg": [
            {'role': '被投资方', 'argument': '明治', "index": [6]},
        ]
    },
    {
        "idx": 277,
        "id": "f97fca085b0bd7508893554efe5e7431",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 3]},
            {'role': '被投资方', 'argument': '迈吉客科技', "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 284,
        "id": "203246b688a838590db7481c5d3a7057",
        "cg": [
            {'role': '被投资方', 'argument': '贝瑞基因', "index": [4]},
        ]
    },
    {
        "idx": 287,
        "id": "8af35576f0d448b00747e089dd04b869",
        "cg": [
            {'role': '被投资方', 'argument': '找食材', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 289,
        "id": "4c5a04e615d0f6f72e05e3d6474518ad",
        "cg": [
            {'role': '被投资方', 'argument': '西方石油公司', "index": [5]},
        ]
    },
    {
        "idx": 291,
        "id": "de0d6335064614a8a32af7fbece84981",
        "cg": [
            {'role': '投资方', 'argument': '长城汽车', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 292,
        "id": "0ec2d2980266ba96286c7d243eeedb94",
        "cg": [
            {'role': '被投资方', 'argument': '德风科技', "index": [0, 1]},
        ]
    },
    {
        "idx": 294,
        "id": "dede659fe75809c013c9dd757102dea4",
        "cg": [
            {'role': '被投资方', 'argument': 'Jukedeck', "index": [2]},
        ]
    },
    {
        "idx": 297,
        "id": "ca063843c8ff9f18c354196e205ec6f8",
        "cg": [
            {'role': '被投资方', 'argument': '同济环境', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 298,
        "id": "799ce44b3cc559ecb8c049f825ecbef2",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'B', "index": [1]},
            {'role': '融资轮次', 'argument': 'E', "index": [0]},
        ]
    },
    {
        "idx": 301,
        "id": "8b3bcfa06dd9a571ace1023906a91a42",
        "cg": [
            {'role': '被投资方', 'argument': '中芯国际', "index": [2]},
        ]
    },
    {
        "idx": 302,
        "id": "e5d556cc9f51eb18bcb4ccbee251c078",
        "cg": [
            {'role': '被投资方', 'argument': '懒龙龙', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 304,
        "id": "efd9b633aa9fa1a15989e56a4357f757",
        "cg": [
            {'role': '被投资方', 'argument': '峰米科技', "index": [0, 4]},
        ]
    },
    {
        "idx": 305,
        "id": "daf696d879b57977125f9b68647fea61",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
        ]
    },
    {
        "idx": 306,
        "id": "ff7eae67c8193ea5c2cb32d8ac6cd49a",
        "cg": [
            {'role': '被投资方', 'argument': 'Zomato', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 307,
        "id": "fada818c1ae5580503c4e71bed8f1e23",
        "cg": [
            {'role': '被投资方', 'argument': '德美医疗', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 309,
        "id": "fd46738914d6a7a24ca470cd48f63655",
        "cg": [
            {'role': '被投资方', 'argument': '卫宁健康', "index": [0, 1, 2, 4]},
        ]
    },
    {
        "idx": 311,
        "id": "b74f05e85feb4a125205f589668e3328",
        "cg": [
            {'role': '投资方', 'argument': '一汽解放', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 312,
        "id": "5b882c33077c3ad8a21357a44a216404",
        "cg": [
            {'role': '被投资方', 'argument': '恒大股份', "index": [0, 1]},
        ]
    },
    {
        "idx": 314,
        "id": "34694ea56bdf11ef63f6ebd7abd9ea06",
        "cg": [
            {'role': '融资金额', 'argument': '2亿元', "index": [0]},
        ]
    },
    {
        "idx": 315,
        "id": "609673c1ca7e0538206bba0e3fcbb7c4",
        "cg": [
            {'role': '被投资方', 'argument': '快车道', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 317,
        "id": "3c7824d20c1ef4cd81521e9017939403",
        "cg": [
            {'role': '被投资方', 'argument': 'Gojek', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 318,
        "id": "ed6fdaf8bc81b45986bff26b85c6603a",
        "cg": [
            {'role': '被投资方', 'argument': 'ZStack', "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 320,
        "id": "66966b35680235aa19670b7c8fcf2441",
        "cg": [
            {'role': '被投资方', 'argument': '恒实科技', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 323,
        "id": "a17647e0ca6d302a07f140ce2047ceb9",
        "cg": [
            {'role': '被投资方', 'argument': '埃睿迪', "index": [0, 1, 4]},
            {'role': '领投方', 'argument': '字节跳动', "index": [0, 2]},
        ]
    },
    {
        "idx": 324,
        "id": "1d1b48d6c43e60ef1638519217f82dc6",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '被投资方', 'argument': '融卡科技', "index": [0, 1, 4]},
            {'role': '领投方', 'argument': '达晨创投', "index": [0, 3]},
            {'role': '投资方', 'argument': '达晨创投', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 325,
        "id": "86124517e70120db2fa3dbd7a1048b90",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [6, 8, 10]},
            {'role': '被投资方', 'argument': 'Bakkt', "index": [3, 4]},
        ]
    },
    {
        "idx": 326,
        "id": "638d6e02465fc748b74b8dd3e9778e77",
        "cg": [
            {'role': '被投资方', 'argument': '阿斯顿·马丁', "index": [3]},
        ]
    },
    {
        "idx": 332,
        "id": "d5c095dbe101356b6b8b08871238661a",
        "cg": [
            {'role': '被投资方', 'argument': "Byju's", "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 333,
        "id": "d59df0afc51244f0a450a5076c648f28",
        "cg": [
            {'role': '被投资方', 'argument': "逸迅科技", "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 334,
        "id": "683acf3e623ca2046b3e5b65365466fb",
        "cg": [
            {'role': '被投资方', 'argument': "Gett", "index": [0, 1]},
        ]
    },
    {
        "idx": 336,
        "id": "8e6c8701c6945a699b66e0988f4c30af",
        "cg": [
            {'role': '被投资方', 'argument': "ZuBlu", "index": [0, 1, 4]},
            {'role': '领投方', 'argument': 'Wavemaker Partners', "index": [0, 2]},
        ]
    },
    {
        "idx": -1,
        "id": "d0eef4e5d6c0463a40ac4a07e79df543",
        "cg": [
            {'role': '领投方', 'argument': '倚锋资本', "index": [0, 2]},
        ]
    },
    {
        "idx": 337,
        "id": "9dbd4d360c73e3f6514cb3f57df2fc03",
        "cg": [
            {'role': '被投资方', 'argument': "7分甜", "index": [0]},
        ]
    },
    {
        "idx": 338,
        "id": "8c38e4b573d2d7391cd6d44cafdeb789",
        "cg": [
            {'role': '被投资方', 'argument': "易成自动驾驶", "index": [0, 1]},
        ]
    },
    {
        "idx": 340,
        "id": "67c1d0bd53896bab5883033316dafff6",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
        ]
    },
    {
        "idx": 342,
        "id": "bbf4d607d043ac466d9db1ceae8d2caf",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
        ]
    },
    {
        "idx": 344,
        "id": "d169fa7509d097201fddfd926b6617f6",
        "cg": [
            {'role': '融资金额', 'argument': '4亿元', "index": [2]},
        ]
    },
    {
        "idx": 345,
        "id": "3df87ff21fbb58e9ce9dc44595c50a12",
        "cg": [
            {'role': '被投资方', 'argument': "点滴能源", "index": [0, 1, 5]},
            {'role': '领投方', 'argument': '蓝驰创投', "index": [0, 2]},
        ]
    },
    {
        "idx": 348,
        "id": "004f3ae85880f30ade5be7f24c188309",
        "cg": [
            {'role': '被投资方', 'argument': "晨光生物", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 350,
        "id": "77455ebaa2966157a9d2b8b7e2dfab43",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [0, 1, 3, 4]},
            {'role': '被投资方', 'argument': "方恩医药", "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 352,
        "id": "f6d4b4619b789705977e7c008ff6af93",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 354,
        "id": "b217e00676cf7d8f545d838bf5757215",
        "cg": [
            {'role': '被投资方', 'argument': "行云集团", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 356,
        "id": "e091fd9e2bbcba81a50217e2b9a4456a",
        "cg": [
            {'role': '被投资方', 'argument': "Cell Propulsion", "index": [0, 1, 2, 4]},
        ]
    },
    {
        "idx": 357,
        "id": "f4fb55f26ab9adefdad0918096173677",
        "cg": [
            {'role': '被投资方', 'argument': "红豆股份", "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 358,
        "id": "1853bc7228ea59df0bac2b7da81d033a",
        "cg": [
            {'role': '被投资方', 'argument': "上海万面智能科技有限公司", "index": [0]},
        ]
    },
    {
        "idx": 359,
        "id": "5336f0eaf3b38631938617ca043e8e71",
        "cg": [
            {'role': '被投资方', 'argument': "蚂蚁集团", "index": [1]},
        ]
    },
    {
        "idx": 362,
        "id": "c6aa79e0f842334c91eb1240f64a5e46",
        "cg": [
            {'role': '被投资方', 'argument': "世纪互联", "index": [0, 1]},
        ]
    },
    {
        "idx": 363,
        "id": "8af5c0046ecc5660c0f0cdea8cd4c551",
        "cg": [
            {'role': '被投资方', 'argument': "海格通信", "index": [0, 1, 2, 3, 4, 7]},
        ]
    },
    {
        "idx": 367,
        "id": "622b3479a5514c02a16620b775c9f9c9",
        "cg": [
            {'role': '被投资方', 'argument': "Tim Hortons", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 368,
        "id": "91f5f695fd63f0de53e355d03d799066",
        "cg": [
            {'role': '被投资方', 'argument': "云霁科技", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 369,
        "id": "4fc09812ba6b70b28d08262fb6e6f832",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 371,
        "id": "2c6e1064f48e8aa58af88a5eed692007",
        "cg": [
            {'role': '被投资方', 'argument': "壹网壹创", "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 372,
        "id": "c72ba59b94e6eba9fcb49d1f6b03d71c",
        "cg": [
            {'role': '被投资方', 'argument': "中粮集团", "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 374,
        "id": "8b90a5e2327fd71b10065d6684cc2afd",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [1]},
            {'role': '被投资方', 'argument': "BankBazaar", "index": [0, 1, 2, 3, 6]},
            {'role': '投资方', 'argument': "亚马逊", "index": [0, 1, 2, 3, 5]},
            {'role': '领投方', 'argument': "亚马逊", "index": [5]},
        ]
    },
    {
        "idx": 377,
        "id": "ffde547285380c7f8183f25a739f838c",
        "cg": [
            {'role': '被投资方', 'argument': "VillageMD", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 378,
        "id": "5107683d1d5915c08a2b05f1e3e3ed93",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [1, 7]},
            {'role': '被投资方', 'argument': "Molecular Assemblies", "index": [1]},
        ]
    },
    {
        "idx": 379,
        "id": "185ba4712ece8b4615d873d9bfaa6e0f",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '被投资方', 'argument': "校聘网", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 384,
        "id": "3e10c0f98f7d4f7abc0bce2768aa6d27",
        "cg": [
            {'role': '被投资方', 'argument': "均联智行", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 385,
        "id": "3c4620adb42660bf214d7466c6a0b8c4",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
            {'role': '被投资方', 'argument': "Cover Genius", "index": [0, 1]},
        ]
    },
    {
        "idx": 386,
        "id": "9a77eb80a163e4eee91e2369c1a0c074",
        "cg": [
            {'role': '被投资方', 'argument': "Unacademy", "index": [0, 1, 2, 3, 4, 5]},
        ]
    },
    {
        "idx": 387,
        "id": "8b4d4bb3a17730e1ccfc39b6d191446f",
        "cg": [
            {'role': '被投资方', 'argument': "华大智造", "index": [0, 1, 3, 4, 5]},
        ]
    },
    {
        "idx": 388,
        "id": "489d2c91479d6069f5fda60d1aceccf0",
        "cg": [
            {'role': '被投资方', 'argument': "华大智造", "index": [7]},
        ]
    },
    {
        "idx": 392,
        "id": "98f8e435644250805aa2b9bd1dc6070c",
        "cg": [
            {'role': '被投资方', 'argument': "社区猫AI接送宝", "index": [0, 1, 5]},
        ]
    },
    {
        "idx": 394,
        "id": "35e911c0d8c52849b777dcf9e91a4ff1",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [1, 3]},
        ]
    },
    {
        "idx": 395,
        "id": "0b4c1034359e59f23e1aca873415c165",
        "cg": [
            {'role': '被投资方', 'argument': "Tala", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 396,
        "id": "ee2ce3b12b3bcc9cc7af5202734ae2ad",
        "cg": [
            {'role': '被投资方', 'argument': "钛深科技", "index": [0, 1, 4]},
            {'role': '领投方', 'argument': "小米集团", "index": [0, 2]},
        ]
    },
    {
        "idx": 398,
        "id": "b265c8512936c5caf70a364350a3a741",
        "cg": [
            {'role': '被投资方', 'argument': "搜狗", "index": [0, 1, 4]},
        ]
    },
    {
        "idx": 399,
        "id": "cd4170eb6066ad8dac37b17ec7e4a9ff",
        "cg": [
            {'role': '被投资方', 'argument': "好衣库", "index": [0, 1]},
        ]
    },
    {
        "idx": 400,
        "id": "3eba247bb916011c79eec6d872a93cb5",
        "cg": [
            {'role': '被投资方', 'argument': "好衣库", "index": [0, 1]},
        ]
    },
    {
        "idx": 401,
        "id": "8ef33bd7a10b03550429f6b3aa742146",
        "cg": [
            {'role': '被投资方', 'argument': "联新医疗科技", "index": [0, 1, 3]},
            {'role': '领投方', 'argument': "深创投", "index": [0, 2]},
        ]
    },
    {
        "idx": 403,
        "id": "6483684cfc0228ed032848761c8b17b2",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [2]},
            {'role': '被投资方', 'argument': "XJet", "index": [2, 4, 5]},
        ]
    },
    {
        "idx": 404,
        "id": "382a7045d334f9fa352f19991173809c",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '被投资方', 'argument': "Yolobus", "index": [0, 1, 2, 5]},
            {'role': '领投方', 'argument': "Nexus Venture Partners", "index": [0]},
            {'role': '投资方', 'argument': "Nexus Venture Partners", "index": [0]},
            {'role': '投资方', 'argument': "India Quotient", "index": [0]},
        ]
    },
    {
        "idx": 407,
        "id": "920b4f5639acdf36f6aa02e0973685ce",
        "cg": [
            {'role': '被投资方', 'argument': "三顿半", "index": [0, 1, 2, 3]},
            {'role': '被投资方', 'argument': "超级零", "index": [0, 1]},
        ]
    },
    {
        "idx": 408,
        "id": "6632c3c5f9960a7071a4b974b5b78134",
        "cg": [
            {'role': '被投资方', 'argument': "世茂服务", "index": [5, 6]},
        ]
    },
    {
        "idx": 409,
        "id": "5d991612b4d838d0d00fdcb3e7cf76f5",
        "cg": [
            {'role': '被投资方', 'argument': "家春秋酒业", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 410,
        "id": "4d9a9d37ec9a500c05710a11d30d1028",
        "cg": [
            {'role': '被投资方', 'argument': "海科新源", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 412,
        "id": "c7ec450435ae5f760fc091ab08a279e4",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '被投资方', 'argument': "埃克斯工业", "index": [0, 1]},
        ]
    },
    {
        "idx": 416,
        "id": "ba6fe52702198a7dc2269484ae30a273",
        "cg": [
            {'role': '被投资方', 'argument': "东兴证券", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 418,
        "id": "bc24e083a65db9ae242d5088986b2cd4",
        "cg": [
            {'role': '融资轮次', 'argument': '首', "index": [0, 1, 2]},
            {'role': '被投资方', 'argument': "Zenius", "index": [0, 1, 3, 4, 5, 6]},
        ]
    },
    {
        "idx": 419,
        "id": "bf9c3d77e8d23871986b5b156ce1717e",
        "cg": [
            {'role': '被投资方', 'argument': "迪英加", "index": [0, 1, 4]},
            {'role': '领投方', 'argument': "中金资本", "index": [0, 5]},
            {'role': '投资方', 'argument': "中金资本", "index": [0, 1, 5]},
        ]
    },
    {
        "idx": 420,
        "id": "187b4793dd70a3652ad58b35144ee047",
        "cg": [
            {'role': '被投资方', 'argument': "智云股份", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 424,
        "id": "e63772125d70dacb8cbfb96b877a1e01",
        "cg": [
            {'role': '被投资方', 'argument': "广联航空", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 425,
        "id": "f5eb599df620b5bc46402a2d8086b7d7",
        "cg": [
            {'role': '被投资方', 'argument': "云从科技", "index": [2]},
        ]
    },
    {
        "idx": 424,
        "id": "e63772125d70dacb8cbfb96b877a1e01",
        "cg": [
            {'role': '被投资方', 'argument': "广联航空", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 430,
        "id": "4fba87b3ca3ba371587c8eaba2da32fe",
        "cg": [
            {'role': '被投资方', 'argument': "杭州信达奥体置业有限公司", "index": [0, 1]},
        ]
    },
    {
        "idx": 431,
        "id": "58fc81ddea6439cad329775b4773c7e4",
        "cg": [
            {'role': '被投资方', 'argument': "萌芽熊", "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "idx": 432,
        "id": "aed90ae875c963806f1bca03213ddbac",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [7]},
        ]
    },
    {
        "idx": 435,
        "id": "be11a1377c0d97bd5f5da146990597fa",
        "cg": [
            {'role': '被投资方', 'argument': "天宜上佳", "index": [3, 4]},
        ]
    },
    {
        "idx": 436,
        "id": "ec86cc5bd00159382d5e851f43114f7b",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [0, 1]},
        ]
    },
    {
        "idx": 437,
        "id": "63400e83748945084f91d49ec2e954ee",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [2, 3]},
            {'role': '被投资方', 'argument': "编程猫", "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 439,
        "id": "cb84b1331e94c777edefd24185b45cdd",
        "cg": [
            {'role': '被投资方', 'argument': "HIFIVE", "index": [0, 1, 2]},
        ]
    },
    {
        "idx": 441,
        "id": "c281623fc63f5f3aa4c37fb48d6f4fee",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "idx": 442,
        "id": "2a462fca2cda8967f6eb55dc5ccc0dd7",
        "cg": [
            {'role': '被投资方', 'argument': "十荟团", "index": [0, 1, 2, 3]},
        ]
    },
    {
        "idx": 444,
        "id": "5a86f6b6ce888f62b8311925ea655643",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2, 4, 5]},
            {'role': '被投资方', 'argument': "转转", "index": [0, 1, 2, 3, 4, 5, 12]},
        ]
    },
    {
        "idx": 446,
        "id": "84e1bf19b5daaf7423eb697ea07d36c2",
        "cg": [
            {'role': '被投资方', 'argument': "生活有鱼", "index": [0, 1, 4]},
        ]
    },
    {
        "idx": 447,
        "id": "bd887ea4321f8bf4ba138d2256c02511",
        "cg": [
            {'role': '被投资方', 'argument': "小鹿科技", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 450,
        "id": "049ba0512b8560b46387719146209223",
        "cg": [
            {'role': '被投资方', 'argument': "丰疆智能", "index": [0, 1, 4]},
        ]
    },
    {
        "idx": 454,
        "id": "205c6b08afaa1707037c65cb259a14c6",
        "cg": [
            {'role': '被投资方', 'argument': "INLT", "index": [6]},
        ]
    },
    {
        "idx": 456,
        "id": "bfb8f91282dfe1f8e8112be2a041c4a0",
        "cg": [
            {'role': '被投资方', 'argument': "车和家", "index": [0, 1, 2, 3, 9, 10]},
        ]
    },
    {
        "idx": 457,
        "id": "389655d8f75c394dbd62f80eced5ad00",
        "cg": [
            {'role': '被投资方', 'argument': "MaxAB", "index": [0, 1]},
        ]
    },
    {
        "idx": 459,
        "id": "fa0f130ca029f2d036e8b3a0d9a01d2d",
        "cg": [
            {'role': '被投资方', 'argument': "柏睿数据", "index": [0, 1, 2, 3]},
            {'role': '领投方', 'argument': "海通证券", "index": [2]},
        ]
    },
    {
        "idx": 460,
        "id": "6d84f8e50af81192fc12eb19cedc80f0",
        "cg": [
            {'role': '被投资方', 'argument': "Finematter", "index": [0, 1]},
        ]
    },
    {
        "idx": 465,
        "id": "617f3bb288f3dba51f43e379287c30b5",
        "cg": [
            {'role': '被投资方', 'argument': "作业帮", "index": [0, 1, 2, 3, 4, 5, 6, 7]},
            {'role': '领投方', 'argument': "方源资本", "index": [2]},
            {'role': '投资方', 'argument': "方源资本", "index": [0, 2]},
            {'role': '领投方', 'argument': "老虎环球", "index": [1]},
            {'role': '投资方', 'argument': "老虎环球", "index": [1]},
        ]
    },
    {
        "idx": 466,
        "id": "25cdf42366f258cad881f9c454cbbee7",
        "cg": [
            {'role': '被投资方', 'argument': "和码编程", "index": [0]},
        ]
    },
    {
        "idx": 468,
        "id": "71a7fedbb252e62c1a3921934b461dc8",
        "cg": [
            {'role': '被投资方', 'argument': "怒喵科技", "index": [0, 1]},
        ]
    },
    {
        "idx": 471,
        "id": "d8e7f9df00f4660a08b6836df0a8db02",
        "cg": [
            {'role': '被投资方', 'argument': "诺信创联", "index": [0, 1, 3]},
            {'role': '领投方', 'argument': "经纬中国", "index": [0, 4]},
        ]
    },
    {
        "idx": 473,
        "id": "545eab1809574276ebb1d675b0484e6f",
        "cg": [
            {'role': '被投资方', 'argument': "丰行智图", "index": [0, 1]},
        ]
    },
    {
        "idx": 475,
        "id": "c256cfc8a1d9e441807289a1ddf9a298",
        "cg": [
            {'role': '被投资方', 'argument': "芯华章", "index": [0, 1, 3]},
            {'role': '领投方', 'argument': "高瓴创投", "index": [0, 3]},
            {'role': '投资方', 'argument': "高瓴创投", "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 477,
        "id": "1b08c89e5092157a48dc0481785a73fc",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '领投方', 'argument': "深创投", "index": [0, 2]},
        ]
    },
    {
        "idx": 479,
        "id": "db789a5c88d5225a6814d590f2a48caf",
        "cg": [
            {'role': '被投资方', 'argument': '云玖科技', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 480,
        "id": "84b91679c167e33c1ff5bfee1260dc2a",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '被投资方', 'argument': '安龙生物', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 481,
        "id": "42a10cd4d68121bdb9d8aa12980466e5",
        "cg": [
            {'role': '被投资方', 'argument': 'Robinhood', "index": [1, 2]},
        ]
    },
    {
        "idx": 482,
        "id": "f15be0f3309fd00c11c6029704ba4029",
        "cg": [
            {'role': '被投资方', 'argument': '蔚来', "index": [9]},
        ]
    },
    {
        "idx": 486,
        "id": "d90fa42b9d2e590dcda16c9c8c673f19",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2]},
            {'role': '被投资方', 'argument': '爱华盈通', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': "明裕创投", "index": [0, 2]},
        ]
    },
    {
        "idx": 487,
        "id": "13f7a34532fc2a25c1ee83d81529fe68",
        "cg": [
            {'role': '被投资方', 'argument': 'Stasher', "index": [0, 1]},
        ]
    },
    {
        "idx": 488,
        "id": "44783d3d4aff2eabd6120ff4212e2c48",
        "cg": [
            {'role': '被投资方', 'argument': '轻喜到家', "index": [0, 1, 3]},
        ]
    },
    {
        "idx": 490,
        "id": "2c37b5f0950b7031461e0ab9fad08a06",
        "cg": [
            {'role': '被投资方', 'argument': 'GOQii', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 0,
        "id": "92df4d967a438df93620424da7f1ead6",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 7,
        "id": "7f52ea0f3882d5a073d5d2ac042349d7",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 10,
        "id": "0637c434b891dd1654a3456ba1956b4e",
        "cg": [
            {'role': '被投资方', 'argument': 'PatPat', "index": [0, 1, 3, 4]},
            {'role': '融资轮次', 'argument': 'C', "index": [3]},
            {'role': '融资轮次', 'argument': 'D', "index": [7]},
        ]
    },
    {
        "type": "add",
        "idx": 6,
        "id": "62dc238cfa781ffad93c888a0be3df0f",
        "cg": [
            {'role': '被投资方', 'argument': '茵曼内衣', "index": [0, 1, 12]},
        ]
    },
    {
        "type": "add",
        "idx": 7,
        "id": "b584593b49eab86b4baca1d4eef91c8c",
        "cg": [
            {'role': '被投资方', 'argument': '火星盒子', "index": [0, 1, 2, 3, 7, 8]},
            {'role': '融资轮次', 'argument': 'A', "index": [7, 8, 9]},
        ]
    },
    {
        "type": "add",
        "idx": 9,
        "id": "e861f745e922b971b079ab14c9232316",
        "cg": [
            {'role': '被投资方', 'argument': '芯启源', "index": [1, 2, 3, 9, 10, 14, 15]},
            {'role': '被投资方', 'argument': '中科驭数', "index": [1, 8]},
            {'role': '融资轮次', 'argument': 'A', "index": [16]},
        ]
    },
    {
        "type": "add",
        "idx": 11,
        "id": "1ac44653f80611c20beb898d3499b581",
        "cg": [
            {'role': '被投资方', 'argument': '永璞', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 12,
        "id": "30b8c230c8b6e9a69ed0915a663fd57a",
        "cg": [
            {'role': '被投资方', 'argument': 'pidan彼诞', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'A', "index": [6]},
            {'role': '融资轮次', 'argument': 'B', "index": [22]},
            {'role': '事件时间', 'argument': '近日', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 13,
        "id": "8cb36c2aa07a6fa6214ddfeb62ce30d6",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
            {'role': '投资方', 'argument': '光速中国', "index": [0]},
            {'role': '领投方', 'argument': '光速中国', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 15,
        "id": "50c9f6059e536b49f4515c6a9a508499",
        "cg": [
            {'role': '被投资方', 'argument': '地平线', "index": [0, 2, 3]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 16,
        "id": "4765e06457ff1d7269fe4fdf9809dbd3",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2, 4, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 19,
        "id": "0235a011441c4a66c101aa82826d15f4",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [5]},
            {'role': '被投资方', 'argument': '实在智能', "index": [0, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 21,
        "id": "ff66aecfeeb7432db06effde9f490d27",
        "cg": [
            {'role': '被投资方', 'argument': 'Touchdog它它', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 22,
        "id": "7a71245ec327e070e165fd62f28e72d4",
        "cg": [
            {'role': '被投资方', 'argument': 'ToyCity', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '不二资本', "index": [0, 1]},
            {'role': '投资方', 'argument': '不二资本', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 26,
        "id": "5e28b11bed5279999a9bfa1f6da1f6c7",
        "cg": [
            {'role': '被投资方', 'argument': '椿风', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 27,
        "id": "0dd29dff743a1132b1250c2c7522582b",
        "cg": [
            {'role': '被投资方', 'argument': '谊品生鲜', "index": [0, 1, 2, 3, 4, 5]},
            {'role': '领投方', 'argument': '今日资本', "index": [0, 1, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 29,
        "id": "56a9d1c7b87acfc470510a66677985ed",
        "cg": [
            {'role': '被投资方', 'argument': 'Keep', "index": [0, 1, 2, 3, 4, 35]},
        ]
    },
    {
        "type": "add",
        "idx": 31,
        "id": "e5ba96eae2cef36e680c7790113f6511",
        "cg": [
            {'role': '被投资方', 'argument': '霸蛮', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 32,
        "id": "8c7d1bb5f4148245de886de5f6c03a50",
        "cg": [
            {'role': '被投资方', 'argument': 'Coterie', "index": [0, 2, 3, 4, 5]},
            {'role': '融资轮次', 'argument': 'C', "index": [1, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 34,
        "id": "cb94684af01bb77daca2e610e6e4453a",
        "cg": [
            {'role': '被投资方', 'argument': '美拆', "index": [0, 1, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 37,
        "id": "145db2679f72204206d251fe0264619d",
        "cg": [
            {'role': '被投资方', 'argument': '弹力猩球', "index": [0, 1, 3, 16]},
            {'role': '融资轮次', 'argument': '天使', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 39,
        "id": "e3fef59bc4d354be818502df2120ead0",
        "cg": [
            {'role': '被投资方', 'argument': '缤果盒子', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 40,
        "id": "989db29860acc808746ab9d2dd935003",
        "cg": [
            {'role': '被投资方', 'argument': '拜安传感', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 41,
        "id": "95822fbc656b3bb361e495394e7141b2",
        "cg": [
            {'role': '被投资方', 'argument': '桃园眷村', "index": [1, 2, 5, 22]},
            {'role': '融资轮次', 'argument': '首', "index": [0, 2, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 43,
        "id": "e713b99fe9ef24b692e705a216b1d558",
        "cg": [
            {'role': '被投资方', 'argument': 'ONLY WRITE', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 45,
        "id": "ae055d1b3c32584a77511eea94a8e454",
        "cg": [
            {'role': '被投资方', 'argument': '迈铸半导体', "index": [0, 2, 3]},
            {'role': '被投资方', 'argument': '合宙', "index": [0, 1, 2, 3]},
            {'role': '被投资方', 'argument': '华芯卓越创业投资中心（有限合伙）', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 46,
        "id": "374210455fe1f749ebb200f66e41df5e",
        "cg": [
            {'role': '被投资方', 'argument': '完美日记', "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20]},
            {'role': '融资金额', 'argument': '10亿美元', "index": [2]},
            {'role': '领投方', 'argument': '高瓴资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": -1,
        "id": "eaf0245800ddb97e21c60d31d9d9cbd5",
        "cg": [

            {'role': '领投方', 'argument': '天图投资', "index": [0, 1, 2, 4]},
            {'role': '投资方', 'argument': '天图投资', "index": [0, 1, 2, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 48,
        "id": "1e5c9f4f51ac7603cab6d70cee166aeb",
        "cg": [
            {'role': '被投资方', 'argument': '堕落虾', "index": [0, 1]},
            {'role': '被投资方', 'argument': '深圳市洪堡智慧餐饮科技有限公司', "index": [0]},
            {'role': '被投资方', 'argument': '洪堡智慧', "index": [2, 7, 8]},
        ]
    },
    {
        "type": "add",
        "idx": 51,
        "id": "c847d856027b35a4a1a5889cca89969c",
        "cg": [
            {'role': '被投资方', 'argument': 'Andie Swim', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [1, 3, 6]},
        ]
    },
    {
        "type": "add",
        "idx": 52,
        "id": "75fd6763673bf47a4be646f3fb1f0308",
        "cg": [
            {'role': '被投资方', 'argument': '英雄体育VSPN', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 54,
        "id": "e5b4355f6207340f0f28f0a0416824d0",
        "cg": [
            {'role': '被投资方', 'argument': '喜茶', "index": [0, 32, 33, 34, 35, 49, 53]},
            {'role': '被投资方', 'argument': '因味茶', "index": [3]},
        ]
    },
    {
        "type": "add",
        "idx": 57,
        "id": "a4dbc7780796b74c4ac492a462b91ffd",
        "cg": [
            {'role': '被投资方', 'argument': '妃鱼', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 58,
        "id": "0f8e2e00628d296f15aa6e8962a64f9f",
        "cg": [
            {'role': '被投资方', 'argument': '元初食品', "index": [0, 1, 4]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 59,
        "id": "5b68c7375ba70d44cd94ad834403ec43",
        "cg": [
            {'role': '被投资方', 'argument': '鹰集', "index": [0, 1, 2, 23]},
        ]
    },
    {
        "type": "add",
        "idx": 60,
        "id": "1750cb0ad99bb86590d9a0c372e01afb",
        "cg": [
            {'role': '被投资方', 'argument': '苏宁易购零售云', "index": [0, 1, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 62,
        "id": "10ebfba6edad9965b5c2d17ad8da6d9d",
        "cg": [
            {'role': '被投资方', 'argument': 'Fashionphile', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 64,
        "id": "600cc006609b463d50c671af6fade5d0",
        "cg": [
            {'role': '被投资方', 'argument': '芯能半导体', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 65,
        "id": "d6c56a51c8a5e0aef2241ce70333317c",
        "cg": [
            {'role': '被投资方', 'argument': '中科融合', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 66,
        "id": "be770662531a7ba71ac28a74dfb6963b",
        "cg": [
            {'role': '被投资方', 'argument': '三顿半', "index": [0, 1, 3, 4]},
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
            {'role': '领投方', 'argument': '红杉资本', "index": [0, 1]},
            {'role': '领投方', 'argument': '峰瑞资本', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 73,
        "id": "9c91d433afde09c562e1f2574ac6d9e2",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '领投方', 'argument': '红杉资本中国基金', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 76,
        "id": "0411b133dceebccc16a691abcb847699",
        "cg": [
            {'role': '被投资方', 'argument': 'LOHO', "index": [1, 4, 11]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '融资轮次', 'argument': 'B', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 82,
        "id": "e450207d3d33f27077f7c95cea6b888d",
        "cg": [
            {'role': '被投资方', 'argument': '胖仙女', "index": [0, 1, 3, 11, 12]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 84,
        "id": "137c19f44e9bf7568b5129b8cf4105db",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 3, 4, 6, 9]},
            {'role': '领投方', 'argument': 'IDG资本', "index": [0]},
            {'role': '领投方', 'argument': 'CMC资本', "index": [0]},
            {'role': '领投方', 'argument': '正心谷', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 85,
        "id": "d311a345fdabc22afb858b91e81f2222",
        "cg": [
            {'role': '被投资方', 'argument': '速石科技', "index": [0, 2, 3]},
            {'role': '领投方', 'argument': '元禾璞华', "index": [0, 1]},
            {'role': '投资方', 'argument': '元禾璞华', "index": [0, 1]},
            {'role': '领投方', 'argument': '凯辉基金', "index": [1]},
            {'role': '投资方', 'argument': '凯辉基金', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": -1,
        "id": "b93e0254e1bcf5771f3347339c152174",
        "cg": [
            {'role': '领投方', 'argument': 'CPE源峰', "index": [0]},
            {'role': '投资方', 'argument': 'CPE源峰', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 88,
        "id": "e9ff24e54eb4cd044e644608cca7abf1",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '领投方', 'argument': '毅达资本', "index": [0]},
            {'role': '投资方', 'argument': '毅达资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 94,
        "id": "9116875a40f488d01bfb366fae2d4ad6",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 95,
        "id": "6f13e6046fc94f1bc50bf58f8f1a0ec7",
        "cg": [
            {'role': '被投资方', 'argument': '喜姐炸串', "index": [0, 1]},
            {'role': '被投资方', 'argument': '夸父炸串', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 96,
        "id": "851b3a825ca501b4d4863b7a1493bd78",
        "cg": [
            {'role': '被投资方', 'argument': '锅圈', "index": [0, 1, 3, 4, 5, 6]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 103,
        "id": "3cfb6bd78c6acbdc611aa37a06bac1b6",
        "cg": [
            {'role': '被投资方', 'argument': '元气森林', "index": [0, 1, 3, 5, 6, 7, 8, 9]},
        ]
    },
    {
        "type": "add",
        "idx": 106,
        "id": "d18ff5cfab67b0b25e8b9ddcc0a6a25a",
        "cg": [
            {'role': '被投资方', 'argument': '溪木源', "index": [0, 1, 3, 4, 5]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': '种子', "index": [0]},
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
            {'role': '领投方', 'argument': '弘毅投资', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 108,
        "id": "3e25984a66274bac6a9bad42533444cb",
        "cg": [
            {'role': '被投资方', 'argument': 'M Stand', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 110,
        "id": "9e0a0e7eadce4c9e73db1e0f9b5d5470",
        "cg": [
            {'role': '被投资方', 'argument': '智联安', "index": [0, 2]},
            {'role': '领投方', 'argument': '北京集成电路尖端芯片基金', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 113,
        "id": "aa818b4c914b1a4a54752fcd98cd919f",
        "cg": [
            {'role': '被投资方', 'argument': 'StockX', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 123,
        "id": "1b5ca493e6400499614e3771e5406984",
        "cg": [
            {'role': '融资轮次', 'argument': 'D', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 127,
        "id": "f0f2f79ec91a1ed1a24051bb8b4ffbe7",
        "cg": [
            {'role': '被投资方', 'argument': 'gaga鲜语', "index": [0, 1, 3]},
            {'role': '领投方', 'argument': '君联资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": -1,
        "id": "a9545a8091458c2101727b9c67d3a170",
        "cg": [
            {'role': '领投方', 'argument': '九宜城', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 130,
        "id": "5efb239f5431e08a0de0b888296134a6",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 134,
        "id": "fba787d1ebc997f0c09436a1ca6d4426",
        "cg": [
            {'role': '被投资方', 'argument': '中科融合', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 137,
        "id": "11ff6ad9a930abe0df73c4691331f0c8",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 138,
        "id": "5dbe127bb105684cf545397bdf9fa4f9",
        "cg": [
            {'role': '被投资方', 'argument': '洛阳卡卡', "index": [0, 1, 2, 24]},
        ]
    },
    {
        "type": "add",
        "idx": 139,
        "id": "bb1f4af54fe14bd527c681f2c4b77aba",
        "cg": [
            {'role': '被投资方', 'argument': 'LGD-Gaming', "index": [0, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 141,
        "id": "1b795888acfc80eb6f9cd420198ea4fc",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [3]},
        ]
    },
    {
        "type": "add",
        "idx": 146,
        "id": "3ba5936051c1a30915dad56df60b57d1",
        "cg": [
            {'role': '被投资方', 'argument': '猫员外', "index": [0, 1, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [1, 3, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 156,
        "id": "c405aae56248ba0d5e0fb99a86ec6d41",
        "cg": [
            {'role': '融资金额', 'argument': '3亿美元', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 169,
        "id": "ffa39360f321e0b1d10c0b39e36461e5",
        "cg": [
            {'role': '被投资方', 'argument': '沉浸世界', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]}
        ]
    },
    {
        "type": "add",
        "idx": 175,
        "id": "9355370d4356cca0d1912f467a491ca9",
        "cg": [
            {'role': '被投资方', 'argument': '蛙来哒', "index": [0, 1, 2, 3, 9]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 3]}
        ]
    },
    {
        "type": "add",
        "idx": 176,
        "id": "1e44c6e9fdb1c01a614e47cb78b38f9e",
        "cg": [
            {'role': '被投资方', 'argument': 'KK集团', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'C', "index": [4]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '领投方', 'argument': 'CMC资本', "index": [1]},
            {'role': '领投方', 'argument': '经纬中国', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 178,
        "id": "04f18999ed560df588000492f4ea3fb8",
        "cg": [
            {'role': '被投资方', 'argument': '华泰半导体', "index": [0, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 180,
        "id": "bc4c2b9e195917979c4e11fcfa189709",
        "cg": [
            {'role': '被投资方', 'argument': '洋码头', "index": [0, 1, 3, 4]},
            {'role': '融资轮次', 'argument': 'D', "index": [3]},
        ]
    },
    {
        "type": "add",
        "idx": 181,
        "id": "260c9029db28dd2945f554a6bc4019ec",
        "cg": [
            {'role': '被投资方', 'argument': 'FIREFLY超能小红梳', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 182,
        "id": "e730ed5c685da5bad032361a245aba8f",
        "cg": [
            {'role': '被投资方', 'argument': '食验室', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 183,
        "id": "63c4625b41e61550f233d5809927a522",
        "cg": [
            {'role': '被投资方', 'argument': '纵慧芯光', "index": [0, 3, 14, 21]},
        ]
    },
    {
        "type": "add",
        "idx": 189,
        "id": "b8949e7b0281c719285d494e95766e85",
        "cg": [
            {'role': '被投资方', 'argument': 'F5未来商店', "index": [0]},
            {'role': '被投资方', 'argument': '缤果盒子', "index": [2, 5]},
            {'role': '融资轮次', 'argument': 'A', "index": [5, 13]},
            {'role': '融资轮次', 'argument': '首', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 192,
        "id": "277f5c45f635e07827e9ddc2042cfb98",
        "cg": [
            {'role': '被投资方', 'argument': '零美优选', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '荟聚资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 194,
        "id": "bc27a44f7da6de38ca0941bb1495796b",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 196,
        "id": "a5521d7256ab24db923109fe2c28beac",
        "cg": [
            {'role': '被投资方', 'argument': '蚂蚁金服', "index": [0, 2, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 200,
        "id": "ef5d2c579da29de50a53f214c1c543de",
        "cg": [
            {'role': '被投资方', 'argument': '辣妈帮', "index": [0, 1, 2, 3, 4, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 201,
        "id": "70f5358a75694c28c0379d6b782d166c",
        "cg": [
            {'role': '被投资方', 'argument': 'GrubMarket', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 205,
        "id": "c91fc9cafd1018a8b3c376d37681e3f3",
        "cg": [
            {'role': '被投资方', 'argument': '贝壳找房', "index": [16, 18, 19]},
            {'role': '投资方', 'argument': '腾讯', "index": [2, 3, 4, 5]},
            {'role': '领投方', 'argument': '腾讯', "index": [5]},
            {'role': '融资轮次', 'argument': 'D', "index": [0, 2]},
            {'role': '融资轮次', 'argument': 'C', "index": [5]},
        ]
    },
    {
        "type": "add",
        "idx": 206,
        "id": "c00faad3beeb4349519ffec8ded0002c",
        "cg": [
            {'role': '被投资方', 'argument': '柠季', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 210,
        "id": "ae89e8907a5f83fb70ff81420e4dbe70",
        "cg": [
            {'role': '被投资方', 'argument': '每日优鲜', "index": [0, 1, 2, 3, 11]},
            {'role': '融资轮次', 'argument': 'B', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 211,
        "id": "12a386d110e941c011f56555bee3ba71",
        "cg": [
            {'role': '被投资方', 'argument': '埃瓦科技', "index": [0, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
            {'role': '领投方', 'argument': '中科创星', "index": [1]},
            {'role': '投资方', 'argument': '中科创星', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 214,
        "id": "b2d78715dc3e58c3b4789f28ca47ae26",
        "cg": [
            {'role': '被投资方', 'argument': '美肌饮品', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 216,
        "id": "ca2f2ca7dd2f4645671c8c708ef7d891",
        "cg": [
            {'role': '被投资方', 'argument': '缤果盒子', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 217,
        "id": "5da0451e87828a83b3c1dd10905117ff",
        "cg": [
            {'role': '被投资方', 'argument': '贵凤凰', "index": [0, 1, 32, 33, 34]},
        ]
    },
    {
        "type": "add",
        "idx": 218,
        "id": "6e5bd9e76e90b8a3b30b1692076561b0",
        "cg": [
            {'role': '被投资方', 'argument': '帝奥微电子有限公司', "index": [0]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 2]},
            {'role': '领投方', 'argument': '沃衍资本', "index": [0, 1]},
            {'role': '投资方', 'argument': '沃衍资本', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 219,
        "id": "c2092a4951ccf5391edad4fd429e2768",
        "cg": [
            {'role': '被投资方', 'argument': '大师兄', "index": [0, 1, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 221,
        "id": "eeddd78099631264048665d34c727e73",
        "cg": [
            {'role': '被投资方', 'argument': '蜻蜓FM', "index": [0, 1, 2, 3, 7]},
            {'role': '融资轮次', 'argument': 'F', "index": [4]},
            {'role': '领投方', 'argument': '中文在线', "index": [0, 1]},
            {'role': '投资方', 'argument': '中文在线', "index": [0, 1, 5, 6]},
            {'role': '投资方', 'argument': '小米', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 226,
        "id": "cb5c3d8c5c94f70227721d8157ff00d0",
        "cg": [
            {'role': '被投资方', 'argument': '言几', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 227,
        "id": "35919160235dd2c2a76411fe1d83ca32",
        "cg": [
            {'role': '被投资方', 'argument': '医药魔方', "index": [0, 1, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 229,
        "id": "db7fa95c664f3ce1cd557b33ed54cf22",
        "cg": [
            {'role': '被投资方', 'argument': '京派鲜卤', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 231,
        "id": "1ecb6dc6c1c5137ee03f1c95a21b1209",
        "cg": [
            {'role': '被投资方', 'argument': '爱芯科技', "index": [0, 1, 2, 3]},
            {'role': '领投方', 'argument': '韦豪创芯', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 236,
        "id": "3792a8a96433c33db4041db701cebba2",
        "cg": [
            {'role': '被投资方', 'argument': '盘子女人坊', "index": [0, 1, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 238,
        "id": "c1101a6833a04d66634b58877887bf7c",
        "cg": [
            {'role': '被投资方', 'argument': '卓视智通', "index": [0, 3, 4, 5, 6, 12]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 243,
        "id": "df767d576c918b465809a64f97115fa8",
        "cg": [
            {'role': '被投资方', 'argument': 'UNISKIN优时颜', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2, 3]},
            {'role': '领投方', 'argument': '弘毅创投', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 244,
        "id": "c53145bd99e7db9ddb33af51e7209fb4",
        "cg": [
            {'role': '被投资方', 'argument': '爆爆奢', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 246,
        "id": "f534e28e9471197f81e77919690e1b5f",
        "cg": [
            {'role': '被投资方', 'argument': '爱华盈通', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 253,
        "id": "d38a5e5e6c6f87b4e643d1a1f7ba95ca",
        "cg": [
            {'role': '被投资方', 'argument': '锅圈食汇', "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 257,
        "id": "42be677603381832beb47f6c72d24ece",
        "cg": [
            {'role': '被投资方', 'argument': 'Cerebras', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 258,
        "id": "a0e2e48c38a8c786481203a415b48662",
        "cg": [
            {'role': '被投资方', 'argument': '赛卓电子', "index": [0, 2, 9, 10]},
            {'role': '领投方', 'argument': '尚颀资本', "index": [0, 1]},
            {'role': '投资方', 'argument': '尚颀资本', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 261,
        "id": "a4a3c63f58f6bb887d169f0adf2d03d3",
        "cg": [
            {'role': '被投资方', 'argument': '云祈文旅', "index": [1, 3]},
            {'role': '投资方', 'argument': '汉博商业', "index": [2, 6]},
        ]
    },
    {
        "type": "add",
        "idx": 263,
        "id": "ff75bd8cd1b11cbd4be2bfd48c17c4da",
        "cg": [
            {'role': '被投资方', 'argument': 'Particle Fever粒子狂热', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 266,
        "id": "59d5eb98c5d6e0dce800200ff45d951f",
        "cg": [
            {'role': '被投资方', 'argument': '宠幸', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'B', "index": [1, 3, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 269,
        "id": "01168100556eac66712246ed6199be79",
        "cg": [
            {'role': '被投资方', 'argument': 'The Citizenry', "index": [0, 1, 2, 14]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 270,
        "id": "4156beed3793974f3822c1b403e14109",
        "cg": [
            {'role': '被投资方', 'argument': 'SECRE时萃咖啡', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '博将资本', "index": [0]},
            {'role': '投资方', 'argument': '博将资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 275,
        "id": "fefdc470cd33606b95d8eabf8edd6178",
        "cg": [
            {'role': '被投资方', 'argument': '英雄体育VSPN', "index": [3]},
            {'role': '被投资方', 'argument': '香蕉游戏传媒', "index": [8, 9, 10]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 277,
        "id": "82805071d0d67acb9e3a19831e0fc50e",
        "cg": [
            {'role': '投资方', 'argument': '保隆科技', "index": [0, 1, 2, 3, 4, 5]},
            {'role': '融资轮次', 'argument': 'C', "index": [2, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 278,
        "id": "033cb0212c7c037f3db4e49ba4861396",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
            {'role': '领投方', 'argument': '中信证券投资', "index": [0, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 281,
        "id": "80a59cc7e8ed12c755405bb6886b5a62",
        "cg": [
            {'role': '被投资方', 'argument': 'Keep', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 283,
        "id": "cfb16e2e8f75d84e0cc963fac8388097",
        "cg": [
            {'role': '被投资方', 'argument': '爱便利', "index": [1, 4, 21]},
            {'role': '融资轮次', 'argument': 'B', "index": [9]},
        ]
    },
    {
        "type": "add",
        "idx": 286,
        "id": "ea0fd3abf18517b86bec1d66574dea2f",
        "cg": [
            {'role': '被投资方', 'argument': '五月美妆', "index": [0, 1, 2, 3, 4, 8]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 288,
        "id": "6f13a70e2b7ac15a7a880076a4cdecbb",
        "cg": [
            {'role': '被投资方', 'argument': '它赞', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 291,
        "id": "610bfc61cae062c93b5415872f35f341",
        "cg": [
            {'role': '被投资方', 'argument': '德尔科技', "index": [0, 3, 4, 8]},
            {'role': '领投方', 'argument': '国家制造业转型升级基金', "index": [0]},
            {'role': '投资方', 'argument': '国家制造业转型升级基金', "index": [0]},
            {'role': '领投方', 'argument': '深创投', "index": [0]},
            {'role': '投资方', 'argument': '深创投', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 293,
        "id": "4e388a5478a85534605dcf62ac08d2fa",
        "cg": [
            {'role': '被投资方', 'argument': '钟薛高', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
            {'role': '事件时间', 'argument': '2018年', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 299,
        "id": "46f26ed913dd0c54b5ccc25d6ba1833a",
        "cg": [
            {'role': '被投资方', 'argument': 'bosie', "index": [0, 2, 3, 5, 8]},
            {'role': '融资轮次', 'argument': 'A', "index": [1, 8, 9]},
            {'role': '融资轮次', 'argument': 'D', "index": [0]},
            {'role': '融资轮次', 'argument': '首', "index": [1, 2]},
            {'role': '融资轮次', 'argument': 'B', "index": [8, 11]},
            {'role': '领投方', 'argument': 'B站', "index": [2, 3]},
            {'role': '领投方', 'argument': '五源资本', "index": [0, 1]},
            {'role': '领投方', 'argument': '金沙江创投', "index": [1]},
            {'role': '领投方', 'argument': 'GGV', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 300,
        "id": "4cc7d35aba10ef94b06a7c59faf6ec7f",
        "cg": [
            {'role': '被投资方', 'argument': 'KK馆', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 302,
        "id": "ce3ee5766e589043d1f8a21d7147decc",
        "cg": [
            {'role': '被投资方', 'argument': '瑞幸咖啡', "index": [0, 1, 3, 9, 12]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '融资轮次', 'argument': 'B', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 303,
        "id": "456a62e6087a5205824b394febbef426",
        "cg": [
            {'role': '被投资方', 'argument': '物垣文化', "index": [0, 1, 2, 3, 11]},
            {'role': '领投方', 'argument': 'IDG资本', "index": [0, 1]},
            {'role': '投资方', 'argument': 'IDG资本', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 306,
        "id": "0f158e3b8b63f04510fcabef9af9ff82",
        "cg": [
            {'role': '被投资方', 'argument': 'Magic Leap', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 309,
        "id": "c2d377888f747da17ceab1d817530ddd",
        "cg": [
            {'role': '被投资方', 'argument': '十荟团', "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                          16, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35,
                                                          36, 37, 38, 39, 41]},
            {'role': '融资轮次', 'argument': 'D', "index": [1, 3, 4]},
            {'role': '融资轮次', 'argument': 'B', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 314,
        "id": "440475dca9cdd02912360b5addbb5313",
        "cg": [
            {'role': '被投资方', 'argument': '云知声', "index": [0, 1, 2, 3, 4, 5, 6, 7]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '融资轮次', 'argument': 'C', "index": [3, 5, 7, 10]},
            {'role': '领投方', 'argument': '中国互联网投资基金', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 315,
        "id": "d80a597b2573ddb93794583b3b84a535",
        "cg": [
            {'role': '被投资方', 'argument': '之间文化', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
            {'role': '融资金额', 'argument': '千万元', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 317,
        "id": "864ffd1f3b414710adbbd3810003c4c5",
        "cg": [
            {'role': '被投资方', 'argument': '十荟团', "index": [4, 14, 27]},
            {'role': '被投资方', 'argument': '兴盛优选', "index": [1, 2, 3]},
            {'role': '事件时间', 'argument': '今年', "index": [0, 6]},
        ]
    },
    {
        "type": "add",
        "idx": 322,
        "id": "fcf66030b0ff32e4c69bca0f2b80b218",
        "cg": [
            {'role': '被投资方', 'argument': 'Magic Leap', "index": [0, 1, 4, ]},
        ]
    },
    {
        "type": "add",
        "idx": 324,
        "id": "9e346eb0ba872e04db91066e261dcb96",
        "cg": [
            {'role': '被投资方', 'argument': 'Beauty Choice', "index": [0]},
            {'role': '被投资方', 'argument': 'BeautyChoice（东点西点）', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 326,
        "id": "691c904c718c287c467cb87bc8d991e6",
        "cg": [
            {'role': '被投资方', 'argument': '育想家', "index": [0, 1, 2, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 329,
        "id": "355c0e8409e4ebfc61f85aaafbb860af",
        "cg": [
            {'role': '被投资方', 'argument': '太空精酿', "index": [0, 1, 2, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 330,
        "id": "e73cccaf719c206dab8da0f23a07edcd",
        "cg": [
            {'role': '被投资方', 'argument': '三顿半', "index": [0, 1, 2, 3]},
            {'role': '被投资方', 'argument': '沪上阿姨', "index": [0, 1]},
            {'role': '被投资方', 'argument': '墨茉点心局', "index": [0, 1]},
            {'role': '被投资方', 'argument': '三餐有料', "index": [0, 1]},
            {'role': '被投资方', 'argument': '五爷拌面', "index": [0, 1]},
            {'role': '被投资方', 'argument': '奈雪的茶', "index": [4]},
            {'role': '融资轮次', 'argument': 'A', "index": [2, 4, 7, 8, 9]},
        ]
    },
    {
        "type": "add",
        "idx": 332,
        "id": "cf00883912fc70d54c45bb7c29f4c043",
        "cg": [
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 334,
        "id": "70ef24b493ee3af1d7fabc591bdd1224",
        "cg": [
            {'role': '被投资方', 'argument': '鹰集咖啡', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 337,
        "id": "34e11fcc99540bc6c65366b019d55fcc",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 340,
        "id": "7c89b740003b5adca560ede8e3499d08",
        "cg": [
            {'role': '被投资方', 'argument': '虎头局', "index": [0, 1, 3, 4, 5, 6]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '红杉中国', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 341,
        "id": "e7fffca6d6b5e14c3598c4d4c680dd80",
        "cg": [
            {'role': '被投资方', 'argument': '乐乐茶', "index": [0, 1, 2, 3, 7]},
        ]
    },
    {
        "type": "add",
        "idx": 344,
        "id": "5eef435a55e818eae17ec11585b97ed3",
        "cg": [
            {'role': '被投资方', 'argument': 'Mercato', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 348,
        "id": "fb42d34100a430751e16d084d570575c",
        "cg": [
            {'role': '融资轮次', 'argument': 'A', "index": [9]},
        ]
    },
    {
        "type": "add",
        "idx": 350,
        "id": "cb39384d4844154c458a6796f8261365",
        "cg": [
            {'role': '被投资方', 'argument': '逐本', "index": [0, 1, 3, 28, 29]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1, 2, 3]},
            {'role': '领投方', 'argument': '元璟资本', "index": [0]},
            {'role': '投资方', 'argument': '元璟资本', "index": [0, 1]},
            {'role': '领投方', 'argument': '元生资本', "index": [0]},
            {'role': '投资方', 'argument': '元生资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 352,
        "id": "f6577d2e748a40c27fcd2e61e38af765",
        "cg": [
            {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '融资轮次', 'argument': 'C', "index": [2, 5, 6]},
        ]
    },
    {
        "type": "add",
        "idx": 353,
        "id": "ae1b5673a93ee1f03a507e8a77b1908a",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 3, 11, 14, 15, 16, 26, 27]},
            {'role': '融资轮次', 'argument': 'B', "index": [105, 106]},
            {'role': '被投资方', 'argument': '肇观电子', "index": [0, 1, 2, 3, 7]},
            {'role': '被投资方', 'argument': '瑞思凯微', "index": [0, 1, 4]},
            {'role': '被投资方', 'argument': '慧能泰', "index": [0, 1, 3]},
            {'role': '被投资方', 'argument': '芯翼信息科技', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 361,
        "id": "616879f3fec39ceff583807033022e27",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '帝视科技', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 362,
        "id": "b75b490cf169729a050d693125da59b8",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '泰斗微电子', "index": [0, 1, 4]},
            {'role': '融资轮次', 'argument': 'B', "index": [0]},
            {'role': '融资轮次', 'argument': 'C', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 363,
        "id": "3432c0955ab5b283f1c2f3adee67c91d",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '仙堡', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 364,
        "id": "e8aaf7c846aec651bbfb6f9abf3832cd",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '同程生活', "index": [1, 6, 25, 26, 27]},
            {'role': '被投资方', 'argument': '兴盛优选', "index": [4, 5, 6, 7]},
        ]
    },
    {
        "type": "add",
        "idx": 367,
        "id": "79d5b7beaee7029576f43ab4579712fe",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': 'OUTPUT', "index": [1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 368,
        "id": "3006012aeac21dd641c7084f3799ca8f",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '小鹏汇天', "index": [0, 2, 3]},
            {'role': '融资轮次', 'argument': 'A', "index": [0, 1]},
            {'role': '领投方', 'argument': '小鹏汽车', "index": [3]},
            {'role': '投资方', 'argument': '小鹏汽车', "index": [3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 370,
        "id": "2dfe37c6829f8195c922d9544f91f591",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '禾赛科技', "index": [0, 2, 3, 4, 6]},
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
            {'role': '融资轮次', 'argument': 'C', "index": [1, 2]},
            {'role': '融资轮次', 'argument': 'D', "index": [2, 4]},
            {'role': '领投方', 'argument': '小米集团', "index": [1]},
            {'role': '领投方', 'argument': '美团', "index": [0, 1]},
            {'role': '领投方', 'argument': 'CPE', "index": [0]},
            {'role': '领投方', 'argument': '百度', "index": [2]},
            {'role': '领投方', 'argument': '百度', "index": [1]},
            {'role': '领投方', 'argument': '光速', "index": [3]},
            {'role': '投资方', 'argument': '光速', "index": [0, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 372,
        "id": "76bcb47ea984950127199d52f9abb69a",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '艾德光子', "index": [0, 3]},
        ]
    },
    {
        "type": "add",
        "idx": 375,
        "id": "074c4699e9a70be7f8223d994c785eb2",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '奈雪的茶', "index": [0, 1, 2, 6, 7, 10, 11, 13, 14]},
            {'role': '融资轮次', 'argument': 'A', "index": [3]},
        ]
    },
    {
        "type": "add",
        "idx": 378,
        "id": "ddf27b7a72960ce1335c039948b1b68c",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '蛮太郎火锅鸡', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 380,
        "id": "3d1b11c82a4f72b9b58b0be556bad3ab",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '复鹄科技', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 382,
        "id": "51e7c5ca8cf4bc89e0b0dc635a71294d",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': 'moody', "index": [0, 1, 2, 3]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2, 4, 6]},
            {'role': '融资轮次', 'argument': 'A', "index": [0]},
            {'role': '领投方', 'argument': 'GGV纪源资本', "index": [0]},
            {'role': '领投方', 'argument': '高瓴创投', "index": [2]},
            {'role': '领投方', 'argument': '经纬中国', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 383,
        "id": "98ed6c4e96f06a2570c841e5aa721c52",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '被投资方', 'argument': '时萃SECRE', "index": [0, 1, 2, 3]},
            {'role': '领投方', 'argument': '远望资本', "index": [0]},
            {'role': '投资方', 'argument': '远望资本', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 384,
        "id": "feb49c00b026d48d981755ac6c76b182",
        "cg": [
            # {'role': '被投资方', 'argument': 'SUGAR Cosmetics', "index": [0, 1, 2, 3, 9, 10]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2, 4, 6]},
            {'role': '融资轮次', 'argument': 'A', "index": [3]},
        ]
    },
    {
        "type": "add",
        "idx": 397,
        "id": "f12eafe6b55249b7774b9f6c2d2c1656",
        "cg": [
            {'role': '被投资方', 'argument': '太空精酿', "index": [0, 1, 2, 5]},
            {'role': '被投资方', 'argument': '猫员外', "index": [0, 1, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [1, 3, 4]},
        ]
    },
    {
        "type": "add",
        "idx": 402,
        "id": "ea6f693be33ca6e3961058b1353c8480",
        "cg": [
            {'role': '被投资方', 'argument': 'moody', "index": [0, 1]},
            {'role': '被投资方', 'argument': '懒熊火锅', "index": [0, 1]},
            {'role': '被投资方', 'argument': '宠物家', "index": [0, 1]},
            {'role': '被投资方', 'argument': '花点时间', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2, 4, 6]},
            {'role': '融资轮次', 'argument': 'A', "index": [2, 3]},
            {'role': '融资金额', 'argument': '亿元', "index": [4, 5]},
        ]
    },
    {
        "type": "add",
        "idx": 403,
        "id": "034c33d802f34114eca3ff869d69d20d",
        "cg": [
            {'role': '被投资方', 'argument': '猎芯', "index": [0, 2, 7]},
        ]
    },
    {
        "type": "add",
        "idx": 406,
        "id": "c2a9cb40f1fc9e418d084c5abf4e12ab",
        "cg": [
            {'role': '被投资方', 'argument': '林清轩', "index": [0, 1, 2]},
            {'role': '领投方', 'argument': '未来资产', "index": [0]},
            {'role': '投资方', 'argument': '未来资产', "index": [0]},
        ]
    },
    {
        "type": "add",
        "idx": 410,
        "id": "64adf01fe1a0ce93dd0b92e591759943",
        "cg": [
            {'role': '被投资方', 'argument': '小鹏汽车', "index": [2]},
        ]
    },
    {
        "type": "add",
        "idx": 415,
        "id": "0f94e3b42a6aa5aa1260ad433ba96b07",
        "cg": [
            {'role': '被投资方', 'argument': '和府捞面', "index": [0, 1, 3, 6]},
            {'role': '融资轮次', 'argument': 'E', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 417,
        "id": "b43ff81606613673085c8daecee8d942",
        "cg": [
            {'role': '被投资方', 'argument': 'Flipkart', "index": [0, 1, 3, 4, 5, 9, 10, 12, 13, 15]},
        ]
    },
    {
        "type": "add",
        "idx": 419,
        "id": "1f0c2659e9f13b9eea59044e27cdd7ce",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0, 2]},
            {'role': '融资轮次', 'argument': 'A', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 420,
        "id": "50cd08db588c67a90c554d76bc198503",
        "cg": [
            {'role': '被投资方', 'argument': '兴盛优选', "index": [0]},
            {'role': '领投方', 'argument': '腾讯', "index": [1]},
        ]
    },
    {
        "type": "add",
        "idx": 421,
        "id": "9e2344f756a8edd62743257aaa79147c",
        "cg": [
            {'role': '被投资方', 'argument': '大师披萨', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 422,
        "id": "07caa94ea496446f1581496f8213a5fc",
        "cg": [
            {'role': '被投资方', 'argument': '乐摇摇', "index": [0, 1]},
        ]
    },
    {
        "type": "add",
        "idx": 423,
        "id": "fc76d0681bb5e3c0b1bc120e4cca85e7",
        "cg": [
            {'role': '被投资方', 'argument': '食得鲜', "index": [0, 1, 2, 10]},
            {'role': '融资轮次', 'argument': 'B', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 427,
        "id": "9d5f9e57a856aeea3308379ba758648b",
        "cg": [
            {'role': '被投资方', 'argument': '食得鲜', "index": [0, 1, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 433,
        "id": "e26dd155e81549575220dad22afdbf53",
        "cg": [
            {'role': '被投资方', 'argument': '帝奥微电子', "index": [0, 1]},
            {'role': '融资轮次', 'argument': 'C', "index": [0, 1]},
            {'role': '领投方', 'argument': '沃衍资本', "index": [0]},
            {'role': '投资方', 'argument': '沃衍资本', "index": [0, 2]},
        ]
    },
    {
        "type": "add",
        "idx": 1010,
        "id": "92df4d967a438df93620424da7f1ead6",
        "cg": [
            {'role': '融资轮次', 'argument': 'B', "index": [0]},
        ]
    }
]

dt_dict = {klist["id"]: klist for klist in dt}


def get_train_data():

    role2id = {'融资金额': 0, '被投资方': 1, '事件时间': 2, '投资方': 3, '披露时间': 4, '融资轮次': 5, '领投方': 6, '估值': 7, '财务顾问': 8}
    role2id_v2 = {'unk': 0}
    for r, rid in role2id.items():
        role2id_v2[r] = rid+1
    label2id = {'O': 0, 'B-0': 1, 'I-0': 2, 'B-1': 3, 'I-1': 4, 'B-2': 5, 'I-2': 6, 'B-3': 7, 'I-3': 8, 'B-4': 9, 'I-4': 10, 'B-5': 11, 'I-5': 12, 'B-6': 13, 'I-6': 14, 'B-7': 15, 'I-7': 16, 'B-8': 17, 'I-8': 18}

    dev_data_size = 150

    rt_data = {
        "bert_data": {
            "train": [],
            "dev": [],
            "all": []
        },
        "crf_data": {
            "train": [],
            "dev": [],
            "all": []
        }
    }

    entity_num = 0
    for ii, doc in enumerate(documents):


        if doc.get("title"):
             text = doc["title"] + "\n" + doc["text"]
        else:
            text = doc["text"]

        event_list = doc["event"]
        # if len(event_list) == 2:
        #     print(text)
        #     print(event_list[0])
        label_data = ["O" for _ in text]
        label_entity = []

        for sub_event in event_list:
            sub_event["arguments"].sort(key=lambda x: x["role"])
            for arg in sub_event["arguments"]:

                if arg["role"] not in role2id:
                    role2id[arg["role"]] = len(role2id)
                    label2id["B-{}".format(role2id[arg["role"]])] = len(label2id)
                    label2id["I-{}".format(role2id[arg["role"]])] = len(label2id)
                if arg["role"] == "披露时间":
                    print(doc["id"], arg["argument"])

                argument = arg["argument"].replace("*", "\*").replace("+", "\+")
                res = re.finditer(argument, text, re.M)
                res_list = [(rs.group(), rs.span()) for rs in res]

                if doc["id"] in dt_dict:
                    for kl in dt_dict[doc["id"]]["cg"]:
                        if kl["role"] == arg["role"] and kl["argument"] == arg["argument"]:
                            # print(kl, doc["id"])
                            res_list = [res_list[sub_index] for sub_index in kl["index"]]

                            break
                for rs in res_list:
                    assert rs[0] == arg["argument"]
                    start, end = rs[1]
                    label_data[start] = "B-{}".format(role2id[arg["role"]])

                    for iv in range(start + 1, end):
                        label_data[iv] = "I-{}".format(role2id[arg["role"]])

                    label_entity.append((start, end, role2id_v2[arg["role"]]))

        label_list_id = [label2id[lb] for lb in label_data]
        start = 0
        for iiv, tt in enumerate(text):
            if tt == "\n":
                if iiv > start:
                    sub_data = {
                        "text": text[start:iiv],
                            "label": label_data[start:iiv],
                            "entity": label_entity
                    }
                    rt_data["crf_data"]["all"].append(sub_data)
                    if ii >= dev_data_size:
                        rt_data["crf_data"]["train"].append(sub_data)
                    else:
                        rt_data["crf_data"]["dev"].append(sub_data)
                start = iiv+1

        if len(text) < 500:

            assert len(text) == len(label_list_id)
            split_word = []
            split_label = []
            for iv, t in enumerate(text):
                if t in split_cha:
                    continue
                split_word.append(t)
                split_label.append(label_list_id[iv])
            sub_data = {
                "text": split_word,
                "label": split_label,
                "entity": label_entity
            }
            rt_data["bert_data"]["all"].append(sub_data)
            if ii >= dev_data_size:
                rt_data["bert_data"]["train"].append(sub_data)
            else:
                rt_data["bert_data"]["dev"].append(sub_data)

        else:
            text_sentence = text.split("\n")
            text_cut = []
            text_cache = []
            cache_len = 0

            for sentence in text_sentence:
                if len(sentence) > 500:
                    continue
                if cache_len + len(text_cache) - 1 + len(sentence) > 500:
                    text_cut.append("\n".join(text_cache))
                    cache_len = 0
                    text_cache = []

                text_cache.append(sentence)
                cache_len += len(sentence)

            # print(len(text))
            if text_cache:
                text_cut.append("\n".join(text_cache))
            if len(text_cut) == 1:
                continue

            last_indx = 0
            for sub_text in text_cut:
                # print(len(sub_text))
                sub_label_list = label_list_id[last_indx:last_indx+len(sub_text)]
                assert len(sub_text) == len(sub_label_list)

                # sub_label_data = label_data[last_indx:last_indx+len(sub_text)]
                split_word = []
                split_label = []
                for iv, t in enumerate(sub_text):
                    if t in split_cha:
                        continue
                    split_word.append(t)
                    split_label.append(sub_label_list[iv])

                sub_data = {
                    "text": split_word,
                    "label": split_label,
                    "label_not_id": label_entity
                }
                rt_data["bert_data"]["all"].append(sub_data)
                if ii >= dev_data_size:
                    rt_data["bert_data"]["train"].append(sub_data)
                else:
                    rt_data["bert_data"]["dev"].append(sub_data)

                # rs = extract_entity(label_data[last_indx:last_indx+len(sub_text)])
                # for rsi in rs:
                #     print(sub_text[rsi[0]:rsi[1]])
                last_indx += len(sub_text)+1

        entity_num += len(label_entity)
    rt_data["label2id"] = label2id
    rt_data["role2id"] = role2id
    rt_data["role2id_v2"] = role2id_v2
    return rt_data


rt_data = get_train_data()

print("document {}".format(len(documents)))
print("valid train {}".format(len(rt_data["bert_data"]["all"])))
print("crf train {}".format(len(rt_data["crf_data"]["all"])))
# print("entity num {}".format(entity_num))
print("label2id {}".format(rt_data["label2id"]))
print("role2id {}".format(rt_data["role2id"]))

with open("D:\data\self-data\\finance_v1.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(rt_data))
