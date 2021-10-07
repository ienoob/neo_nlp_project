#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import json
from bs4 import BeautifulSoup
from pytorch.other_info_extraction.policy_data import item_list
from pytorch.other_info_extraction.policy_information_extractor import Document


labels = [
    {
        "项目": [
            {
                "name": "深化电商应用",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                    "当年网上销售超过3000万元，且年销售增幅在30%、50%、100%的"
                ],
                "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "电子商务专项审计报告（附电子商务销售明细、票据等）"
                ],
                "项目结果": [
                    "给予最高不超过20万元，30万元，50万元的一次性规模提升奖励"
                ]
            },
            {
                "name": "驰名商标认定项目",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                    "经认定“驰名商标”注册企业，电子商务当年网上销售额超过1000万元"
                ],
                "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "2019年度电子商务专项审计报告（附电子商务网上销售明细、票据等）"
                ],
                "项目结果": [
                    "给予年度电商交易额5%的奖励，最高不超过100万元"
                ]
            },
            {
                "name": "电子商务产业园建设项目",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                    "对于统一物业管理，具备电子商务企业运营所需的配套服务设施，且入驻的开展电子商务经营企业达10家以上"
                ],
                "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "园区发展规划和管理制度",
                    "园区管理机构和开园证明材料",
                    "入驻园区企业清单及营业执照复印件",
                    "园区建筑面积证明材料"
                ],
                "项目结果": [
                    "对于统一物业管理，具备电子商务企业运营所需的配套服务设施，且入驻的开展电子商务经营企业达10家以上，使用建筑面积超过5000平方米的电子商务产业园区，给予园区主办方最高不超过10万元的一次性补贴;",
                    "入驻的开展电子商务经营企业达20家以上，使用建筑面积超过1万平方米的电子商务产业园区，给予园区主办方最高不超过30万元的一次性补贴;",
                    "入驻的开展电子商务经营企业达30家以上，使用建筑面积超过1.5万平方米的电子商务产业园区，给予园区主办方最高不超过50万元的一次性补贴。"
                ]
            },
            {
                "name": "促进电子商务行业发展",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                ],
                "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "实际支出凭证",
                    "举办培训活动相关证明材料",
                    "举办会议、展会活动相关证明材料"
                ],
                "项目结果": [
                    "支持行业协会、有资质的社会培训机构或高职院校开展电子商务紧缺人才培训。对用人单位组织输送在职员工（签订劳动合同、交纳社保）参加在本区举办的电子商务紧缺人才培训，按最高不超过企业实际缴纳培训费用的50%予以补助，最高不超过20万元。"
                ]
            },
            {
                "name": "大宗生产资料市场开展电子商务项目",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                    "年交易额首次达到300亿元"
                ],
                "申报材料": [
                    "2019年度电子商务专项审计报告（附电子商务交易明细）"
                ],
                "项目结果": [
                    "对于形成一定市场规模和行业影响力的大宗生产资料市场开展电子商务的，年交易额首次达到300亿元、500亿元、1000亿元的，分别给予最高不超过30万元、50万元（含前档）、100万元（含前档）奖励。"
                ]
            },
            {
                "name": "促进电子商务代理服务企业发展",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                    "对本区电子商务代理服务企业年代理服务费收入达1000万"
                ],
                "申报材料": [
                    "代理服务企业证明材料（企业清单，合同或协议）",
                    "2019年电子商务专项审计报告（附电子商务代理服务费明细、票据等）"
                ],
                "项目结果": [
                    "对本区电子商务代理服务企业年代理服务费收入达1000万，2000万以上的企业，分别给予不超过10万元，20万元（含前档）奖励。"
                ]
            },
            {
                "name": "电商会议、展会项目",
                "condition": [
                    "申报单位须具有独立法人资格，注册地在姑苏区范围内，持续经营且税务关系隶属于姑苏区。",
                    "业务经营正常，近三年无严重失信行为",
                ],
                "申报材料": [
                    "实际支出凭证",
                    "举办会议、展会活动相关证明材料"
                ],
                "项目结果": [
                    "在区内举办的高水平电子商务领域专题会议、专业展会，给予承办机构专项经费支持，最高不超过10万元。"
                ]
            }
        ]
    }
]

d = {
    0: [
        {
            "项目名称": "深化电商应用",
            "企业注册地区": "姑苏区",
            "企业营业额": "超过3000万元",
            "企业年销售增幅": ["30%", "50%", "100%"],
            "项目奖励": "最高不超过20万元，30万元，50万元的一次性规模提升奖励",
            "申报材料": ["项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                     "电子商务专项审计报告（附电子商务销售明细、票据等）"],
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "驰名商标认定项目",
            "企业注册地区": "姑苏区",
            "企业营业额": "超过1000万元",
            "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "2019年度电子商务专项审计报告（附电子商务网上销售明细、票据等）"
                ],
            "项目奖励": "年度电商交易额5%的奖励，最高不超过100万元",
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "电子商务产业园建设项目",
            "企业注册地区": "姑苏区",
            "申报材料": [
                "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                "园区发展规划和管理制度",
                "园区管理机构和开园证明材料",
                "入驻园区企业清单及营业执照复印件",
                "园区建筑面积证明材料"
            ],
            "项目奖励": [
                "园区主办方最高不超过10万元的一次性补贴",
                "园区主办方最高不超过30万元的一次性补贴",
                "园区主办方最高不超过50万元的一次性补贴"],
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "促进电子商务行业发展",
            "企业注册地区": "姑苏区",
            "申报材料": [
                    "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                    "实际支出凭证",
                    "举办培训活动相关证明材料",
                    "举办会议、展会活动相关证明材料"
            ],
            "项目奖励": [
                "最高不超过企业实际缴纳培训费用的50%予以补助，最高不超过20万元"
            ],
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "大宗生产资料市场开展电子商务项目",
            "企业注册地区": "姑苏区",
            "申报材料": [
                "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                "2019年度电子商务专项审计报告（附电子商务交易明细）"
            ],
            "项目奖励": [
                "最高不超过30万元、50万元（含前档）、100万元（含前档）奖励"
            ],
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "促进电子商务代理服务企业发展",
            "企业注册地区": "姑苏区",
            "申报材料": [
                "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                "代理服务企业证明材料（企业清单，合同或协议）",
                "2019年电子商务专项审计报告（附电子商务代理服务费明细、票据等）"
            ],
            "项目奖励": [
                "不超过10万元，20万元（含前档）奖励"
            ],
            "申报时间截止时间": "2021年9月30日"
        },
        {
            "项目名称": "电商会议、展会项目",
            "企业注册地区": "姑苏区",
            "申报材料": [
                "项目资金申请表、项目承诺书、营业执照、2019年度财务报表、纳税证明及可以说明项目的其他证明材料",
                "实际支出凭证",
                "举办会议、展会活动相关证明材料"
            ],
            "项目奖励": [
                "最高不超过10万元"
            ],
            "申报时间截止时间": "2021年9月30日"
        }
    ],
    1: [
        {
            "项目名称": "2021年苏州市商务发展专项资金项目（第一批）",
            "企业注册地址": "江苏省",
            "企业资质": ["三星级、四星级上云企业"],
            "申报时间起始时间": "9月3日",
            "申报时间截止时间": "9月30日"
        }
    ],
    2:[
        {
            "项目名称": "2021年度第二批省星级上云企业",
            "企业注册地址": "苏州工业园区",
            "企业性质": ["企事业单位", "社会团体", "民办非企业"],
            "申报时间截止时间": ["2021年6月30日", "7月2日"]
        }
    ],
    3: [
        {
            "项目名称": "重点产业技术创新",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        },
        {
            "项目名称": "农业科技创新",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        },
        {
            "项目名称": "社会发展科技创新",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        },
        {
            "项目名称": "医疗卫生科技创新",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        },
        {
            "项目名称": "柔性引进海外人才智力“海鸥计划”",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        },
        {
            "项目名称": "产学研协同创新",
            "企业注册地址": "苏州市",
            "企业性质": ["科技"],
            "申报时间截止时间": "申报截止日期当日的17:00"
        }
    ],
    4: [
        {
            "申报时间截止时间": "2021年7月16日24时整"
        }
    ],
    5: [
        {
            "研发人数": "不低于 100 人",
            "研发经费": "不低于3000万元"
        }
    ],
    6: [
        {
            "企业性质": ["软件企业"],
            "申报时间截止时间": "2021年8月13日前",
        }
    ],
    7: [
        {
            "申报材料": ["《申报指南》中明确的申报资料"],
            "企业注册地址": "昆山市",
            "企业性质": ["工业企业"],
            "申报时间截止时间": ["项目线上注册申报截止时间为2021年7月16日下午17:00", "相关材料线上上传及纸质材料线下报送截止时间为2021年8月17日"]
        }
    ],
    8: [
        {
            "项目名称": "企业上市挂牌奖励",
            "申报材料": [
                "吴中区企业上市、挂牌奖励申请表(附件一)",
                "营业执照复印件"
                "已股改:国家企业信息信用公示系统市场主体类型变更记录截图，与三方机构签订协议复印件",
                "进入辅导期:江苏证监局受理函复印件",
                "材料已受理:申报获中国证监会或证券交易所受理证明复印件",
                "已上市:证监会同意首次公开发行股票注册的批复复印件，企业首发募集资金金额证明文件",
                "近三年企业征信报告",
                "企业上市挂牌奖励已申请金额汇总表",
                "其他需要补充的材料"
            ],
            "企业资质": [
                "拟上市公司",
                "上市公司",
                "保荐券商"
            ],
            "申报时间起始时间": "5月31日",
            "申报时间截止时间": "6月3日",
            "企业注册地址": "吴中区",
            "项目奖励": [
                "奖励100万元(其中镇区承担50万元)",
                "奖励100万元(其中镇区承担50万元)",
                "奖励400万元(其中镇区承担200万元)"
            ]
        },
        {
            "项目名称": "保荐券商奖励",
            "申报材料": [
                "吴中区保荐券商奖励申请表(附件二)",
                "保荐券商营业执照复印件"
                "证监会同意首发股票注册的批复复印件",
                "其他需要补充的材料"
            ],
        }
    ],
    9: [
        {
            "项目名称": "2021年领军项目平台使用补贴",
            "申报时间截止时间": "2021-8-15中午12:00",
            "项目奖金\补贴": ["两年期总额不超过50万元的平台使用补贴"]
        }
    ],
    10: [
        {
            "项目名称": "集成创新类项目",
            "企业注册地址": "江苏省",
            "项目奖金\补贴": "省财政补助资金50万元左右",
            "企业性质": ["农机（农业）企业"],
            "申报时间截止时间": ["7月9日17:30", "7月12日17:30"]
        },
        {
            "项目名称": "推广应用类项目",
            "企业注册地址": "江苏省",
            "项目奖金\补贴": "县省财政补助资金100万元左右，2008类项目省财政补助资金150万元左右",
            "企业性质": ["农机（农业）企业"],
            "申报时间截止时间": ["7月9日17:30", "7月12日17:30"]
        }
    ],
    11: [],
    12: [
        {
            "项目名称": "2021年第二批星级上云企业申报相关工作的通知",
            "申报时间起始时间": "9月3日",
            "申报时间截止时间": "10月8日"
        }

    ],
    13: [
        {
            "网上申报时间": "2021年7月12日",
            "纸质材料报送截止时间": "2021年8月6日"
        }
    ],
    14: [
        {
            "企业注册地址": "苏州市",
            "申报时间截止时间": "申报截止日期当日的17:00"

        }
    ],
    15: [
        {
            "企业注册地址": "苏州工业园区",
            "企业性质": ["科技型企业",
                     "各级领军人才企业",
                     "获得科技成果转化等科技计划项目企业",
                     "经认定的科技型自主品牌培育企业",
                     "高新技术企业",
                     "纳米技术、生物医药、人工智能等园区重点发展领域内的企业"],
            "项目奖金\补贴": [
                "贷款贴息：对企业获取融风科贷、银行等金融机构提供的各类科技信贷产品，给予企业一定比例的贴息支持。对单家企业信贷产品年度融资的利息补贴不超过50万元",
                "科技保险费补贴：企业购买经园区科技金融服务平台备案的科技保险，对企业支出的保险费给予补贴。每年每家企业科技保险费补贴总额不超过10万元",
                "融资担保费补贴：对企业的融资担保费或参与金融创新产品支付的融资担保费给予50%补贴，每年每家企业补贴总额不超过10万元",
                "融资租赁费用补贴：对企业支付的融资租赁费用或企业参与金融创新产品支付的融资租赁费用，按同期贷款市场报价利率（LPR）的50%给予补贴，每年每家企业补贴总额不超过10万元"
            ]
        }
    ],
    16: [],
    17: [{
        "申报时间起始时间": "2020年1月1日",
        "申报时间截止时间": "12月31日"
    }],
    18: [
        {
            "企业注册地址": "苏州市"
        }
    ],
    19: [],
    20: [],
    22: [
        {"项目名称": "优秀版权奖",
         "企业注册地址": "昆山市"},
        {"项目名称": "昆山市版权示范单位",
         "企业注册地址": "昆山市"}
    ],
    28: [
        {"项目名称": "330101科技合作与交流"},
        {"项目名称": "330102“技联苏州日高校”产学研合作项目后补助",
         "企业注册地址": "苏州"}
    ]
}

if __name__ == "__main__":
    path = "D:\data\政策信息抽取\\text"
    file_list = os.listdir(path)
    print(len(file_list))

    doc = Document()
    for file in file_list[91:]:
        print(file)
        file_path = path + "\\" + file
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        doc.parse_content_v3(data)
        doc.display_document()
        res = doc.parse_half_struction()
        print(json.dumps(res, ensure_ascii=False, indent=4))

        doc.parse_precision(res["conditions"])
        doc.parse_contact_info()

        for k_project in res["project_infos"]:
            print("项目名称", k_project["project_name"])
            doc.parse_precision(k_project["conditions"])


        break

    # doc = Document()
    # # doc.parse_content(content2)
    # # doc.extract_zb()
    # # res = mrc("服务收入占比", "1、申报主体需获评省级服务型制造示范企业，服务收入占企业营业收入比重达30%以上；")
    # # print(res)
    #
    # # for item in item_list[3:]:
    # #     file_name = "D:\data\\政策信息抽取\\{}.txt".format(item[1])
    # #
    # #     with open(file_name, "rb") as f:
    # #         data = f.read()
    # #
    # #     soup = BeautifulSoup(data, 'html.parser')
    # #     doc.parse_content(soup.text)
    # #     doc.parse_root()
    #
    # file_name = "D:\data\政策信息抽取\\text\关于组织开展2021年太仓市小巨人企业申报推荐工作的通知.txt"
    # with open(file_name, "r", encoding="utf-8") as f:
    #     data = f.read()
    # doc.parse_content(data)
    # doc.display_document()
    # res = doc.extract_zb()
    # for target_info in res:
    #     print(json.dumps(target_info, indent=4, ensure_ascii=False))
        # doc.display_document()
        # for res in doc.content:
        #     print()
        # res = doc.extract_zb()
        # for target_info in res:
        #     print(json.dumps(target_info, indent=4, ensure_ascii=False))

    # print(json.dumps({"name": "分类与排序 "}, indent=4, ensure_ascii=False))
