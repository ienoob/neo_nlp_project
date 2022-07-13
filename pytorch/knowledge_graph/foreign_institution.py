#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/18 10:23
    @Author  : jack.li
    @Site    : 
    @File    : foreign_institution.py

"""
institution_list = {
    "ACA Investments":
        {
            "id": 0,
            "name": "ACA Investments",
            "alias": ["ACA"],
            "desc": "ACA Investments provides Investment related services, including advisory services for mergers and acquisitions.ACA Investments提供与投资相关的服务，包括并购咨询服务。  ",
            "area": "新加坡"
        },
    "3M Ventures":
        {
            "id": 1,
            "name": "3M Ventures",
            "alias": ["3M"],
            "desc": "",
            "area": ""
        },
    " ADV Capital Management Co., Ltd.":
        {
            "id": 2,
            "name": " ADV Capital Management Co., Ltd.",
            "alias": ["ADV"],
            "desc": "ADV是一个耐心的风险投资引擎，旨在支持一代定义的数字技术公司。 ADV is a patient venture investment engine. Its team are entrepreneurs and operators who have learnt the hard lessons and want to pay them forward. Currently investing £150M, ADV takes the long view of business building, investing across the funding lifecycle of startups, scaleups and ‘scalebigs’. ADV champions the innovators - the people who build complex, technical, generation-defining businesses. ADV’s investors are British Business Bank, Legal & General and Woodford Investment Management. ADV为开曼群岛获豁免有限责任合伙，主要从事全球技术、媒体、电信行业、物联网、车联网、人工智能、机器人、智能硬件、生物行业及医药行业投资。",
            "area": "开曼群岛"
        },
    "The Aes Corporation":
        {
            "id": 3,
            "name": "The Aes Corporation",
            "alias": ["AES"],
            "desc": "爱依斯电力公司于1981年成立，是一家投资控股公司，通过其子公司和附属公司，经营一个地理上分散的发电和配电业务组合。它经营两种业务：发电业务和公用事业业务。发电业务方面，它经营发电厂，向大功率用户（如公用事业及其他中介机构）出售电力。在公用事业方面，它经营公共设施，向特定的服务区域内的住宅、商业、工业和政府部门的终端用户客户生成、分配、传输和出售电力。爱依斯电力公司是由六个市场为导向的战略业务单位（ “事业部” ）的多元化发电和电力公司：美国（美国） ，安第斯（智利，哥伦比亚和阿根廷） ，巴西， MCAC （墨西哥，中美洲和加勒比） ， EMEA（欧洲，中东和非洲）和亚洲。",
            "area": "美国"
        },
    "佳兆業集團有限公司":
        {
            "id": 4,
            "name": "佳兆業集團有限公司",
            "alias": ["佳兆业"],
            "desc": "",
            "area": "香港"
        },
    "Uber":
        {
            "id": 5,
            "name": "Uber",
            "alias": ["优步", "Uber"],
            "desc": "Uber（Uber Technologies,Inc.）中文译作“优步”，是一家美国硅谷的科技公司。Uber在2009年，由加利福尼亚大学洛杉矶分校辍学生特拉维斯·卡兰尼克和好友加勒特·坎普（Garrett Camp）创立。因旗下同名打车APP而名声大噪。Uber已经进入中国大陆的60余座城市，并在全球范围内覆盖了70多个国家的400余座城市 [1]  。",
            "area": "美国"
        },
    "AMF":
        {
            "id": 5,
            "name": "AMF",
            "alias": ["AMF"],
            "desc": "AMF 管理着超过 6500 亿瑞典克朗资产，是瑞典领先的养老金公司之一。",
            "area": "瑞典"
        },
    "APG":
        {
            "id": 6,
            "name": "APG",
            "alias": ["APG"],
            "desc": "荷兰养老金资产管理公司APG 是全球最大的房地产投资者之一。",
            "area": "荷兰"
        },
    "ATW Partners LLC":
        {
            "id": 7,
            "name": "ATW Partners LLC",
            "alias": ["ATW"],
            "desc": "ATW Partners is a New York-based hybrid venture capital/private equity firm that focuses on investing in high quality companies at various stages. Investors choose ATW for our unique and disciplined investment process targeting high risk-adjusted returns. We invest in debt and equity and offer investment flexibility to our portfolio companies with tailored investment structure solutions.",
            "area": "美国"
        },
    "Amazon":
        {
            "id": 8,
            "name": "Amazon",
            "alias": ["AWS", "AMZN"],
            "desc": "亚马逊公司（Amazon，简称亚马逊；NASDAQ：AMZN），是美国最大的一家网络电子商务公司，位于华盛顿州的西雅图。是网络上最早开始经营电子商务的公司之一，亚马逊成立于1994年， [12]  一开始只经营网络的书籍销售业务，现在则扩及了范围相当广的其他产品，已成为全球商品品种最多的网上零售商和全球第二大互联网企业，在公司名下，也包括了AlexaInternet、a9、lab126、和互联网电影数据库（Internet Movie Database，IMDB）等子公司。",
            "area": "美国"
        },
    "Blue Mountain Labs":
        {
            "id": 9,
            "name": "Blue Mountain Labs",
            "alias": ["BML"],
            "desc": "Blue Mountain Labs 区块链专业基金，专注于赋能区块链高级项目，和区块链知识的普及。",
            "area": "国外"
        },
    "BTG International Asia Limited":
        {
            "id": 10,
            "name": "BTG International Asia Limited",
            "alias": ["BTG"],
            "desc": "英国技术集团（BTG Plc）的子公司英国技术集团国际亚洲有限公司（BTG International Asia Limited）",
            "area": "英国"
        },
    ".406 Ventures":
        {
            "id": 11,
            "name": ".406 Ventures",
            "alias": [".406 Ventures"],
            "desc": "406 Ventures 成立于2005年，是一家早期风险投资机构。",
            "area": "美国"
        },
    "01 Advisors":
        {
            "id": 12,
            "name": "01 Advisors",
            "alias": ["01 Advisors"],
            "desc": "01 Advisors是一家风险与咨询公司，可帮助创始人从开发产品到建立公司。",
            "area": "美国"
        },
    "021 Capital":
        {
            "id": 13,
            "name": "021 Capital",
            "alias": ["021 Capital"],
            "desc": "021 Capital 是一家风险投资公司，主要投资生物技术、农业技术和互联网。",
            "area": "印度"
        },
    "0331创投":
        {
            "id": 14,
            "name": "0331创投",
            "alias": ["0331创投"],
            "desc": "0331创投是卓航国际教育集团投资的全资子公司，专注于投资消费升级领域，投资了40多个项目，如仔皇煲、台资味外卖、新佳宜便利超市、天然工坊、众益传媒、R2game等。0331创投也是薛蛮子“蛮有趣”基金创始合伙人，与薛蛮子在天使投资领域保持紧密合作。  ",
            "area": "中国"
        },
    "创客壹佰科技孵化器（北京）有限公司":
        {
            "id": 16,
            "name": "创客壹佰科技孵化器（北京）有限公司",
            "alias": ["创客壹佰科技孵化器（北京）有限公司", "创客100基金"],
            "desc": "创客100基金是由知名媒体人曹健先生创办，旗下管理的有人民币基金和美元基金，基金规模合计为5000万美元左右。主要投资B轮以前的TMT领域的创业项目。LP大部分来自国内和硅谷的知名企业和知名企业家。创客100基金对所投项目不仅仅体现在资本上，更多的是利用背后LP在互联网流量和技术创新优势帮助创业项目。旗下拥有的媒体平台也是创业项目所看中的不可多得的重要资源。",
            "area": "中国"
        },
    "10100 Fund":
        {
            "id": 17,
            "name": "10100 Fund",
            "alias": ["10100 Fund", "10100 fund"],
            "desc": "10100 fund focuses on real estate, e-commerce, and emerging tech companies.",
            "area": "美国"
        },
    "120 Capital, L.P.":
        {
            "id": 18,
            "name": "120 Capital, L.P.",
            "alias": ["120 Capital, L.P.", "120 Capital"],
            "desc": "投资基金为一家依据美国特拉华州法律成立的有限合伙企业，主要专注于投资位于中国（包含中国大陆、香港特别行政区、台湾地区）、北美和欧洲地区的医疗健康和生命科学企业",
            "area": "美国"
        },
    "122 WEST VENTURES":
        {
            "id": 19,
            "name": "122 WEST VENTURES",
            "alias": ["122 WEST VENTURES", "122West"],
            "desc": "122WEST is a venture capital firm focused on early-stage internet and software investments in the San Francisco Bay Area.",
            "area": "美国"
        },
    "1835i Ventures":
        {
            "id": 20,
            "name": "1835i Ventures",
            "alias": ["1835i Ventures"],
            "desc": "1835i Ventures 是澳新银行的企业风险投资部门。",
            "area": "澳大利亚"
        },
    "THIEVES, LLC":
        {
            "id": 21,
            "name": "THIEVES, LLC",
            "alias": ["THIEVES, LLC", "100 Thieves"],
            "desc": "100 Thieves是一家美国电竞公司，主要业务包括职业电竞、潮流服装设计和生产等，旗下的电竞战队参与的项目包括：《英雄联盟》、《使命召唤》、《堡垒之夜》、《皇室战争》。中国电竞选手Cody Sun便在其《英雄联盟》战队阵容之中。  ",
            "area": "美国"
        },
    "1030am.com":
        {
            "id": 22,
            "name": "1030am.com",
            "alias": ["1030am.com", "1030am"],
            "desc": "1030am.com，2015年6月在新加坡成立，致力于将高品质的品牌产品带给新加坡本地的年轻客户。主营产品包括品牌包包，香水，化妆品，家居生活用品等。新加坡是个小市场，电商的发展也相对比较滞后。1030am.com是新加坡电商的新秀，网站仅仅上线了四个月就完成了PreA轮的投资，说明投资人还是很看好这个商业模式以及团队的执行力。据1030am的管理层透露，315万新币（合计约1500万人民币）的投资款已经到账，投资人都是新加坡本地著名的投资者，这笔资金将用于新加坡，马来西亚，印尼市场的拓展。",
            "area": "新加坡"
        },
    "Group14 Technologies":
        {
            "id": 22,
            "name": "Group14 Technologies",
            "alias": ["Group14 Technologies"],
            "desc": "Group14 Technologies于2015年在华盛顿州成立，是总部位于西雅图的EnerG2的子公司。Group14的关键技术，是一种可以替代或增强石墨阳极的硅碳粉末。",
            "area": "美国"
        },
    "1906newhighs":
        {
            "id": 23,
            "name": "1906newhighs",
            "alias": ["1906newhighs"],
            "desc": "1906是大麻行业最具创新性的品牌之一，创造了具有突破意义的大麻和植物药功能性配方。1906支持药用大麻研究，并将其收入的一部分投资于临床试验和教育医疗保健行业的从业人员。",
            "area": "美国"
        },
    "1Huddle":
        {
            "id": 24,
            "name": "1Huddle",
            "alias": ["1Huddle"],
            "desc": "总部位于新泽西州纽瓦克的劳动力科技公司",
            "area": "美国"
        },
    "1Password":
        {
            "id": 25,
            "name": "1Password",
            "alias": ["1Password"],
            "desc": "总部位于加拿大多伦多的安全公司.因此，在首席执行官Jeff Shiner的领导下，1Password提供了一个密码管理和凭据安全平台。1Password于2015年开始作为消费者密码管理应用程序推出，为企业构建产品，允许团队安全地共享和管理密码。2016年5月，1Password的商业版本与其他几个工作平台集成推出，Shiner表示，目前Slack和IBM等超过10万家公司都在使用它。",
            "area": "加拿大"
        },
    "1MG Technologies Pvt. Ltd.":
        {
            "id": 26,
            "name": "1MG Technologies Pvt. Ltd.",
            "alias": ["1MG Technologies Pvt. Ltd.", "1mg"],
            "desc": "1mg是一家印度智能医药电商平台，推出了智能处方、自定义健康食谱、My Health Feed历史治疗记录等数字健康产品，借助智能处方，用户可以了解到医生所开药物的信息，并且了解这些药物的副作用及潜在的药物反应。  ",
            "area": "印度"
        },
    "2C2P Pte. Ltd.":
        {
            "id": 27,
            "name": "2C2P Pte. Ltd.",
            "alias": ["2C2P Pte. Ltd.", "2C2P", "2C2P Pte"],
            "desc": "2C2P 成立于2008年，是一家主打东南亚市场的第三方支付服务商，2013年处理了超过5亿美元的在线支付，年收入超过200万美元。 ",
            "area": "新加坡"
        },
    "305 Fitness, Inc.":
        {
            "id": 28,
            "name": "305 Fitness, Inc.",
            "alias": ["305 Fitness, Inc.", "305 Fitness"],
            "desc": "305 Fitness是一家美国精品舞蹈健身连锁品牌提供商，305 Fitness 在纽约有4家工作室，在波士顿和华盛顿特区各有1家工作室，另外在芝加哥、洛杉矶和旧金山设有快闪工作室。每节课55分钟，价格为26到34美元不等。课程有三种选择，包括有氧舞蹈课程 305 Cardio、柔韧性和力量调节课程 305 Flx 、高强度力量训练 305 PWR ，其中 305 Cardio 是品牌的核心课程。",
            "area": "美国"
        },
    "360Learning":
        {
            "id": 28,
            "name": "360Learning",
            "alias": ["360Learning"],
            "desc": "360Learning是企业学习的学习参与平台：高度吸引培训师，专家和学习者。",
            "area": "法国"
        },
    "3rdFlix Visual Effects":
        {
            "id": 28,
            "name": "3rdFlix Visual Effects",
            "alias": ["3rdFlix Visual Effects"],
            "desc": "3rdFlix Visual Effects是印度一家智能教育科技服务商，这家软件即服务(SaaS)公司利用视觉效果、人工智能(AI)、增强现实(AR)和虚拟现实(VR)等技术，为提升学习效果创造逼真的体验。该公司现在主打一款名为Corsalite的人工智能学习平台，并计划在今年年底前推出另一款名为3rdFlix的沉浸式内容产品。",
            "area": "印度"
        },
    "4G Capital":
        {
            "id": 28,
            "name": "4G Capital",
            "alias": ["4G Capital"],
            "desc": "4G Capital 正在通过流动资金贷款提供金融知识培训，以帮助小企业实现可持续发展。",
            "area": "乌干达"
        },
    "龍光地產控股有限公司":
        {
            "name": "龍光地產控股有限公司",
            "alias": ["龍光地產控股有限公司", "龙光地产控股有限公司"],
            "desc": "龙光地産控股有限公司是两广一体化建筑开发商,旗下拥有广州、深圳、汕头、佛山、南宁、成都、海南陵水等二十多家下属公司。根据中国指数研究院数据,公司于2012年在中国房地産开发企业排行第46位(以销售额计)。",
            "area": "中国香港"
        },
    "5B AUSTRALIA PTY LTD":
        {
            "name": "5B AUSTRALIA PTY LTD",
            "alias": ["5B AUSTRALIA PTY LTD", "5B"],
            "desc": "5B是一家澳大利亚太阳能发电解决方案提供商，提供了创新的模块化“即插即用”太阳能发电解决方案，旨在改善利用太阳能的发电系统，致力于使太阳能项目的成本更低、进展速度更快，并且希望使能源的利用过程更加智能。",
            "area": "澳大利亚"
        },
    "黑石集团":
        {
            "name": "黑石集团",
            "alias": ["黑石集团", "黑石生命科学"],
            "desc": "黑石集团，总部位于美国纽约，是美国规模最大的上市投资管理公司，也是世界知名的顶级投资公司，美国规模最大的上市投资管理公司， [15]  1985年由彼得·彼得森(Peter G. Peterson)和史蒂夫·施瓦茨曼(Steve Schwarzman)共同创建。2007年6月22日在纽约证券交易所挂牌上市（NYSE：BX） [1]  。2011年9月，黑石集团首次撤资中国房地产。",
            "area": "美国"
        },
    "麦子钱包MathWallet":
        {
            "name": "麦子钱包MathWallet",
            "alias": ["麦子钱包MathWallet", "麦子钱包", "MathWallet"],
            "desc": "麦子钱包MathWallet是一个多平台跨链钱包，产品包括 App 钱包、网页钱包、浏览器插件钱包、硬件钱包等，且支持 20 多个公链，去中心化的跨链交易，构建了一个多链的 dApp 生态系统，并参与多个 PoS 公链的节点生态",
            "area": "瑞士"
        },
    "WhaleEx 鲸交所":
        {
            "name": "WhaleEx 鲸交所",
            "alias": ["鲸交所", "WhaleEx 鲸交所"],
            "desc": "鲸交所是全球最大的去中心化服务平台之一，创造了不丢币﹑0佣金﹑0上币费﹑没有假币和秒级提现等独特的用户价值，旨在用区块链技术为加密资产行业带来透明度和信任，帮助用户一站式配置全球加密资产。区块链的核心价值就是从 don't be evil（不作恶）到 can't be evil（无法作恶），鲸交所的使命就是 can't be evil（无法作恶），用区块链技术建立一个集体见证不可篡改的可信价值交换网络。鲸交所的愿景是Everything Exchange。鲸交所自主研发了跨链技术，支持比特币﹑以太坊等多种跨链资产，支持法币交易。鲸交所还拥有鲸矿池﹑鲸算力比特币挖矿﹑鲸借贷﹑鲸理财﹑鲸钱包﹑鲸定投等全方位金融场景，也即将开通期货﹑期权﹑永续合约等衍生品服务。",
            "area": "新加坡"
        },
    "香港苏宁易购有限公司":
        {
            "name": "香港苏宁易购有限公司",
            "alias": ["香港苏宁易购有限公司", "香港蘇寧易購有限公司"],
            "desc": "香港苏宁易购有限公司，成立于2010-01-21，法定代表人为YAOHUA XU，经营状态为仍注册  ",
            "area": "香港"
        },
    "马可波罗协议":
        {
            "name": "马可波罗协议",
            "alias": ["马可波罗协议"],
            "desc": "马可波罗协议是一家点对点电子现金系统，旨在通过一系列技术机制创建新的点对点现金系统基础架构，以实现公共区块链TPS共享和智能调度，从而成为整个区块链世界的交易调度平台。点对点现金系统应用程序可以利用TPS的无限扩展和最佳GAS费用。  ",
            "area": "海外"
        },
    "皮尔法伯集团":
        {
            "name": "皮尔法伯集团",
            "alias": ["皮尔法伯集团", "馥绿德雅"],
            "desc": "法国PIERRE FABRE皮尔法伯制药集团总部位于法国西南部的卡斯特，其制药及化妆品业务范围遍布全球五大洲，130余个国家，拥有逾1400名研发人员。",
            "area": "法国"
        },
    "金米技术有限公司":
        {
            "name": "金米技术有限公司",
            "alias": ["金米技术有限公司", "香港金米", "香港金米技术公司"],
            "desc": "香港金米是一家数字化跨境供应链服务提供商，采用S2B2C模式，利用大数据、人工智能等技术提供数字化的跨境供应链服务，赋能中小商家，帮助更多中国中小零售企业参与进口贸易，帮助他们构建私域流量，将更多小众、优质进口产品以更加灵活的销售方式触达消费者。  ",
            "area": "香港"
        },
    "荷兰皇家飞利浦公司":
        {
            "name": "荷兰皇家飞利浦公司",
            "alias": ["飞利浦", "荷兰皇家飞利浦公司"],
            "desc": "飞利浦，1891年成立于荷兰，主要生产照明、家庭电器、医疗系统方面的产品。飞利浦公司，2007年全球员工已达128,100人，在全球28个国家有生产基地，在150个国家设有销售机构，拥有8万项专利，实力超群。2011年7月11日，飞利浦宣布收购奔腾电器（上海）有限公司，金额约25亿元。 ",
            "area": "荷兰"
        },
    "领创集团":
        {
            "name": "领创集团",
            "alias": ["领创集团", "Advance Intelligence Group"],
            "desc": "领创集团成立于2016年，致力于搭建一个人工智能化、信用为基础的市场生态系统，打造服务企业、消费者和商户的智慧科技生态圈 ",
            "area": "新加坡"
        },
    "Yahoo!":
        {
            "name": "Yahoo!",
            "alias": ["Yahoo!", "雅虎"],
            "desc": "雅虎（Yahoo!，NASDAQ：YHOO）是美国著名的互联网门户网站，也是20世纪末互联网奇迹的创造者之一。其服务包括搜索引擎、电邮、新闻等，业务遍及24个国家和地区，为全球超过5亿的独立用户提供多元化的网络服务。同时也是一家全球性的因特网通讯、商贸及媒体公司。雅虎是全球第一家提供因特网导航服务的网站，总部设在美国加州圣克拉克市，在欧洲、亚太区、拉丁美洲、加拿大及美国均设有办事处。",
            "area": "美国"
        },
    "15Five":
        {
            "name": "15Five",
            "alias": ["15Five"],
            "desc": "15Five是一家持续绩效管理解决方案提供商，开发了基于云计算的软件来评估员工的表现，并将OKR、脉搏调查、同伴认可和反馈结合在一起，帮助团队提高。",
            "area": "美国"
        },
    "66° NORTH":
        {
            "name": "66° NORTH",
            "alias": ["66° NORTH", "66NORTH"],
            "desc": "66° NORTH——是冰岛最古老的制造企业之一。自1926年成立以来一直秉持着在极寒之地给你温暖的品牌理念（Keeping Iceland warm since 1926）。作为极寒之地服饰的长期领先创新者，在过去几十年中，一直受到冰岛人民的喜爱。在气候条件极度恶劣的情况下，几乎每个冰岛人都会选择依靠66° NORTH的服装御寒，而它极强的实用性和设计感也从来不会让大家失望。如此以来，称其为国民品牌也不为过。",
            "area": "冰岛"
        },
    "6Sense":
        {
            "name": "6Sense",
            "alias": ["6Sense", "6sense"],
            "desc": "6Sense是一家大数据营销服务商，专注于B2B企业营销分析的智能预测平台，挖掘与分析网络上潜在客户产生的相关数据，来预测销售线索，准确率高达85%。  ",
            "area": "美国"
        },
    "通用汽车公司（General Motors Company)":
        {
            "name": "通用汽车公司（General Motors Company)",
            "alias": ["通用汽车公司", "通用汽车", "通用"],
            "desc": "通用汽车公司（General Motors Company，GM）成立于1908年9月16日，自从威廉·杜兰特创建了美国通用汽车公司以来，通用汽车在全球生产和销售包括别克、雪佛兰、凯迪拉克、GMC及霍顿等一系列品牌车型并提供服务。2014年，通用汽车旗下多个品牌全系列车型畅销于全球120多个国家和地区，包括电动车、微车、重型全尺寸卡车、紧凑型车及敞篷车。 ",
            "area": "美国"
        },
    "谷歌公司（Google Inc.）":
        {
            "name": "谷歌公司（Google Inc.）",
            "alias": ["谷歌公司（Google Inc.）", "谷歌风投", "谷歌资本", "谷歌母公司字母表", "谷歌旗下风投GV", "谷歌助理投资计划", "谷歌CapitalG", "谷歌"],
            "desc": "谷歌公司（Google Inc.）成立于1998年9月4日，由拉里·佩奇和谢尔盖·布林共同创建，被公认为全球最大的搜索引擎公司 [1]  。谷歌是一家位于美国的跨国科技企业，业务包括互联网搜索、云计算、广告技术等，同时开发并提供大量基于互联网的产品与服务，其主要利润来自于AdWords等广告服务 [2]  。",
            "area": "美国"
        },
    "苹果公司（Apple Inc. ）":
        {
            "name": "苹果公司（Apple Inc. ）",
            "alias": ["苹果公司（Apple Inc. ）", "苹果公司", "苹果"],
            "desc": "苹果公司（Apple Inc. ）是美国一家高科技公司。苹果营收达到3658亿美元， [169]  由史蒂夫·乔布斯、斯蒂夫·盖瑞·沃兹尼亚克和罗纳德·杰拉尔德·韦恩（Ron Wayne）等人于1976年4月1日创立，并命名为美国苹果电脑公司（Apple Computer Inc.），2007年1月9日更名为苹果公司，总部位于加利福尼亚州的库比蒂诺。",
            "area": "美国"
        },
    "龙门资本(老猫)":
        {
            "name": "龙门资本(老猫)",
            "alias": ["龙门资本"],
            "desc": "对于未来基金的发展，老猫表示除了部分自有资产以及一些客户出于信任继续投资的资产，已经由龙门资本（longmen.fund）来进行管理，已经开始在做合规，等到合规完成之前，不会接受任何其他人的参与，老猫自己说龙门资本是自己长期的事业，要把它做成“老钱”。",
            "area": "中国"
        }


}


unk = {
        "123GO":
        {
            "name": "[无消息_v1]",
            "alias": ["123GO"],
            "desc": "“123GO”售货柜中会装配图像识别系统与摄像头，摄像头会对顾客的动作进行捕捉，通过图像识别系统识别出顾客拿走的商品。“这种方式不仅可以大幅度地节省消费者的购物时间，还可以根据不同场景快速变化销售品类，解决了目前自贩机的痛点。”",
            "area": "中国"
        },
    "33财经":
        {
            "name": "[无消息_v2]",
            "alias": ["33财经"],
            "desc": "33财经注册地在中国广东省深圳市。33财经是一家全球数字货币财经媒体，33财经24小时提供独立客观的区块链技术与产业资讯和行情分析，并拥有5万用户的社群雪碧社区。",
            "area": "中国"
        },
    "56食品":
        {
            "name": "[无消息_v3]",
            "alias": ["56食品"],
            "desc": "56食品是一个复合调味品品牌提供商，品牌聚焦于复合调味品的综合运用，依托川渝地区特产渠道和新零售方式，搭建了互联网电商+线下实体门店一体化经营模式。",
            "area": "中国"
        },
    "雪胶de蘑菇之家":
        {
            "name": "[无消息_v4]",
            "alias": ["雪胶de蘑菇之家", "雪胶de蘑菇之"],
            "desc": "雪胶de蘑菇之家成立于2021年6月，通过4平方米的线下店铺，以“菌plus+”的产品思路，为消费者提供以高端食用菌为食材的饮品和小吃，解决消费者既想“好吃解馋”又想“健康减肥”“养生”“美白”等的多重诉求。目前已在上海和重庆开设3家门店，主流客群集中在25-45岁的女性。  ",
            "area": "中国"
        }
}

path = "G:\out\\4be395873c7d668b5e455be8b03c8763.txt"

with open(path, "r", encoding="utf-8") as f:
    data = f.read()

print(data)



