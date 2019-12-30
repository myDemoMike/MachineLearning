# coding: utf-8
USER_SESSION = "user"
KEY_SESSION = "key"

REFS = {
    "项目名称": ["ichinascope", "chinascopefinancial", "基金", "智投", "数库云", "数库港", "后台系统"],
    "数据库": ["mongodb", "mysql", "oracle", "sqlserver"],
    "web": ["PHP", "Nginx", "Apache", "other"],
    "应用服务": [
        "portalmobile",
        "ada",
        "datafeed",
        "datashift",
        "datareceiver",
        "datacloud",
        "hqservice",
        "csdata",
        "otcds",
        "bts",
        "shukugang",
        "idexservice",
        "indexer",
        "fund",
        "etl",
        "exportservice",
        "jobscheduler",
        "search",
        "bda",
        "ccas",
        "userservice",
        "dal",
        "etlinger",
        "indexservice",
        "egw",
        "notification",
        "datasupply"
    ],
    "服务": ["数据库", "web", "应用服务"],
    "地点": ["办公室", "上海IDC", "亚马逊(北京)", "亚马逊(新加坡)"],
    "审批": ["内部测试", "运维发布", "正式环境测试"],
    "未通过原因": ["BUG", "文件未成功修改", "数据问题", "配置文件错误", "问题未修改", "发布问题", "其他"],
    "库存状态": ["领用", "借用", "维修", "报废"],
    "入库类型": ["采购入库", "其它入库"]
}


class DEFAULT_GROUP:
    QA = "QA"
    OP = "OP"
    GUEST = "GUEST"
    HR = "HR"
    CONTACT = "CONTACT"
    CEO = "BOSS"

    class LEAVE:
        MAIL = "LEAVE_MAIL"
        APPROVAL = "LEAVE_APPROVAL"
        SEARCH = "LEAVE_SEARCH"

    class PURCHASE:
        FIN = "FIN"  # "财务"
        UP = "UP"  # 上级
        BOSS = "BOSS"  # 老板
        LAW = "LAW"  # 法务

    class SMS:
        ALERT = "SMS_ALERT"


class RELEASE:
    PASS = u"已确认"
    FAIL = u"退回"
    FLOW_TEST = u"内部测试"
    FLOW_OP = [u"运维发布"]
    FLOW_QA = [FLOW_TEST, u"正式环境测试"]
    OPTIONS = "options"


class RELEASE_SERVICE:
    DATA_BASE = u"数据库"


class LEAVE:
    PASS = u"已确认"
    FAIL = u"退回"
    APPLYING = u""
    TYPE = u"请假类型"
    HOLIDAY = u"假期"
    WORKDAY = u"工作日"
    ANNUAL = u"年假"
    _BOSS = u"boss"
    _2ND = u"2nd"
    _1ST = u"1st"
    DRAFT = u"草稿"
    HR = u"HR"
    CEO = u"BOSS"
    GOUT = u"LEAVE_GOUT"
    GOUT_DESC = u"人事主管"
    TRAIN = u"外出培训"
    MEET = u"外出会议"
    SEARCH = u"LEAVE_SEARCH"
    GL = "GL"

class EXPENSE:
    PASS = u"已确认"
    FAIL = u"退回"
    DRAFT = u"草稿"
    ADMIN = u"ADMIN"
    _2ND = u"2nd"
    FIN = u"FIN"
    CEO = u"BOSS"
    HR = u"HR"
    BUILD = u"BUILD"
    OT = u"加班报销"
    TEAM_BUILD = u"团建报销"
    CATE = u"报销项目"
    ITEM = u"报销类型"
    LIMIT = 10000
    DESC_2ND = "部门主管"
    GL = "GL"
    PRESIDENT = u"PRESIDENT"


class PURCHASE:
    PASS = u"已确认"
    FAIL = u"退回"
    FIN = "FIN"
    FLOW_ONE = "UP"
    FLOW_TWO = "FIN"
    FLOW_THREE = "BOSS"
    FLOW_FOUR = "LAW"
    FLOW_FIVE = "PAY"
    FLOW = [FLOW_ONE, FLOW_TWO, FLOW_THREE, FLOW_FOUR, FLOW_FIVE]
    PRICE_LIMIT = 10000

    class APPROVAL_TYPE:
        CYCLE = "CYCLE"


class SCRAP:
    PASS = u"已确认"
    FAIL = u"退回"
    TYPE = u"报废"


class STOCK:
    TYPE = u"库存"


class ACCESS:
    DEPLOY = u"已部署"
    PASS = u"已确认"
    FAIL = u"退回"
    DRAFT = u"草稿"
    OP = u"OP"
    SSH_TYP = u"服务器登录"
    _1ST = u"1st"


class CONTRACT2:
    LEGAL = u"Legal"
    FIN = u"Finance"
    SALE = u"BD"
    BOSS = u"CEO OFFICE"
    PASS = u"已确认"
    FAIL = u"退回"
    DRAFT = u"草稿"
    BILL = u"BILL"
    PAY = u"PAY"
    OVER_TIME = u"OVER_TIME"
    MAINTAIN = u"MAINTAIN"


class PROJECT:
    EDIT_PROJ = "edit_proj"
    EDIT_DOC = "edit_doc"
    STATUS_NEW = "新建"
    STATUS_ON_TIME = "正常推进"
    STATUS_FINISH = "完成"
    STATUS_AHEAD = "提前完成"
    STATUS_DELAY = "延误"

    @staticmethod
    def EDIT_MEMBER(type):
        return "%s_member" % type

    @staticmethod
    def EDIT_SCHEDULE(type):
        return "edit_sched_%s" % type


class DOC:
    PUT = "PUT"
    DELETE = "DELETE"
    NEW = "NEW"


class FLOW:
    PASS = u"已确认"
    FAIL = u"退回"

    _1ST = "1st"
    _2ND = "2nd"
    _BOSS = "boss"


if __name__ == "__main__":
    pass
