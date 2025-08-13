from typing import List, Dict, Any

MOCK_BROKERS = [
    {
        "name": "XM",
        "aliases": ["XM", "XM Global", "XM国际", "xm"],
        "regulation": ["ASIC 443670", "IFSC 000261/106"],
        "score": 7.9,
        "risk": "近30天曝光上升",
        "sources": [
            {"title": "监管信息", "url": "https://www.wikifx.com/zh-cn/search.html?keyword=xm&scene=1"}
        ],
    },
    {
        "name": "Exness",
        "aliases": ["Exness", "艾克森", "exness"],
        "regulation": ["FSA SD025", "CySEC 178/12"],
        "score": 8.3,
        "risk": "总体稳定",
        "sources": [
            {"title": "监管信息", "url": "https://www.wikifx.com/zh-cn/search.html?keyword=exness&scene=1"}
        ],
    },
]

MOCK_CONTENT = [
    {
        "title": "用户反馈 XM 出金到账缓慢，如何处理？",
        "url": "https://www.wikifx.com/zh-cn/exposure.html",
        "snippet": "近期有交易者反馈 XM 出金用时较长，建议先核对账户信息并联系官方客服…",
    },
    {
        "title": "如何辨别 XM 官网真伪（含域名与证书检查）",
        "url": "https://www.wikifx.com/zh-cn/",
        "snippet": "通过域名、SSL 证书与跳转行为识别镜像网站风险…",
    },
    {
        "title": "监管快讯：某券商因反洗钱被罚",
        "url": "https://www.wikifx.com/zh-cn/",
        "snippet": "监管层发布公告提醒风险，建议关注平台合规与公告…",
    },
]


class BrokerAdapter:
    def search(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        broker_name = (entities.get("broker_name") or "").upper()
        if broker_name:
            for b in MOCK_BROKERS:
                if b["name"].upper() == broker_name or any(a.lower() in query for a in b["aliases"]):
                    r = dict(b)
                    r["vertical"] = "broker"
                    return [r]
        # fallback: return top one
        r = dict(MOCK_BROKERS[0])
        r["vertical"] = "broker"
        return [r]


class ContentAdapter:
    def search(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        broker_name = (entities.get("broker_name") or "").lower()
        for c in MOCK_CONTENT:
            hit = False
            if broker_name and broker_name in c["title"].lower():
                hit = True
            if any(k in query for k in ["出金", "曝光", "投诉", "监管", "官网", "真假"]):
                hit = True
            if hit:
                item = dict(c)
                item["vertical"] = "content"
                items.append(item)
        if not items:
            item = dict(MOCK_CONTENT[0])
            item["vertical"] = "content"
            items.append(item)
        return items