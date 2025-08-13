import re
from typing import Dict, Any

BRAND_ALIASES = {
    "xm": ["xm", "xm global", "xm国际", "xm平台"],
    "exness": ["exness", "艾克森"],
    "ic markets": ["ic markets", "icm", "艾熙市场"],
}

TASK_KEYWORDS = {
    "official_site_verification": ["官网", "官网地址", "真假", "真伪", "下载", "app"],
    "license_lookup": ["监管", "牌照", "license", "执照"],
    "complaint_support": ["维权", "投诉", "出金", "不到账", "冻结", "跑路", "黑", "曝光"],
    "server_info": ["mt4", "mt5", "服务器", "延迟", "登录"],
    "general_lookup": ["是什么", "怎么样", "评分", "安全吗", "评价"],
}

VERTICAL_PRIOR = {
    "official_site_verification": ["broker", "official"],
    "license_lookup": ["broker"],
    "complaint_support": ["content", "broker"],
    "server_info": ["broker", "content"],
    "general_lookup": ["broker", "content"],
}


class IntentDetector:
    def normalize(self, query: str) -> str:
        return query.strip().lower()

    def extract_entities(self, query: str) -> Dict[str, Any]:
        broker_name = None
        for brand, aliases in BRAND_ALIASES.items():
            for a in aliases:
                if a in query:
                    broker_name = brand.upper()
                    break
            if broker_name:
                break

        domain_match = re.search(r"([a-z0-9-]+\.)+[a-z]{2,}", query)
        domain = domain_match.group(0) if domain_match else None

        return {"broker_name": broker_name, "domain": domain}

    def classify_intent(self, query: str, entities: Dict[str, Any]):
        task = "general_lookup"
        for t, kws in TASK_KEYWORDS.items():
            if any(kw in query for kw in kws):
                task = t
                break
        verticals = VERTICAL_PRIOR.get(task, ["broker"])  # default broker first

        return {
            "vertical": verticals,
            "task": task,
            "entities": {k: v for k, v in entities.items() if v},
        }