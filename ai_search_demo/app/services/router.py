from typing import List, Dict, Any

class NavigationRouter:
    def compute_route(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        verticals = intent.get("vertical", ["broker"])  # ordered by priority
        # simple weights by position
        total = len(verticals)
        route = []
        for i, v in enumerate(verticals):
            weight = round((total - i) / total, 2)
            route.append({"vertical": v, "weight": weight})
        return route

    def compose_cards(self, results: List[Dict[str, Any]], intent: Dict[str, Any], entities: Dict[str, Any]):
        task = intent.get("task")
        cards: List[Dict[str, Any]] = []

        # group results by vertical
        broker_items = [r for r in results if r.get("vertical") == "broker"]
        content_items = [r for r in results if r.get("vertical") == "content"]

        if broker_items:
            b = broker_items[0]
            cards.append({
                "type": "broker_summary",
                "title": f"{b.get('name', '平台')} 概览",
                "facts": {
                    "regulation": b.get("regulation", []),
                    "score": b.get("score"),
                    "risk": b.get("risk"),
                },
                "sources": b.get("sources", []),
            })

        if task == "complaint_support" or content_items:
            cards.append({
                "type": "action_suggestion",
                "title": "行动建议",
                "steps": [
                    "核对账户实名信息",
                    "优先联系官方客服并保留沟通记录",
                    "在 WikiFX 提交维权工单并上传证据",
                ],
                "links": [
                    {"text": "WikiFX 曝光/维权入口", "url": "https://www.wikifx.com/zh-cn/exposure.html"}
                ]
            })

        if content_items:
            top = content_items[:3]
            cards.append({
                "type": "risk_exposure",
                "title": "相关曝光与讨论",
                "items": [
                    {"title": i.get("title"), "url": i.get("url"), "snippet": i.get("snippet")}
                    for i in top
                ]
            })

        # fallback when nothing
        if not cards:
            cards.append({
                "type": "empty",
                "title": "未找到直接结果，已为你提供通用建议",
                "steps": ["尝试更具体的关键词，如 '平台名 + 出金' 或 '平台名 + 监管牌照'"],
            })

        return cards