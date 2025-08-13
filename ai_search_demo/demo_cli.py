#!/usr/bin/env python3
import json
import sys
from typing import List, Dict, Any

# Local imports
from app.services.intent import IntentDetector
from app.services.router import NavigationRouter
from app.services.adapters import BrokerAdapter, ContentAdapter


def navigate(query: str) -> Dict[str, Any]:
    intent_service = IntentDetector()
    router_service = NavigationRouter()
    broker_adapter = BrokerAdapter()
    content_adapter = ContentAdapter()

    normalized = intent_service.normalize(query)
    entities = intent_service.extract_entities(normalized)
    intent = intent_service.classify_intent(normalized, entities)
    route = router_service.compute_route(intent)

    results: List[Dict[str, Any]] = []
    for r in route:
        if r["vertical"] == "broker":
            results.extend(broker_adapter.search(normalized, entities))
        elif r["vertical"] == "content":
            results.extend(content_adapter.search(normalized, entities))

    cards = router_service.compose_cards(results, intent, entities)
    return {
        "intent": intent,
        "route": route,
        "cards": cards,
    }


def main():
    query = "xm 出金困难"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    payload = navigate(query)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()