from flask import Flask, request, send_from_directory, jsonify
from pathlib import Path
from typing import List, Dict, Any

from app.services.intent import IntentDetector
from app.services.router import NavigationRouter
from app.services.adapters import BrokerAdapter, ContentAdapter

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

intent_service = IntentDetector()
router_service = NavigationRouter()
broker_adapter = BrokerAdapter()
content_adapter = ContentAdapter()


@app.get("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.post("/ai-search/navigate")
def navigate():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
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

    payload = {"intent": intent, "route": route, "cards": cards}
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)