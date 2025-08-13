from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.services.intent import IntentDetector
from app.services.router import NavigationRouter
from app.services.adapters import BrokerAdapter, ContentAdapter

app = FastAPI(title="WikiFX AI Search Demo", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


class NavigateRequest(BaseModel):
    query: str
    locale: str = "zh-CN"
    scene: int = 1
    user_id: Optional[str] = None


intent_service = IntentDetector()
router_service = NavigationRouter()

broker_adapter = BrokerAdapter()
content_adapter = ContentAdapter()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/ai-search/navigate")
async def navigate(req: NavigateRequest):
    normalized = intent_service.normalize(req.query)
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

    payload = {
        "intent": intent,
        "route": route,
        "cards": cards,
    }
    return JSONResponse(payload)