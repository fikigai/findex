import html
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.query import QuerySyntaxError
from findex.ranking import ScorerName, make_scorer
from findex.search import SearchEngine
from findex.web.config import Settings, get_settings
from findex.web.schemas import (
    DocumentOut,
    HealthOut,
    SearchHitOut,
    SearchResponseOut,
    StatsOut,
)

logger = logging.getLogger("findex.web")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_index_safe(index_path: Path) -> Index | None:
    if not index_path.exists():
        logger.warning("Файл індексу не знайдено при старті: %s", index_path)
        return None
    if index_path.suffix == ".json":
        return Index.load_json(index_path)
    return Index.load_pickle(index_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.index = _load_index_safe(settings.index_path)
    if app.state.index is not None:
        logger.info(
            "Індекс завантажено у lifespan: %s (документів: %d)",
            settings.index_path,
            app.state.index.num_docs,
        )
    yield
    app.state.index = None


def get_optional_index(request: Request) -> Index | None:
    idx: Index | None = getattr(request.app.state, "index", None)
    return idx


def get_required_index(
    idx: Annotated[Index | None, Depends(get_optional_index)],
) -> Index:
    if idx is None:
        raise HTTPException(status_code=503, detail="Індекс ще не завантажений")
    return idx


def get_app_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def _format_html_snippet(snippet: str) -> str:
    escaped = html.escape(snippet)
    return re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", escaped)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Findex Web Search",
        version="0.7.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception:
            logger.exception("Unhandled error during request_id=%s", req_id)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )

    @application.get("/health", response_model=HealthOut)
    def health_endpoint(
        idx: Annotated[Index | None, Depends(get_optional_index)],
    ) -> JSONResponse | HealthOut:
        if idx is None:
            payload = HealthOut(status="unavailable", index_loaded=False, num_docs=0)
            return JSONResponse(status_code=503, content=payload.model_dump())
        return HealthOut(status="ok", index_loaded=True, num_docs=idx.num_docs)

    @application.get("/stats", response_model=StatsOut)
    def stats_endpoint(
        idx: Annotated[Index, Depends(get_required_index)],
    ) -> StatsOut:
        return StatsOut(
            num_docs=idx.num_docs,
            vocab_size=len(idx),
            avg_doc_length=round(idx.avg_doc_length, 2),
        )

    @application.get("/docs/{doc_id:path}", response_model=DocumentOut)
    def get_document_endpoint(
        doc_id: str,
        idx: Annotated[Index, Depends(get_required_index)],
        settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> DocumentOut:
        meta = idx.docs.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не знайдено")

        engine = SearchEngine(idx, corpus_dir=settings.corpus_dir)
        content = engine.load_doc_text(doc_id)
        return DocumentOut(doc_id=doc_id, length=meta.length, content=content)

    @application.get("/search", response_model=SearchResponseOut)
    def search_endpoint(
        q: Annotated[str, Query(min_length=1, max_length=256, description="Пошуковий запит")],
        idx: Annotated[Index, Depends(get_required_index)],
        settings: Annotated[Settings, Depends(get_app_settings)],
        k: Annotated[int, Query(ge=1, le=50, description="Розмір сторінки результатів")] = 5,
        scorer: Annotated[ScorerName, Query(description="Модель ранжування")] = "bm25",
        page: Annotated[int, Query(ge=1, le=100, description="Номер сторінки")] = 1,
    ) -> SearchResponseOut:
        t0 = time.perf_counter()
        engine = SearchEngine(
            idx,
            scorer=make_scorer(scorer),
            corpus_dir=settings.corpus_dir,
        )
        try:
            all_hits = engine.search(q, top_k=max(1000, page * k))
        except QuerySyntaxError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

        total_hits = len(all_hits)
        total_pages = max(1, math.ceil(total_hits / k))
        start = (page - 1) * k
        page_slice = all_hits[start : start + k]

        items = [
            SearchHitOut(
                rank=start + i + 1,
                doc_id=hit.doc_id,
                score=round(hit.score, 6),
                snippet=hit.snippet,
            )
            for i, hit in enumerate(page_slice)
        ]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return SearchResponseOut(
            query=q,
            scorer=scorer,
            page=page,
            page_size=k,
            total_hits=total_hits,
            total_pages=total_pages,
            elapsed_ms=elapsed_ms,
            results=items,
        )

    @application.get("/", response_class=HTMLResponse)
    def index_ui(
        request: Request,
        idx: Annotated[Index | None, Depends(get_optional_index)],
        settings: Annotated[Settings, Depends(get_app_settings)],
        q: str = "",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
        doc: str = "",
    ) -> HTMLResponse:
        num_docs = idx.num_docs if idx is not None else 0
        vocab_size = len(idx) if idx is not None else 0
        error_msg: str | None = None
        search_data: dict[str, Any] | None = None
        selected_doc: DocumentOut | None = None

        if idx is None:
            error_msg = "Індекс не завантажений (503). Вкажіть коректний INDEX_PATH."
        else:
            if doc.strip():
                meta = idx.docs.get(doc.strip())
                if meta is not None:
                    engine = SearchEngine(idx, corpus_dir=settings.corpus_dir)
                    selected_doc = DocumentOut(
                        doc_id=doc.strip(),
                        length=meta.length,
                        content=engine.load_doc_text(doc.strip()),
                    )
                else:
                    error_msg = f"Документ '{doc}' не знайдено."

            if q.strip():
                t0 = time.perf_counter()
                k = 5
                engine = SearchEngine(
                    idx,
                    scorer=make_scorer(scorer),
                    corpus_dir=settings.corpus_dir,
                )
                try:
                    all_hits = engine.search(q.strip(), top_k=max(1000, page * k))
                    total_hits = len(all_hits)
                    total_pages = max(1, math.ceil(total_hits / k))
                    start = (page - 1) * k
                    page_slice = all_hits[start : start + k]
                    results_ui = [
                        {
                            "rank": start + i + 1,
                            "doc_id": h.doc_id,
                            "score": h.score,
                            "highlighted_snippet": _format_html_snippet(h.snippet),
                        }
                        for i, h in enumerate(page_slice)
                    ]
                    search_data = {
                        "total_hits": total_hits,
                        "total_pages": total_pages,
                        "page": page,
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "results": results_ui,
                    }
                except QuerySyntaxError as exc:
                    error_msg = f"Помилка синтаксису запиту: {exc}"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": settings.app_title,
                "num_docs": num_docs,
                "vocab_size": vocab_size,
                "q": q,
                "scorer": scorer,
                "error_msg": error_msg,
                "search_data": search_data,
                "selected_doc": selected_doc,
            },
        )

    return application


app = create_app()