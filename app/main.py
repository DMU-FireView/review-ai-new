"""Re:view AI 서비스의 FastAPI 애플리케이션 진입점.

수정 범위:
- [ENTRY POINT]
- router 등록과 애플리케이션 설정을 담당한다.
- analyzer 또는 scoring 로직을 직접 작성하지 않는다.

주의:
- 수집 서비스와 통신하는 httpx client는 애플리케이션 수명에 묶는다. 요청마다 새로 만들면
  매번 TCP·TLS 연결을 다시 맺고, 종료 시 닫히지 않은 연결이 남는다.
- lifespan이 만든 객체는 `app.state.crawler_gateway` / `app.state.crawler_stream_client`에
  둔다. `app.api.analysis`의 의존성이 이 이름으로 꺼내 쓰므로 한쪽만 바꾸지 않는다.
- 설정은 기동 시 한 번 읽는다. 프로세스가 도는 중에 환경변수가 바뀌어도 다시 읽지 않는다.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.analysis import router as analysis_router
from app.config import load_crawler_settings
from app.integrations.crawler_client import AsyncHttpCrawlerGateway
from app.integrations.crawler_stream import CrawlerStreamClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """수집 서비스 client를 만들고, 애플리케이션이 내려갈 때 닫는다."""

    settings = load_crawler_settings()
    app.state.crawler_gateway = AsyncHttpCrawlerGateway(settings=settings)
    app.state.crawler_stream_client = CrawlerStreamClient(settings=settings)
    try:
        yield
    finally:
        await app.state.crawler_gateway.aclose()
        await app.state.crawler_stream_client.aclose()


app = FastAPI(title="Re:view AI", lifespan=lifespan)
app.include_router(analysis_router)
