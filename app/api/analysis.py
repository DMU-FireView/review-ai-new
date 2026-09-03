r"""리뷰 분석 요청을 받는 HTTP router 모듈.

역할:
- 이미 수집된 crawler payload를 받아 분석한다.
- 수집 서비스를 호출해 리뷰를 가져온 뒤 곧바로 분석한다.
- 수집이 오래 걸리는 요청을 SSE로 흘려보내며 진행 상황과 최종 분석 결과를 내려준다.
- 결과에 crawler 원본 식별자를 다시 붙여 응답한다.

수정 범위:
- [INTEGRATION]
- 엔드포인트 경로·요청·응답 형태 변경은 Spring 담당자와 협의한다.
- AI scoring 로직은 이 영역에 구현하지 않는다. 분석은 항상 `_analyze` 한 경로로만 간다.

주의:
- 분석 service는 입력 순서대로 결과를 돌려주므로 요청 리뷰와 1:1로 대응시킨다.
- 외부 호출이 없는 similarity adapter만 주입한다. 감성 분석 adapter는 자격 증명과
  호출 비용 정책이 정해진 뒤 주입한다.
- `/collect`는 수집이 끝날 때까지 응답을 붙잡는다. 수집 시간은 플랫폼과 limit에 좌우되므로
  호출 측 timeout을 `CRAWLER_TIMEOUT`보다 넉넉히 잡아야 한다. 진행 상황이 필요하면
  `/collect/stream`을 쓴다.
- 수집 gateway와 스트림 client는 이 모듈이 만들지 않는다. `app.main`의 lifespan이 만들어
  `app.state.crawler_gateway` / `app.state.crawler_stream_client`에 넣어 둔 것을 쓴다.
  httpx client 수명을 애플리케이션 수명에 묶기 위해서다. 두 속성 이름은 `app.main`과
  이 파일 양쪽에 나오므로 한쪽만 바꾸지 않는다.
- 분석은 CPU 작업이라 event loop를 막는다. async 경로에서는 반드시 threadpool로 넘긴다.

크롤러 서버가 구현해야 하는 SSE 계약:
- **이 경로는 아직 크롤러 서버(ai-review-crawler)에 없다. 크롤러 담당자와 합의 후 구현이 필요하다.**
  현재 크롤러 서버가 제공하는 경로는 `GET /platforms`, `GET /{platform}/search`,
  `GET /{platform}/products/{product_id}`, `GET /{platform}/products/{product_id}/reviews`
  네 개뿐이다. 아래는 이 저장소가 **제안하는** 계약이며, 합의 전까지 `/collect/stream`은
  동작하지 않는다(크롤러가 404를 돌려주므로 502로 나간다).
- 경로: `GET /{platform}/products/{product_id}/reviews/stream`
  (단발 조회 경로 `.../reviews`에 `/stream`을 붙인 형태. 실제 경로는
  `app.integrations.crawler_stream.stream_reviews_path`가 만든다.)
- 요청 파라미터: `limit`(int, 수집할 최대 리뷰 수). 재연결 시 `Last-Event-ID` 헤더로
  마지막으로 내려보낸 `id`를 함께 보낸다. 서버가 이 헤더를 무시해도 되지만, 그 경우
  이미 보낸 리뷰가 다시 온다.
- 응답: `Content-Type: text/event-stream`. 이벤트는 다음 5개뿐이다.

  | event       | data                                                        |
  |-------------|-------------------------------------------------------------|
  | `review`    | `CrawlerReview` JSON object 1건                              |
  | `progress`  | `{"job_id": str, "collected": int, "target": int \| null}`   |
  | `done`      | `{"job_id": str, "collected": int}`                          |
  | `error`     | `{"job_id": str, "detail": str, "retryable": bool}`          |
  | `heartbeat` | `{}`                                                         |

- `done`이 정상 종료 신호다. `done` 없이 끊긴 스트림은 성공으로 보지 않는다.
- 수집 실패는 반드시 `error` 이벤트로 알린다. 빈 스트림을 조용히 닫아 "리뷰 0건"처럼
  보이게 하지 않는다. `target`을 모르면 null로 보낸다. 0으로 채우지 않는다.
- 이벤트 모델은 `app.schemas.stream`, 수신 구현은 `app.integrations.crawler_stream`에 있다.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.integrations.crawler import to_analysis_inputs
from app.integrations.crawler_client import (
    AsyncCrawlerGateway,
    CrawlerRequestError,
    CrawlerUnavailableError,
)
from app.integrations.crawler_stream import CrawlerStreamClient
from app.integrations.similarity import NormalizedTextSimilarityAdapter
from app.schemas.analysis import ProductAnalysisResponse, to_review_response
from app.schemas.crawler import (
    CollectAnalysisRequest,
    CollectStreamRequest,
    CrawlerAnalysisRequest,
    CrawlerReview,
    PlatformsResponse,
)
from app.schemas.stream import (
    DoneEvent,
    ErrorEvent,
    HeartbeatEvent,
    ProgressEvent,
    ReviewEvent,
)
from app.services.analysis import analyze_product_reviews

router = APIRouter(prefix="/analysis", tags=["analysis"])

SSE_HEADERS = {
    "Cache-Control": "no-store",
    # nginx가 스트림을 버퍼링해 진행 상황을 뭉쳐 보내지 않도록 한다.
    "X-Accel-Buffering": "no",
}
"""이 서버가 SSE 응답에 붙이는 헤더."""


def get_crawler_gateway(request: Request) -> AsyncCrawlerGateway:
    """lifespan이 만든 수집 gateway를 준다. 테스트는 이 의존성을 교체한다."""

    return request.app.state.crawler_gateway


def get_crawler_stream_client(request: Request) -> CrawlerStreamClient:
    """lifespan이 만든 SSE 수신 client를 준다. 테스트는 이 의존성을 교체한다."""

    return request.app.state.crawler_stream_client


@router.get(
    "/platforms",
    response_model=PlatformsResponse,
    summary="수집 가능한 플랫폼 목록",
)
async def list_platforms(
    gateway: AsyncCrawlerGateway = Depends(get_crawler_gateway),
) -> PlatformsResponse:
    """수집 서비스가 현재 제공하는 플랫폼과 로드 실패 플랫폼을 반환한다."""

    available, failed = await _call_crawler(gateway.list_platforms)
    return PlatformsResponse(available=list(available), failed=list(failed))


@router.post(
    "/collect",
    response_model=ProductAnalysisResponse,
    summary="상품 리뷰를 수집한 뒤 RTI 분석",
)
async def collect_and_analyze(
    request: CollectAnalysisRequest,
    gateway: AsyncCrawlerGateway = Depends(get_crawler_gateway),
) -> ProductAnalysisResponse:
    """수집 서비스에서 리뷰를 가져와 리뷰별 RTI와 판단 근거를 반환한다."""

    reviews = await _call_crawler(
        gateway.fetch_reviews,
        request.platform,
        request.product_id,
        limit=request.limit,
    )
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=_empty_collection_detail(request.platform, request.product_id),
        )

    # 분석은 CPU 작업이다. async 경로에서 직접 부르면 다른 요청까지 멈춘다.
    return await run_in_threadpool(_analyze, reviews, product_key=request.product_key)


@router.post(
    "/collect/stream",
    summary="상품 리뷰를 수집하며 진행 상황과 RTI 분석 결과를 SSE로 전송",
    response_class=StreamingResponse,
)
async def collect_stream(
    request: CollectStreamRequest,
    stream_client: CrawlerStreamClient = Depends(get_crawler_stream_client),
) -> StreamingResponse:
    """크롤러 SSE를 구독하면서 진행 상황을 흘리고, 수집이 끝나면 분석 결과를 보낸다.

    이 서버가 내보내는 이벤트 (크롤러 이벤트를 그대로 되쏘지 않는다):

    | event       | data                                                       | 언제                         |
    |-------------|------------------------------------------------------------|------------------------------|
    | `progress`  | `{"job_id": str \\| null, "collected": int, "target": int \\| null}` | 리뷰를 받을 때마다, 크롤러 progress를 받을 때마다 |
    | `result`    | `ProductAnalysisResponse` JSON 그대로                       | 수집 완료 후 분석이 끝났을 때 한 번 |
    | `error`     | `{"status": int, "detail": str}`                            | 실패했을 때 한 번            |
    | `heartbeat` | `{}`                                                        | 크롤러 heartbeat를 받을 때마다 |

    - `result`와 `error`가 종단 이벤트다. 둘 중 하나가 나가면 스트림을 닫는다.
    - `progress.collected`는 **이 서버가 실제로 받은 리뷰 수**다. 크롤러가 주장한 수가 아니라
      분석에 넣을 수 있는 관측값이다. `target`은 크롤러가 알려준 값이고 모르면 null로 둔다.
    - `job_id`는 크롤러 이벤트에 실려 온 값을 쓰고, 아직 못 받았으면 요청에 담긴 값을,
      그것도 없으면 null을 보낸다. 이 서버가 임의로 만들지 않는다.
    - 크롤러가 보낸 미지의 이벤트(`UnknownEvent`)는 하류로 전파하지 않는다. 계약에 없는
      이름을 Spring에 흘리면 이 서버의 이벤트 목록이 크롤러 사정에 따라 늘어난다.

    분석 시점: **수집이 끝난 뒤 전체를 한 번에 분석한다. 배치마다 재분석하지 않는다.**
    근거:
    - network analyzer는 같은 상품 리뷰 묶음 안의 유사도를 본다. 부분 집합으로 계산한 RTI는
      최종 RTI와 다른 값이다. 배치마다 결과를 내보내면 같은 리뷰가 중간엔 safe, 끝엔 danger로
      바뀌어 나가고, 호출 측은 어느 값을 저장해야 하는지 알 수 없게 된다.
    - 배치 재분석은 이 문제를 줄이지 못한다. 배치도 여전히 부분 집합이고, 유사도 계산은
      리뷰 수에 대해 O(n²)라 배치 수만큼 같은 계산을 반복한다.
    - 스트리밍으로 얻으려던 것(수집이 오래 걸릴 때 진행 상황을 알리고 커넥션을 살려 두는 것)은
      `progress`만으로 이미 얻는다. 분석 결과를 쪼개 보낼 이유가 없다.

    실패를 알리는 방법 (SSE는 헤더를 먼저 보내므로 경계가 있다):
    - **첫 이벤트를 받기 전** 실패하면 아직 아무 바이트도 보내지 않았으므로 평소처럼 HTTP
      상태 코드로 알린다(연결 실패 504, 크롤러가 거절 404/502). 호출 측이 스트림 본문을
      파싱하지 않고도 실패를 알 수 있어 이 편이 낫다.
    - **첫 이벤트를 받은 뒤** 실패하면 상태 코드는 이미 200으로 나갔다. 이때는 `error`
      이벤트로 알리고 스트림을 닫는다. 리뷰 0건도 여기에 해당해 `status: 404`인 `error`
      이벤트로 나간다. 실패를 빈 `result`로 감추지 않는다.
    """

    events = stream_client.stream_reviews(
        request.platform,
        request.product_id,
        limit=request.limit,
    )
    # 첫 이벤트를 미리 당겨, 연결 자체가 실패한 경우를 HTTP 상태 코드로 알린다.
    first_event = await _first_stream_event(events)

    return StreamingResponse(
        _emit_analysis_stream(first_event, events, request),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post(
    "/crawler-reviews",
    response_model=ProductAnalysisResponse,
    summary="이미 수집된 crawler 리뷰 묶음의 RTI 분석",
)
def analyze_crawler_reviews(request: CrawlerAnalysisRequest) -> ProductAnalysisResponse:
    """동일 상품 crawler 리뷰를 분석해 리뷰별 RTI와 판단 근거를 반환한다."""

    return _analyze(tuple(request.reviews), product_key=request.product_key)


async def _emit_analysis_stream(
    first_event: object,
    events: AsyncIterator[object],
    request: CollectStreamRequest,
) -> AsyncIterator[bytes]:
    """크롤러 이벤트를 이 서버의 이벤트로 옮기고, 마지막에 분석 결과를 보낸다."""

    reviews: list[CrawlerReview] = []
    job_id = request.job_id
    target: int | None = None

    try:
        async for event in _chain(first_event, events):
            match event:
                case ReviewEvent():
                    reviews.append(event)
                    yield _sse("progress", _progress_data(job_id, len(reviews), target))
                case ProgressEvent():
                    job_id = event.job_id
                    target = event.target
                    yield _sse("progress", _progress_data(job_id, len(reviews), target))
                case HeartbeatEvent():
                    yield _sse("heartbeat", "{}")
                case DoneEvent():
                    job_id = event.job_id
                    yield await _final_event(tuple(reviews), request)
                    return
                case ErrorEvent():  # pragma: no cover - client가 예외로 올린다
                    raise CrawlerRequestError(event.detail, status_code=502)
                case _:
                    # 계약에 없는 이벤트. 하류로 전파하지 않는다.
                    continue
    except (CrawlerUnavailableError, CrawlerRequestError) as exc:
        yield _sse("error", _error_data(_crawler_http_error(exc)))
        return

    # `done` 없이 끝났다면 수집이 완료됐다고 볼 수 없다. 받은 리뷰로 결과를 만들지 않는다.
    yield _sse(
        "error",
        _error_data(
            HTTPException(
                status_code=502,
                detail="수집 서비스가 done 이벤트 없이 스트림을 닫았습니다.",
            )
        ),
    )


async def _final_event(
    reviews: tuple[CrawlerReview, ...],
    request: CollectStreamRequest,
) -> bytes:
    """수집 완료 시점에 내보낼 종단 이벤트를 만든다."""

    if not reviews:
        # 헤더는 이미 나갔으므로 404를 상태 코드로 알릴 수 없다. 이벤트로 알린다.
        return _sse(
            "error",
            _error_data(
                HTTPException(
                    status_code=404,
                    detail=_empty_collection_detail(request.platform, request.product_id),
                )
            ),
        )

    try:
        # 분석은 CPU 작업이다. 스트림을 만드는 event loop를 막지 않도록 threadpool로 넘긴다.
        response = await run_in_threadpool(
            _analyze, reviews, product_key=request.product_key
        )
    except HTTPException as exc:
        return _sse("error", _error_data(exc))

    return _sse("result", response.model_dump_json())


async def _first_stream_event(events: AsyncIterator[object]) -> object:
    """스트림의 첫 이벤트를 당긴다. 여기서의 실패는 HTTP 상태 코드로 올린다."""

    try:
        return await anext(events)
    except (CrawlerUnavailableError, CrawlerRequestError) as exc:
        raise _crawler_http_error(exc) from exc
    except StopAsyncIteration as exc:
        raise HTTPException(
            status_code=502,
            detail="수집 서비스가 이벤트를 하나도 보내지 않고 스트림을 닫았습니다.",
        ) from exc


async def _chain(
    first_event: object,
    events: AsyncIterator[object],
) -> AsyncIterator[object]:
    """미리 당겨 둔 첫 이벤트를 스트림 앞에 다시 붙인다."""

    yield first_event
    async for event in events:
        yield event


def _analyze(
    reviews: tuple[CrawlerReview, ...],
    *,
    product_key: str | None,
) -> ProductAnalysisResponse:
    """crawler 리뷰를 분석하고 원본 식별자를 붙여 응답을 만든다."""

    try:
        analysis_inputs = to_analysis_inputs(reviews, product_key=product_key)
        results = analyze_product_reviews(
            analysis_inputs[0].product_id,
            analysis_inputs,
            similarity_adapter=NormalizedTextSimilarityAdapter(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    responses = [
        to_review_response(
            result,
            platform=review.platform,
            review_id=review.review_id,
            product_id=review.product_id,
        )
        for review, result in zip(reviews, results, strict=True)
    ]
    return ProductAnalysisResponse(
        product_key=analysis_inputs[0].product_id,
        review_count=len(responses),
        results=responses,
    )


async def _call_crawler(operation, *args: object, **kwargs: object):
    """수집 서비스 예외를 이 API의 상태 코드로 옮긴다."""

    try:
        return await operation(*args, **kwargs)
    except (CrawlerUnavailableError, CrawlerRequestError) as exc:
        raise _crawler_http_error(exc) from exc


def _crawler_http_error(exc: Exception) -> HTTPException:
    """수집 서비스 예외를 이 API의 HTTP 오류로 옮긴다."""

    if isinstance(exc, CrawlerUnavailableError):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, CrawlerRequestError):
        # 수집 서비스가 알려준 사유를 그대로 전달하되, 5xx는 게이트웨이 오류로 표시한다.
        status_code = exc.status_code if exc.status_code < 500 else 502
        return HTTPException(status_code=status_code, detail=str(exc))
    raise exc


def _empty_collection_detail(platform: str, product_id: str) -> str:
    """수집 결과가 비었을 때의 사유. 단발 경로와 스트림 경로가 같은 문장을 쓴다."""

    return f"'{platform}' 상품 {product_id}에서 수집된 리뷰가 없습니다."


def _progress_data(job_id: str | None, collected: int, target: int | None) -> str:
    """진행 상황 payload를 만든다. 모르는 값은 null로 두고 0으로 채우지 않는다."""

    return json.dumps(
        {"job_id": job_id, "collected": collected, "target": target},
        ensure_ascii=False,
    )


def _error_data(exc: HTTPException) -> str:
    """오류 payload를 만든다. HTTP 상태 코드를 이벤트 안에 그대로 싣는다."""

    return json.dumps(
        {"status": exc.status_code, "detail": str(exc.detail)},
        ensure_ascii=False,
    )


def _sse(name: str, data: str) -> bytes:
    """SSE 프레임 하나를 만든다. `data`는 줄바꿈이 없는 직렬화된 JSON이어야 한다."""

    return f"event: {name}\ndata: {data}\n\n".encode()
