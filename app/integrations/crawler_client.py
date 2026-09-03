"""수집 서비스(ai-review-crawler)를 HTTP로 호출하는 gateway 모듈.

역할:
- 수집 서비스에서 플랫폼 목록과 상품별 리뷰를 가져온다.
- 수집 서비스의 오류를 이 서비스가 다룰 수 있는 예외로 옮긴다.
- 동기 gateway와 async gateway를 같은 예외·같은 응답 해석 규칙으로 함께 제공한다.

수정 범위:
- [MODEL / EXTERNAL INTEGRATION]
- 수집 서비스 주소·경로·오류 규약이 바뀌면 이 파일을 수정한다.
- base URL과 timeout 기본값·환경변수 이름은 `app.config`에서 관리한다. 여기서 다시 정의하지 않는다.
- analyzer 점수 의미나 scoring 정책은 이 파일에서 변경하지 않는다.

주의:
- 수집 서비스를 이 저장소의 패키지로 가져오지 않는다. 결합점은 base URL 하나로 유지한다.
  나중에 두 서비스를 합치기로 하면 이 파일과 설정만 교체하면 된다.
- 수집 실패를 빈 리뷰 목록으로 감추지 않는다. 실패는 실패로 올린다.
- FastAPI 라우터에서는 `AsyncHttpCrawlerGateway`를 쓴다. 동기 gateway를 `async def`
  안에서 호출하면 수집이 끝날 때까지 이벤트 루프가 통째로 멈춘다.

HTTP client 수명 (호출마다 새로 만들던 구조를 바꾼 근거):
- 이전에는 요청마다 `httpx.Client`를 만들고 버렸다. 매번 TCP + TLS 연결을 새로 맺는다는 뜻이고,
  `/collect` 한 번이 `platforms` 조회와 리뷰 조회로 나뉘는 지금 구조에서는 그 비용이 그대로 쌓인다.
- 그래서 gateway 하나가 client 하나를 갖고 연결을 재사용한다. 대신 수명을 명시적으로 다룬다:
  `close()` / `aclose()`를 부르거나 context manager로 쓰면 닫힌다.
- 밖에서 client를 주입하면(`client=`) 그 client는 gateway가 닫지 않는다. 만든 쪽이 닫는다.
  애플리케이션 lifespan이 client를 소유하는 배선을 막지 않기 위해서다.
"""

from types import TracebackType
from typing import Protocol

import httpx

from app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    CrawlerSettings,
    load_crawler_settings,
)
from app.schemas.crawler import CrawlerReview

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "AsyncCrawlerGateway",
    "AsyncHttpCrawlerGateway",
    "CrawlerGateway",
    "CrawlerRequestError",
    "CrawlerUnavailableError",
    "HttpCrawlerGateway",
]

PLATFORMS_PATH = "/platforms"
"""수집 서비스의 플랫폼 목록 경로."""


def reviews_path(platform: str, product_id: str) -> str:
    """상품 리뷰 조회 경로를 만든다. 동기·async gateway가 같은 경로를 쓰게 한다."""

    return f"/{platform}/products/{product_id}/reviews"


class CrawlerUnavailableError(RuntimeError):
    """수집 서비스에 닿지 못했거나 응답을 기다리지 못한 경우."""


class CrawlerRequestError(RuntimeError):
    """수집 서비스가 요청을 거절한 경우. 원본 상태 코드를 함께 전달한다."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class CrawlerGateway(Protocol):
    """수집 서비스 호출 방식과 무관한 동기 gateway 계약."""

    def list_platforms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(사용 가능한 platform, 로드에 실패한 platform)을 반환한다."""

    def fetch_reviews(
        self,
        platform: str,
        product_id: str,
        *,
        limit: int,
    ) -> tuple[CrawlerReview, ...]:
        """수집 서비스에서 상품 리뷰를 가져온다."""


class AsyncCrawlerGateway(Protocol):
    """`CrawlerGateway`와 같은 계약의 async 버전.

    메서드 이름과 반환 타입은 동기 계약과 같다. 라우터가 어느 쪽을 주입받든
    같은 코드 모양으로 쓰도록, 그리고 테스트 fake를 한 벌만 쓰도록 맞춘 것이다.
    """

    async def list_platforms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(사용 가능한 platform, 로드에 실패한 platform)을 반환한다."""

    async def fetch_reviews(
        self,
        platform: str,
        product_id: str,
        *,
        limit: int,
    ) -> tuple[CrawlerReview, ...]:
        """수집 서비스에서 상품 리뷰를 가져온다."""


class HttpCrawlerGateway:
    """`crawler serve`가 노출하는 HTTP API를 호출하는 동기 gateway."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        settings: CrawlerSettings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_crawler_settings()
        resolved = base_url if base_url is not None else self.settings.base_url
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout if timeout is not None else self.settings.timeout
        self._client = client
        # 주입받은 client는 만든 쪽이 닫는다. 직접 만든 client만 이 gateway가 닫는다.
        self._owns_client = client is None

    def list_platforms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """수집 서비스의 `GET /platforms` 결과를 그대로 옮긴다."""

        return _as_platforms(self._get(PLATFORMS_PATH))

    def fetch_reviews(
        self,
        platform: str,
        product_id: str,
        *,
        limit: int,
    ) -> tuple[CrawlerReview, ...]:
        """수집 서비스의 `GET /{platform}/products/{product_id}/reviews`를 호출한다."""

        payload = self._get(
            reviews_path(platform, product_id),
            params={"limit": limit},
        )
        return _as_reviews(payload)

    def close(self) -> None:
        """이 gateway가 만든 client를 닫는다. 주입받은 client는 건드리지 않는다."""

        if self._client is not None and self._owns_client:
            self._client.close()
        if self._owns_client:
            self._client = None

    def __enter__(self) -> "HttpCrawlerGateway":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_client(self) -> httpx.Client:
        """client를 처음 쓸 때 만들고, 이후 호출은 같은 연결 풀을 재사용한다."""

        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _get(self, path: str, *, params: dict | None = None) -> object:
        try:
            response = self._ensure_client().get(path, params=params)
        except httpx.TimeoutException as exc:
            raise _timeout_error(self.base_url, self.timeout) from exc
        except httpx.HTTPError as exc:
            raise _connection_error(self.base_url, exc) from exc

        _raise_for_error(response)
        return response.json()


class AsyncHttpCrawlerGateway:
    """`HttpCrawlerGateway`와 같은 API를 async로 제공하는 gateway.

    FastAPI 라우터가 수집 서비스를 기다리는 동안 이벤트 루프를 놓아주게 하는 것이 목적이다.
    예외 종류·메시지와 응답 해석 규칙은 동기 gateway와 같다.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        settings: CrawlerSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_crawler_settings()
        resolved = base_url if base_url is not None else self.settings.base_url
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout if timeout is not None else self.settings.timeout
        self._client = client
        self._owns_client = client is None

    async def list_platforms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """수집 서비스의 `GET /platforms` 결과를 그대로 옮긴다."""

        return _as_platforms(await self._get(PLATFORMS_PATH))

    async def fetch_reviews(
        self,
        platform: str,
        product_id: str,
        *,
        limit: int,
    ) -> tuple[CrawlerReview, ...]:
        """수집 서비스의 `GET /{platform}/products/{product_id}/reviews`를 호출한다."""

        payload = await self._get(
            reviews_path(platform, product_id),
            params={"limit": limit},
        )
        return _as_reviews(payload)

    async def aclose(self) -> None:
        """이 gateway가 만든 client를 닫는다. 주입받은 client는 건드리지 않는다."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._owns_client:
            self._client = None

    async def __aenter__(self) -> "AsyncHttpCrawlerGateway":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def _get(self, path: str, *, params: dict | None = None) -> object:
        try:
            response = await self._ensure_client().get(path, params=params)
        except httpx.TimeoutException as exc:
            raise _timeout_error(self.base_url, self.timeout) from exc
        except httpx.HTTPError as exc:
            raise _connection_error(self.base_url, exc) from exc

        _raise_for_error(response)
        return response.json()


def _as_platforms(payload: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """플랫폼 목록 응답을 (available, failed)로 옮긴다."""

    if not isinstance(payload, dict):
        raise CrawlerRequestError(
            f"수집 서비스가 플랫폼 목록이 아닌 {type(payload).__name__}을 반환했습니다.",
            status_code=502,
        )
    available = tuple(str(name) for name in payload.get("available", ()))
    failed = tuple(str(name) for name in payload.get("failed", ()))
    return available, failed


def _as_reviews(payload: object) -> tuple[CrawlerReview, ...]:
    """리뷰 목록 응답을 `CrawlerReview`로 옮긴다."""

    if not isinstance(payload, list):
        raise CrawlerRequestError(
            f"수집 서비스가 리뷰 목록이 아닌 {type(payload).__name__}을 반환했습니다.",
            status_code=502,
        )
    return tuple(CrawlerReview.model_validate(item) for item in payload)


def _raise_for_error(response: httpx.Response) -> None:
    """수집 서비스가 오류를 내려주면 사유를 그대로 실어 올린다."""

    if response.is_error:
        raise CrawlerRequestError(
            _error_detail(response),
            status_code=response.status_code,
        )


def _timeout_error(base_url: str, timeout: float) -> CrawlerUnavailableError:
    return CrawlerUnavailableError(
        f"수집 서비스 응답이 {timeout}초 안에 오지 않았습니다: {base_url}"
    )


def _connection_error(base_url: str, exc: httpx.HTTPError) -> CrawlerUnavailableError:
    return CrawlerUnavailableError(
        f"수집 서비스에 연결하지 못했습니다: {base_url} ({type(exc).__name__})"
    )


def _error_detail(response: httpx.Response) -> str:
    """수집 서비스가 내려준 오류 사유를 최대한 그대로 옮긴다."""

    try:
        payload = response.json()
    except ValueError:
        return response.text or f"수집 서비스 오류 (HTTP {response.status_code})"

    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)
