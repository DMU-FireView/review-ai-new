"""크롤러 서버(ai-review-crawler) 연동 설정을 모으는 모듈.

역할:
- 크롤러 서버 base URL과 timeout, 재시도 상한을 한곳에서 읽는다.
- 환경변수 이름과 기본값을 이 파일 하나로 고정해 gateway마다 흩어지지 않게 한다.

수정 범위:
- [CONFIG]
- 환경변수 이름·기본값 변경은 배포 담당자와 협의한다. 이미 배포된 이름
  (`CRAWLER_BASE_URL`, `CRAWLER_TIMEOUT`)은 제거하거나 개명하지 않는다.
- analyzer 임계값이나 scoring 정책은 이 파일에 두지 않는다. 여기는 외부 연동 설정만 담는다.

주의:
- pydantic-settings를 쓰지 않는다. 의존성은 이미 있는 pydantic만으로 유지한다.
- 설정 값을 모듈 수준에 캐싱하지 않는다. `load_crawler_settings()`는 호출할 때마다
  환경변수를 다시 읽는다. 프로세스를 재기동하지 않고 설정을 바꿔 끼우는 테스트와
  의존성 주입을 막지 않기 위해서다.

잘못된 환경변수 값을 어떻게 다루는지 (결정과 근거):
- 파싱에 실패하거나 범위를 벗어난 값(예: `CRAWLER_TIMEOUT=not-a-number`, 음수 timeout)은
  **예외로 올리지 않고 기본값으로 되돌리되, WARNING 로그를 남긴다.**
- 근거: 이 값들은 관측 데이터가 아니라 운영 파라미터다. 이 저장소의 "관측되지 않은 값을
  0/중립값으로 보정하지 않는다" 규칙은 리뷰에서 읽어낸 신호를 지어내지 말라는 뜻이고,
  운영 파라미터에는 적용되지 않는다. 오타 하나로 분석 서버 전체가 기동에 실패해
  모든 요청이 멈추는 쪽이, 기본 timeout으로 도는 쪽보다 손해가 크다.
- 대신 조용히 넘기지는 않는다. WARNING 로그로 잘못된 값과 실제 적용된 값을 함께 남겨
  운영자가 부팅 로그에서 바로 알아볼 수 있게 한다.
"""

import logging
import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"
"""크롤러 서버 기본 주소. 로컬에서 `crawler serve`로 띄운 주소와 같다."""

DEFAULT_TIMEOUT = 120.0
"""단발 요청 timeout. 수집은 플랫폼 응답 지연에 좌우되므로 분석보다 여유를 둔다."""

DEFAULT_STREAM_TIMEOUT = 600.0
"""SSE 스트림 timeout. 한 job이 리뷰를 끝까지 흘려보낼 때까지 열어 둔다.

단발 요청보다 훨씬 길다. 스트림은 heartbeat 이벤트로 살아 있음을 알리므로,
단발 요청과 같은 값을 쓰면 정상 수집 중에도 끊긴다.
"""

DEFAULT_MAX_RETRIES = 2
"""한 요청을 다시 시도할 수 있는 최대 횟수(최초 호출 제외).

이 값은 상한만 정의한다. 실제 재시도 실행은 호출 측(gateway 사용자)이 결정한다.
수집은 부작용이 없는 GET이 대부분이라 재시도가 안전하지만, 재시도 여부를
gateway가 임의로 삼키면 실패가 지연으로 둔갑하므로 정책을 여기서 강제하지 않는다.
"""

ENV_BASE_URL = "CRAWLER_BASE_URL"
ENV_TIMEOUT = "CRAWLER_TIMEOUT"
ENV_STREAM_TIMEOUT = "CRAWLER_STREAM_TIMEOUT"
ENV_MAX_RETRIES = "CRAWLER_MAX_RETRIES"


class CrawlerSettings(BaseModel):
    """크롤러 서버 연동에 필요한 설정 한 벌."""

    # 설정은 이 저장소가 정의하는 닫힌 계약이다. 모르는 키는 오타일 가능성이 높으므로 막는다.
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        min_length=1,
        description="크롤러 서버 base URL. 뒤쪽 슬래시는 제거해 보관한다.",
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        gt=0,
        description="단발 HTTP 요청 timeout(초)",
    )
    stream_timeout: float = Field(
        default=DEFAULT_STREAM_TIMEOUT,
        gt=0,
        description="SSE 스트림 timeout(초)",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="한 요청의 최대 재시도 횟수 상한(최초 호출 제외)",
    )

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        """`/reviews` 같은 경로를 붙일 때 슬래시가 겹치지 않게 정규화한다."""

        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be blank")
        return normalized


def load_crawler_settings(env: Mapping[str, str] | None = None) -> CrawlerSettings:
    """환경변수에서 크롤러 연동 설정을 읽는다.

    `env`를 주면 그 매핑을 읽는다. 생략하면 `os.environ`을 그때그때 읽는다.
    잘못된 값 처리 방침은 모듈 docstring 참고.
    """

    source = os.environ if env is None else env

    return CrawlerSettings(
        base_url=_read_base_url(source),
        timeout=_read_positive_float(source, ENV_TIMEOUT, DEFAULT_TIMEOUT),
        stream_timeout=_read_positive_float(
            source, ENV_STREAM_TIMEOUT, DEFAULT_STREAM_TIMEOUT
        ),
        max_retries=_read_non_negative_int(
            source, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES
        ),
    )


def _read_base_url(env: Mapping[str, str]) -> str:
    raw = env.get(ENV_BASE_URL)
    if raw is None:
        return DEFAULT_BASE_URL

    normalized = raw.strip().rstrip("/")
    if not normalized:
        LOGGER.warning(
            "%s 값이 비어 있어 기본값 %s을 사용합니다.", ENV_BASE_URL, DEFAULT_BASE_URL
        )
        return DEFAULT_BASE_URL
    return normalized


def _read_positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning(
            "%s 값 %r을 숫자로 읽지 못해 기본값 %s초를 사용합니다.", name, raw, default
        )
        return default

    if value <= 0:
        LOGGER.warning(
            "%s 값 %r이 0 이하라 기본값 %s초를 사용합니다.", name, raw, default
        )
        return default
    return value


def _read_non_negative_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning(
            "%s 값 %r을 정수로 읽지 못해 기본값 %s를 사용합니다.", name, raw, default
        )
        return default

    if value < 0:
        LOGGER.warning(
            "%s 값 %r이 음수라 기본값 %s를 사용합니다.", name, raw, default
        )
        return default
    return value
