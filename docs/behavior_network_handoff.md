# AI Python 모듈 인계 — 2026-09-30

P_text 기본 모델은 `models/ptext-koelectra-v1-2epoch-20260929` 그대로다.
generalized v2는 보존하며 채택하지 않는다. 이 작업에서 학습·dataset·checkpoint는 변경하지 않았다.

## 설치/환경

- 검증 환경: Python 3.12.10. 프로젝트 지원 범위는 `pyproject.toml` 기준 Python 3.11 이상이다.
- 프로젝트 루트에서 runtime/ML 의존성 설치: `python -m pip install -e ".[ml]"`
- 테스트 의존성 설치: `python -m pip install pytest pytest-asyncio`
- 전체 테스트: `python -m pytest -q -p no:cacheprovider`
- runtime 기본 모델: `models/ptext-koelectra-v1-2epoch-20260929`
- P_text 고정 threshold: `0.5`
- 최종 P_text 모델 크기: 약 431.83 MiB
- runtime은 CUDA 사용 가능 시 GPU를 자동 선택하고, CUDA가 없으면 CPU로 fallback한다.
- 현재 로컬 최종 검증은 CPU 환경에서 완료했다. 실제 GPU 서버 배포 시 CUDA inference smoke test 1회를 권장한다.
- FastAPI multi-worker에서는 worker마다 모델을 별도로 로드할 수 있으므로 worker 수 설정 시 RAM/VRAM 사용량을 고려해야 한다.
- generalized v2는 runtime 기본 모델로 사용하지 않는다.

## 입력 조사 근거

- `app/schemas/crawler.py:CrawlerReview`: 필수 platform/product_id/review_id/content.
  선택 rating/author/written_at/option/images/helpful_count/collected_at.
- `app/services/analysis.py:ReviewAnalysisInput`: review_id/product_id/content,
  선택 user_id/review_date/verified_purchase/account_created_at/user_review_dates.
  후자의 행동 필드는 서비스와 명시적 unit fixture에서만 확인되며 실제 수집 가용성을 뜻하지 않는다.
- `tests/unit/test_crawler_adapter.py`, `test_crawler_client.py`, `test_crawler_stream.py` 및
  integration fixtures는 계약 예시이며 실제 수집 증거와 구분한다.
- 실제 sample은 인접 저장소 `../ai-review-crawler/data/`의 JSON 5개, 총 17건을 읽었다.
  현재 AI 저장소에는 해당 crawler 원본 JSON이 없어 인접 저장소를 읽기 전용으로 확인했다.
- collector 근거: `../ai-review-crawler/src/review_crawler/core/models.py`,
  `collectors/{kurly,ohouse,elevenst,oliveyoung,musinsa,ably,auction}/collector.py`.

| 플랫폼 | 실제 JSON 건수 | author | written_at | rating | helpful_count | images |
|---|---:|---|---|---|---|---|
| kurly | 8 | 관측, 마스킹 | 관측 | null | 관측 | 배열 관측, 빈 배열 포함 |
| ohouse | 3 | null | 관측 | 관측 | 관측 | 관측 |
| elevenst | 3 | 관측, 마스킹 | 관측 | 관측 | 관측 | 관측 |
| ably | 3 | null | null | null | null | 관측 |
| oliveyoung | 없음 | collector 매핑만 확인 | 매핑만 확인 | 매핑만 확인 | recommendCount 매핑 | 매핑만 확인 |
| musinsa | 없음 | userNickName 매핑만 확인 | 매핑만 확인 | 매핑만 확인 | likeCount 매핑 | 매핑만 확인 |
| auction | 없음 | collector 매핑만 확인 | 매핑만 확인 | 매핑만 확인 | 매핑만 확인 | 매핑만 확인 |

모든 플랫폼의 표준 crawler 계약에 user_id, verified_purchase, account_age_days,
reviews_written_today, account_created_at은 없다. 이들 필드의 실수집 가용성은 확인되지 않았다.
`created_at/registered_at`도 표준 리뷰 필드가 아니며 리뷰 날짜의 실제 이름은 `written_at`이다.
`like_count`의 통합 필드는 `helpful_count`, `image_count`는 제공되지 않지만 `len(images)`는 파생 가능하다.
빈 images 배열이 수집 실패인지 실제 사진 없음인지는 구분할 수 없다.
rating/helpful_count/images/option/collected_at은 관측되더라도 행동 의심 점수로 사용하지 않는다.
닉네임은 마스킹 여부와 관계없이 고유 계정 ID로 승격하지 않는다.

## Behavior MVP

기존 `app.analyzers.behavior.analyze_behavior(BehaviorInput(...))`를 재사용한다.
내부 기존 인터페이스 `p_behavior=None`은 유지하며 public `behavior_score` 속성은 -1을 반환한다.
최종 JSON에는 `behavior_score`만 노출한다.

| 실제 제공된 근거 | 점수 | 가중치 | 감점 reason suffix |
|---|---|---:|---|
| verified_purchase | True 100 / False 70 | .5 | PURCHASE_NOT_VERIFIED |
| review_date와 account_created_at | 가입 7일 미만 80 / 나머지 100 | .2 | NEW_ACCOUNT |
| stable user_id + review_date + user_review_dates 2개 이상 | 같은 날짜 3건 이상 85 / 미만 100 | .3 | MULTIPLE_REVIEWS_SAME_DAY |

`sum(관측 신호 점수 × 가중치) / sum(관측 신호 가중치)`.
누락 신호는 제외하고 모두 없으면 -1. 리뷰보다 늦은 계정 생성일은 사용하지 않는다.
계정 나이/빈도 숫자를 입력에 있다고 가정하지 않고 기존 날짜 필드로만 파생한다.
원천 날짜는 호출자가 동일 시간대 기준으로 제공해야 한다. 이력은 동일 실제 사용자에 대한
중복 없는 관측 이력이어야 하며, 표본 이력은 실제 작성 빈도의 하한이다.
상수는 `app/analyzers/behavior.py`에 모았다. 검증되지 않은 MVP 휴리스틱이다.

## Network MVP

`analyze_network_batch`는 동일 상품 batch에 Hybrid A similarity를 계산한다.
`Hybrid A = 0.40 × canonical TF-IDF cosine + 0.60 × canonical bigram Dice`이다.
canonical 표현은 Unicode NFKC와 casefold를 적용한 뒤 공백과 문장부호를 제거하고
한글·영숫자만 유지한다. 이 표현은 비교에만 사용하며 원본 content를 수정하지 않는다.
TF-IDF는 canonical 문자 2~4 gram, 등장 횟수 TF, `1 + log((1+N)/(1+df))` IDF,
L2 정규화를 사용한다. 외부 모델이나 network 호출은 없다.

strong similarity 기준은 Hybrid A `>= 0.85`이다. 다른 strong review 수를 `k`,
최대 similarity를 `m`, 상위 3개 평균을 `t`라 할 때 다음 연속 점수를 사용한다.

```text
s(x) = clip((x - 0.50) / 0.50, 0, 1)
c(k) = 1 - exp(-k / 2)
network_score = clip(100 - 55*s(m)^2 - 30*s(t)^2 - 15*c(k), 0, 100)
```

결과는 소수점 한 자리다. strong peer 1~4건은 `NETWORK_SIMILAR_REVIEW_PATTERN`,
5건 이상은 `NETWORK_SIMILAR_REVIEW_CLUSTER`를 생성한다. 충분한 길이의 canonical
exact duplicate는 similarity 1.0이다. 정규화 후 한글·영숫자 정보량이 10자 미만인
짧은 리뷰는 evidence에서 제외한다. 짧은 review 한 건이 섞여도 다른 긴 review의 batch
분석은 계속된다. 해당 review에 비교 가능한 evidence가 없으면 network_score는 -1.0이고
reason을 만들지 않는다. author/user_id는 관계 계산에 쓰지 않는다.

시간 복잡도는 리뷰 수에 대해 O(N²)이며 dense NxN 행렬은 보관하지 않는다.
0.85 threshold는 아직 대규모 labeled near-duplicate dataset으로 검증되지 않은 MVP 기준이다.
network similarity는 조작 확정이 아니라 이상 신호이며 batch 구성에 따라 TF-IDF가 달라질 수 있다.

## Python 호출과 Contract

```python
import json
from pathlib import Path
from app.services.analysis import analyze_reviews

reviews = json.loads(Path(
    '../ai-review-crawler/data/kurly/reviews_20260730_194312.json'
).read_text(encoding='utf-8'))
result = analyze_reviews(
    platform='kurly', product_id='1001196970', reviews=reviews,
)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

동기 함수이며 FastAPI가 필수는 아니다. dict 또는 ReviewAnalysisInput 목록을 받는다.
dict의 written_at을 review_date로 변환한다. author는 무시한다. platform/product_id 혼합,
중복 review_id는 오류다. 필수 본문이 비어 있으면 해당 신호는 unavailable이다.
기존 P_text predictor의 프로세스 내 모델 캐시를 유지한다. 기본 경로는 v1이며 기존
PTEXT_MODEL_PATH 환경변수 override 기능은 그대로이므로 배포 시 설정을 확인한다.
async 서버 호출에서는 담당자가 threadpool 사용 등 CPU 작업 실행 방식을 정하면 된다.

입력 필드는 다음과 같다.

| 계층 | 필수 | 선택 |
|---|---|---|
| crawler-shaped dict | review_id, content; 함수 인자로 platform, product_id | platform, product_id, written_at 및 crawler 부가 필드 |
| ReviewAnalysisInput | review_id, product_id, content | user_id, review_date, verified_purchase, account_created_at, user_review_dates |

dict에서 실제로 분석 입력으로 읽는 선택 행동 필드는 `user_id`, `review_date`,
`verified_purchase`, `account_created_at`, `user_review_dates`다. `written_at`은
`review_date`가 없을 때 매핑된다. crawler의 `author`는 user_id로 매핑하지 않는다.

RTI는 기존 `.50/.30/.20` 가중치로 -1 제외 후 재정규화하고 소수점 한 자리로 반환한다.
70 이상 safe, 40 이상 warn, 미만 danger. 전부 -1이면 rti=-1, level=null이다.
score null, signals, available, rti_available, unavailable_reasons는 출력에 없다.

Result Contract v0.5 serializer는 score를 float로 출력하므로 계산 불가 sentinel은 `-1.0`이다.
score 의미는 `0.0~100.0 = 계산된 신뢰 점수`, `-1.0 = 해당 신호 계산 불가`다.
flat result만 사용하며 `signals`, `available`, `rti_available`, `unavailable_reasons`는 없다.
모든 score는 null이 아니고, `level`만 rti가 -1.0일 때 null이다.

전체 출력 예시:

```json
{"platform":"kurly","product_id":"1001196970","review_count":1,"results":[
  {"review_id":"136082007","rti":100.0,"level":"safe","text_score":100.0,
   "behavior_score":-1.0,"network_score":-1.0,"reasons":[]}
]}
```

## 최종 통합 fixture

production 데이터가 아닌 통합 검증용 fixture에서 P_text를 80.0으로 고정하고,
미인증 구매·가입 2일 계정·당일 3회 작성 이력과 유사 문장 2건을 함께 전달했다.
첫 리뷰는 behavior 76.5, network 39.1이었다. 수동 계산과 serializer 결과는 다음과 같다.

`80.0 × .50 + 76.5 × .30 + 39.1 × .20 = 70.77 → 70.8`

유사 리뷰에는 `NETWORK_SIMILAR_REVIEW_PATTERN`이 생성되며, 무관한 세 번째 리뷰는
network 100.0이고 behavior/network reason이 없다. 누락 행동 필드는 감점하지 않는다.

## FastAPI 담당자용 최소 정보

- import: `from app.services.analysis import analyze_reviews`
- 호출: `analyze_reviews(platform=..., product_id=..., reviews=...)`
- 입력: `Sequence[dict]` 또는 `Sequence[ReviewAnalysisInput]`; 같은 platform/product_id의 리뷰 batch
- 반환: Pydantic 검증을 거친 Result Contract v0.5 `dict`
- 예외: 빈 platform/product_id/review_id, 중복 review_id, 혼합 platform/product_id,
  잘못된 타입·날짜에는 `ValueError` 또는 validation 예외가 발생할 수 있다.
- P_text 추론 실패는 예외 전파 대신 text_score -1.0으로 변환한다.
- 최초 호출은 로컬 KoELECTRA 모델 로딩 때문에 지연될 수 있고 이후 프로세스 내 캐시를 사용한다.
- P_text 로드와 추론은 lock으로 직렬화된다. 함수는 동기 CPU 작업이며 async FastAPI에서는
  threadpool로 넘겨 event loop를 막지 않아야 한다. 다중 worker는 worker마다 모델을 로드한다.

## 검증과 한계

실제 sample 17건 모두 text_score=100, behavior_score=-1, batch network_score=100이었다.
Hybrid A max similarity 범위는 0.00~0.13이었고 network reason은 없었다.
각 sample의 첫 리뷰만 분석하면 network_score=-1이었다. 전체 결과와 원본 hash는
`reports/behavior_network_smoke.json`에 기록했다. 첫 호출 포함 4건 9.596초,
이후 batch+single 4~6건은 .182~.395초였다.
실제 sample에는 사용 가능한 행동 근거나 높은 본문 유사도 pair가 없었다.
해당 사례는 sample을 위조하지 않고 명시적인 unit fixture로만 검증했다.
모델이 100점을 반환했다는 사실은 이 리뷰들의 진위를 검증했다는 뜻이 아니다.

최종 전체 pytest 결과는 이 문서와 함께 전달된 검증 보고를 기준으로 확인한다.
`venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`.
신규 테스트는 모델을 mock하며 학습 모델을 로드하지 않는다.
실제 KoELECTRA 로드는 별도 smoke에서만 수행했다.

P_text의 실제 positive recall은 여전히 검증되지 않았다. Behavior 점수·similarity threshold는
라벨 검증 없는 임시 정책이고, 정상적인 상투적 문장도 network 오탐이 가능하다.
본문 유사성은 조직적 조작이나 동일 계정의 증거가 아니다. batch 구성에 따라 IDF/점수가 변한다.
실수집 행동 필드가 없는 현재 계약에서는 behavior=-1이 정상 동작이다.
API endpoint, 모델, 학습 dataset은 이번 작업에서 수정하지 않았다. commit/push도 하지 않았다.
