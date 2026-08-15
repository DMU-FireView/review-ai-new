# Re:view AI

이 저장소는 Re:view의 AI 분석 전용 저장소입니다.

Spring에서 정규화된 리뷰 데이터를 전달받아 `P_text`, `P_behavior`, `P_network`를 계산합니다. Meta-Scorer는 사용 가능한 분석 신호를 종합해 최종 RTI를 계산하고, AI 분석 결과를 Spring으로 반환합니다.

과거 `review-ai-db`에서 사용한 네이버 실험, SQLite, Redis 프로토타입 코드는 이 저장소에 그대로 가져오지 않습니다. 또한 실제로 확보되지 않은 원천 데이터는 임의로 생성하지 않습니다.

## 폴더 구조

- `app/main.py`: FastAPI 애플리케이션 진입점
- `app/api/`: Spring 연동 API
- `app/schemas/`: 요청/응답 데이터 스키마
- `app/analyzers/`: `P_text`, `P_behavior`, `P_network` 분석
- `app/scoring/`: Meta-Scorer 및 RTI 계산
- `app/services/`: 분석 흐름 오케스트레이션
- `app/integrations/`: 외부 모델/서비스 연동
- `tests/unit/`: 단위 테스트
- `tests/integration/`: 통합 테스트
- `docs/`: 설계 및 연동 문서
