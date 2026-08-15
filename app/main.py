"""Re:view AI 서비스의 FastAPI 애플리케이션 진입점.

수정 범위:
- [ENTRY POINT]
- router 등록과 애플리케이션 설정을 담당한다.
- analyzer 또는 scoring 로직을 직접 작성하지 않는다.
"""

from fastapi import FastAPI

app = FastAPI(title="Re:view AI")
