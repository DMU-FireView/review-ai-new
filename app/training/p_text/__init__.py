"""리뷰 텍스트만 사용하는 P_text 학습 baseline을 제공하는 패키지.

역할: [AI 학습] 공통 label 상수와 오프라인 P_text 모듈을 제공한다.
수정 범위: label mapping 또는 패키지 공개 항목 변경은 AI 담당자 검토가
필요하다.
주의: collection_reason과 진단 metadata는 모델 입력으로 사용하지 않으며,
이 패키지를 import하는 것만으로 학습이나 추론이 실행되어서는 안 된다.
"""

from .data import LABEL_MAPPING, EXCLUDED_LABELS

__all__ = ["LABEL_MAPPING", "EXCLUDED_LABELS"]

