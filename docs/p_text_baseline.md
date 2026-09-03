# P_text v0 KoELECTRA baseline

이 코드는 현재 RTI 파이프라인과 완전히 분리된 KoELECTRA 오프라인 학습 실험이다. NORMAL은 0, SUSPICIOUS는 1로 매핑한다. 입력 feature는 리뷰 `text` 하나뿐이며 `platform`, `collection_reason`, `source_group`, `annotator_note` 및 기타 사람이 작성한 검수 근거를 tokenizer나 모델에 전달하지 않는다.

Pretrained checkpoint는 `monologg/koelectra-base-v3-discriminator`다. 공개된 Hugging Face config의 `model_type`이 `electra`이고 tokenizer vocab/config가 함께 제공되므로 `AutoTokenizer`로 로드할 수 있다. `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`는 ELECTRA discriminator 본체를 불러오고 NORMAL/SUSPICIOUS 분류용 2-class head를 새로 초기화하는 구조다.

```text
review text
    -> KoELECTRA tokenizer
    -> KoELECTRA sequence classification
    -> NORMAL(0) / SUSPICIOUS(1)
```

## 설치 및 실행

Python 3.11 이상 환경에서 다음을 실행한다.

```powershell
python -m pip install -e ".[ml]"
python -m app.training.p_text.train
```

기본 설정은 `monologg/koelectra-base-v3-discriminator`, 3 epochs, batch size 8, learning rate `2e-5`, max length 256, seed 42다. 예를 들어 CPU에서 더 가벼운 smoke experiment를 실행하려면:

```powershell
python -m app.training.p_text.train --epochs 1 --batch-size 4 --max-length 128
```

CUDA가 사용 가능하면 Hugging Face Trainer가 GPU를 자동 사용하고, 아니면 CPU를 사용한다. 결과는 `artifacts/p_text_baseline/` 아래 `model/`, `metrics.json`, `confusion_matrix.json`, `training_config.json`에 저장된다. 모델과 tokenizer는 각각 `AutoModelForSequenceClassification(..., num_labels=2)` 및 `AutoTokenizer`로 로드된다. 이 명령은 준비 문서일 뿐이며 모델을 현재 P_text analyzer 또는 RTI 계산에 연결하지 않는다.

## 데이터와 분할 정책

`REVIEW_MASTER` 시트에서 NORMAL(0), SUSPICIOUS(1)만 사용하며 UNCERTAIN, DISCLOSED_PROMO, INVALID 및 빈 라벨은 제외한다. 공백과 대소문자를 정규화해 중복을 찾고, 같은 텍스트에 라벨이 충돌하면 해당 그룹 전체를 제거한다. 이 작업은 분할 전에 수행하므로 동일 리뷰가 train/test에 걸쳐 들어가지 않는다.

현재 검수본은 이진 라벨 240행(NORMAL 233, SUSPICIOUS 7)이다. 이 중 라벨 충돌 중복 3그룹(6행)을 제거해 실제 실험에는 234개(NORMAL 230, SUSPICIOUS 4)를 쓴다. 일반 fractional stratified split은 4개의 minority 표본에서 validation/test 양성 표본을 보장하지 못하므로, seed 42로 class별 shuffle 후 약 80/10/10으로 나누면서 각 split에 각 label을 최소 1개 보존한다. 결과는 train 186(184/2), validation 24(23/1), test 24(23/1)다. 어느 class든 고유 표본이 3개 미만이면 오해를 낳는 평가를 만들지 않고 명시적 오류를 낸다.

train 분포로 계산하는 balanced loss weight 공식은 `n / (2 * class_count)`이며 현재 `[0.5054347826, 46.5]`다. 이를 `CrossEntropyLoss`에 적용한다. SUSPICIOUS를 1:1이 되도록 복제하거나 synthetic review를 생성하지 않으며 평가 분포도 변경하지 않는다.

## 결과 해석 제한

SUSPICIOUS 고유 표본이 4개뿐이고 validation/test에는 각각 단 1개만 들어가므로 특히 SUSPICIOUS recall/F1과 confusion matrix의 분산이 극단적으로 크다. 또한 SUSPICIOUS는 실제 댓글 알바 여부가 외부 ground truth로 확인된 데이터가 아니라 사람이 리뷰의 텍스트/관계 신호를 검수해 만든 의심 라벨이다. 따라서 이 결과는 "실제 댓글 알바 탐지 정확도"나 확정 모델 성능으로 해석할 수 없으며, 오직 P_text 학습 파이프라인을 검증하는 v0 baseline이다.
