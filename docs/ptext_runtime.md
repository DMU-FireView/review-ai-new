# P_text runtime / Result Contract v0.5

`app.analyzers.p_text.predict_text_score(content)` lazily loads the local best
model at `models/ptext-koelectra-v1-2epoch-20260929` (resolved relative to the
repository, independent of the working directory). `PTEXT_MODEL_PATH` or the
optional `model_path` argument overrides this location. The former training
inference module re-exports the same interface for compatibility.

The model and tokenizer are cached per process; first load and inference are
serialized to avoid duplicate loads and simultaneous CPU/RAM spikes. Runtime
does not download or train models. Install the project's `ml` extra to supply
PyTorch/Transformers, and deploy the model directory including tokenizer files
and `inference_config.json`. Each server worker owns its own model copy.

Only content is tokenized. `text_score = round((1-p_suspicious)*100)`; the saved
threshold (0.5 for this model) determines the predicted label. Empty/invalid
content or a loading/inference failure returns text_score=-1, probability=null,
label=null. Backend errors are logged; a heuristic score is never substituted.

## Pipeline and contract

Existing `/analysis/crawler-reviews`, `/analysis/collect`, and the final `result`
event of `/analysis/collect/stream` use the same analysis service and flat v0.5
serializer. Crawler clients and collection behavior are unchanged. API requests
must refer to one `(platform, product_id)` even with an explicit `product_key`;
mixed-source requests return 422 instead of losing source identity in v0.5.

The existing behavior and network analyzers run only on supplied evidence.
Unavailable values are internally represented by the existing missing-signal
type and serialized as -1. Empty text does not count as a network comparison.
RTI uses the existing weights text=0.50, behavior=0.30, network=0.20, excluding
missing signals and renormalizing the remaining weights:

`RTI = sum(score * weight for available signals) / sum(available weights)`

`calculate_rti(text_score, behavior_score=-1, network_score=-1)` exposes the
sentinel-aware utility. Public RTI is rounded to one decimal; the displayed RTI
determines the level: >=70 safe, >=40 warn, otherwise danger. All missing gives
RTI=-1 and **level=null**, preserving the existing absent-level convention.
There are no null scores, nested signals, or availability fields in the result.

Existing rule reasons are emitted as `SOURCE_CODE` strings, for example
`NETWORK_SIMILAR_REVIEW_PATTERN`. Text rules supply observable reasons only;
model probability never generates a specific promotional reason by itself.

Example response for a mocked model score of 87 with no other evidence:

```json
{"platform":"kurly","product_id":"p","review_count":1,"results":[{"review_id":"r","rti":87.0,"level":"safe","text_score":87.0,"behavior_score":-1,"network_score":-1,"reasons":[]}]}
```

## Commands (PowerShell, repository root)

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\venv\Scripts\python.exe -X utf8 -m app.analyzers.p_text --content "2주 정도 사용해봤는데 세안 후 당김이 덜하고 자극도 없어서 만족합니다." --content "피부 보습과 탄력 개선에 탁월한 제품입니다. 아직 사용 전이지만 효과는 확실하니 꼭 구매하세요."
```

POST `/analysis/crawler-reviews` with:

```json
{"reviews":[{"platform":"kurly","product_id":"p","review_id":"r","content":"2주 정도 사용해봤는데 세안 후 당김이 덜하고 자극도 없어서 만족합니다."}]}
```

Tests use mock inference, not the model. The standalone smoke command loads the
saved model and checks executable inference; it is not a performance evaluation.

Actual smoke results for the two sentences above, in order:

| Input | suspicious_probability | text_score | predicted_label |
|---|---:|---:|---|
| Two weeks of use, less tightness and no irritation | 0.002177400514483452 | 100 | NORMAL |
| Not yet used, asserts effects and urges purchase | 0.0024832640774548054 | 100 | NORMAL |

The suspicious-style example was also classified NORMAL. No threshold or model
was changed to fit this example; this is a concrete limitation of the current
model on an unseen sentence, not evidence of detection accuracy.

## Preserved evaluation limitations

Best epoch 1, validation/test F1=1.0, both real NORMAL subgroup FPRs=0%, synthetic
recall=100%. **Validation/test SUSPICIOUS examples are entirely synthetic; real
positive recall remains unvalidated.** Model outputs and text_score are not
calibrated authenticity probabilities. Original training reports/checkpoints
and datasets are not modified by runtime integration.
