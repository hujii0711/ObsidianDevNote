
파이썬에서 JSON 다루는 방법을 기본부터 실전까지 정리해드릴게요.

## 1. 기본 모듈: `json`

파이썬 표준 라이브러리에 내장되어 있어 별도 설치 없이 바로 사용 가능합니다.

```python
import json
```

## 2. 핵심 함수 4가지

| 함수             | 방향           | 용도                          |
| -------------- | ------------ | --------------------------- |
| `json.dumps()` | Python → 문자열 | 파이썬 객체를 JSON **문자열**로 변환    |
| `json.dump()`  | Python → 파일  | 파이썬 객체를 JSON **파일**로 저장     |
| `json.loads()` | 문자열 → Python | JSON **문자열**을 파이썬 객체로 변환    |
| `json.load()`  | 파일 → Python  | JSON **파일**을 읽어서 파이썬 객체로 변환 |

> 💡 외우는 팁: **s가 붙으면 string(문자열)**, s가 없으면 파일

## 3. Python ↔ JSON 타입 대응표

|Python|JSON|
|---|---|
|`dict`|`object` `{}`|
|`list`, `tuple`|`array` `[]`|
|`str`|`string`|
|`int`, `float`|`number`|
|`True` / `False`|`true` / `false`|
|`None`|`null`|

## 4. 딕셔너리 → JSON 문자열 (`dumps`)

```python
import json
# <class 'dict'>
data = {
    "법령명": "주택임대차보호법",
    "MST": 276291,
    "현행여부": True,
    "관련법령": None
}

json_str = json.dumps(data)
print(json_str)
# {"\ubc95\ub839\uba85": "\uc8fc\ud0dd..."}  ← 한글이 유니코드로 escape됨
# 타입 확인 및 출력 print(type(json_str)) # 결과: <class 'str'>
```

### 한글 깨짐(유니코드 escape) 방지: `ensure_ascii=False`

```python
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)
# {"법령명": "주택임대차보호법", "MST": 276291, "현행여부": true, "관련법령": null}
```

### 보기 좋게 들여쓰기: `indent`

```python
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)
```

```json
{
  "법령명": "주택임대차보호법",
  "MST": 276291,
  "현행여부": true,
  "관련법령": null
}
```

## 5. JSON 문자열 → 딕셔너리 (`loads`)

```python
json_str = '{"법령명": "주택임대차보호법", "MST": 276291}'
data = json.loads(json_str)

print(data)          # {'법령명': '주택임대차보호법', 'MST': 276291}
print(data["법령명"])  # 주택임대차보호법
print(type(data))    # <class 'dict'>
```

## 6. 파일로 저장하기 (`dump`)

```python
with open("law_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

`open()`에도 `encoding="utf-8"`을 꼭 지정해줘야 파일 자체가 깨지지 않습니다.

## 7. 파일에서 읽기 (`load`)

```python
with open("law_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data["법령명"])
```

## 8. JSONL(JSON Lines) 다루기 — 파이프라인에서 자주 씀

앞서 보신 `chunks_dir` 같은 곳에서 자주 쓰이는 형식입니다. **한 줄에 JSON 객체 하나씩** 저장하는 방식입니다.

```json
{"id": 1, "text": "첫 번째 청크"}
{"id": 2, "text": "두 번째 청크"}
```

**쓰기:**

```python
records = [
    {"id": 1, "text": "첫 번째 청크"},
    {"id": 2, "text": "두 번째 청크"},
]

with open("chunks.jsonl", "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

**읽기:**

```python
records = []
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
```

일반 JSON 파일과 달리 `.jsonl`은 `json.load()`로 통째로 못 읽습니다 (파일 전체가 하나의 유효한 JSON이 아니라서). **한 줄씩 `loads()`** 해야 합니다.

## 9. requests와 함께 쓰기 (API 응답 처리)

```python
import requests

response = requests.get(url)
data = response.json()   # 내부적으로 json.loads(response.text)와 동일
```

`response.json()`은 편의 메서드로, 직접 `json.loads()` 안 해도 됩니다.

## 10. 자주 만나는 에러

### `json.decoder.JSONDecodeError`

문자열이 유효한 JSON 형식이 아닐 때 발생.

```python
json.loads("{법령명: 주택임대차보호법}")  # ❌ 키에 따옴표 없음 → 에러
json.loads('{"법령명": "주택임대차보호법"}')  # ✅ 정상
```

JSON은 **키에 반드시 큰따옴표**가 필요합니다 (파이썬 딕셔너리와 달리 작은따옴표 불가).

### `TypeError: Object of type ... is not JSON serializable`

`dumps()`할 때 JSON이 표현 못 하는 타입(예: `datetime`, `Path`)이 섞여 있을 때:

```python
from pathlib import Path
json.dumps({"path": Path("/repo/data")})  # ❌ TypeError
```

**해결 1**: 미리 문자열로 변환

```python
json.dumps({"path": str(Path("/repo/data"))})  # ✅
```

**해결 2**: `default` 파라미터로 변환 규칙 지정

```python
json.dumps({"path": Path("/repo/data")}, default=str)  # ✅
```

## 실전 팁: XML → dict → JSON 흐름

앞서 다루신 `lxml.etree`와 연결지어 보면, 보통 이런 흐름을 씁니다:

```python
root = etree.fromstring(search_xml.encode("utf-8"))

record = {
    "법령명": root.findtext("법령명한글"),
    "MST": root.findtext("법령일련번호"),
}

with open("law.json", "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
```

XML을 파싱해서 필요한 필드만 뽑아 딕셔너리로 만들고, 그걸 JSON으로 저장하는 패턴이 데이터 파이프라인에서 아주 흔하게 쓰입니다.

---

지금 작업하시는 파이프라인에서 구체적으로 어떤 JSON 처리가 필요하신가요? (예: XML 응답을 JSON으로 변환해서 `raw_dir`에 저장하는 부분)