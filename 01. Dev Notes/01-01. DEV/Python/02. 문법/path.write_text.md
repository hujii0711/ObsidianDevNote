
이 한 줄은 **파이썬 객체를 JSON 문자열로 만들어서 파일에 저장**하는 코드입니다. 안쪽부터 설명드릴게요.

```python
import json
from pathlib import Path

# 1. 저장할 파이썬 데이터 (딕셔너리 또는 리스트)
data = {
	"project": "파이썬 가이드",
	"version": 1.0,
	"author": "홍길동",
	"features": ["가변 인자", "파일 입출력", "JSON 처리"],
	"is_active": True
}

# 2. 저장할 파일 경로 설정 (현재 디렉토리의 'output.json' 파일)
path = Path("output.json")

# 3. JSON 문자열로 변환 후 파일에 쓰기
# - ensure_ascii=False: 한글이 유니코드(\uXXXX)로 깨지는 것을 방지하고 원본 한글로 저장
# - indent=2: 들여쓰기 2칸을 주어 사람이 읽기 쉽도록 포맷팅
# - encoding="utf-8": 한글 및 특수문자가 깨지지 않도록 UTF-8 인코딩 지정
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

## 1단계: `json.dumps(obj, ensure_ascii=False, indent=2)`

파이썬 객체(`obj`, 보통 dict나 list)를 **JSON 형식의 문자열**로 변환합니다.

```python
obj = {"법령명": "주택임대차보호법", "MST": 276291}

json.dumps(obj, ensure_ascii=False, indent=2)
```

**결과 (문자열):**

```json
{
  "법령명": "주택임대차보호법",
  "MST": 276291
}
```

### 옵션 두 개 설명

|옵션|역할|
|---|---|
|`ensure_ascii=False`|한글을 `\uc8fc\ud0dd...` 같은 유니코드 이스케이프가 아니라, **한글 그대로** 출력|
|`indent=2`|한 줄로 압축하지 않고, **2칸 들여쓰기로 예쁘게** 포맷팅|

`ensure_ascii=False`가 없으면:

```json
{"\ubc95\ub839\uba85": "\uc8fc\ud0dd\uc784\ub300\ucc28\ubcf4\ud638\ubc95", "MST": 276291}
```

이렇게 사람이 읽기 힘든 형태가 되므로, 한글 데이터를 다룰 땐 거의 필수로 붙입니다.

## 2단계: `path.write_text(...)`

`path`는 `pathlib.Path` 객체입니다 (앞서 다룬 `Config` 클래스의 `data_root`, `raw_dir` 등과 같은 타입).

```python
from pathlib import Path

path = Path("/repo/data/raw/276291.json")
```

`.write_text()`는 **문자열을 파일에 통째로 쓰는** `Path`의 메서드입니다. 내부적으로는 다음 코드와 동일한 일을 합니다:

```python
# path.write_text(내용, encoding="utf-8") 은 아래와 같음
with open(path, "w", encoding="utf-8") as f:
    f.write(내용)
```

즉 `open()` → `write()` → `close()` 과정을 **한 줄로 축약**해주는 편의 메서드입니다. `with` 블록 없이도 파일이 자동으로 닫힙니다.

### `encoding="utf-8"` 이 왜 필요한가

파일에 한글을 저장할 때 인코딩을 명시 안 하면, OS나 환경에 따라 기본 인코딩이 다르게 적용되어(특히 Windows에서 `cp949`가 기본일 수 있음) 파일이 깨질 수 있습니다. 명시적으로 `utf-8`을 지정해서 어떤 환경에서든 동일하게 저장되도록 보장합니다.

## 전체 흐름 그림

```
obj (파이썬 dict)
        │
        ▼ json.dumps(ensure_ascii=False, indent=2)
"{"법령명": "주택임대차보호법", ...}"  (예쁘게 포맷된 문자열)
        │
        ▼ path.write_text(..., encoding="utf-8")
디스크에 UTF-8로 저장된 .json 파일
```

## 실제 사용 예시 (전체 맥락)

```python
from pathlib import Path
import json

obj = {
    "법령명": "주택임대차보호법",
    "MST": 276291,
    "현행여부": True,
}

path = Path("/repo/data/raw/276291.json")
path.parent.mkdir(parents=True, exist_ok=True)  # 상위 폴더 없으면 생성

path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
```

실행 후 `276291.json` 파일 내용:

```json
{
  "법령명": "주택임대차보호법",
  "MST": 276291,
  "현행여부": true
}
```

## 참고: `json.dump()`를 대신 쓰는 방법과 비교

같은 결과를 얻는 다른 방식도 있습니다:

```python
# 방법 1: 지금 코드 (write_text + dumps)
path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# 방법 2: open + json.dump (전통적인 방식)
with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
```

두 방식 모두 결과는 동일합니다. `write_text()` 방식이 **한 줄로 짧고 간결**해서 `pathlib`을 쓰는 최신 코드베이스에서 자주 선호됩니다.

---

혹시 이 코드가 파이프라인에서 어느 단계(예: `raw_dir`에 원본 저장, `chunks_dir`에 청크 저장)에 쓰이는지 여쭤봐도 될까요? 저장 위치에 따라 파일명 짓는 패턴도 조언드릴 수 있어요.