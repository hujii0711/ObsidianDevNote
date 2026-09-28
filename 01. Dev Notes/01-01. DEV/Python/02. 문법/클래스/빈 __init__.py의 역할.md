
**"이 디렉터리는 Python 패키지다"** 라고 인터프리터에게 알려주는 마커 파일입니다.

---

### 없으면 어떻게 되나?

```
project/
├── main.py
└── index/
    ├── embedder.py
    └── build_index.py
```

`__init__.py`가 없으면 `index/`는 그냥 폴더입니다.

```python
# main.py
from index import embedder   # ❌ ModuleNotFoundError
import index.build_index     # ❌ ModuleNotFoundError
```

---

### 있으면?

```
index/
├── __init__.py   ← 빈 파일이어도 OK
├── embedder.py
└── build_index.py
```

```python
from index import embedder      # ✅
import index.build_index        # ✅
from index.embedder import BGEEmbedder  # ✅
```

---

### 정리

||`__init__.py` 없음|`__init__.py` 있음 (빈 파일)|
|---|---|---|
|디렉터리 인식|일반 폴더|Python 패키지|
|내부 모듈 임포트|❌|✅|
|파일 내용 필요|—|없어도 됨|

> Python 3.3+부터는 `__init__.py` 없이도 임포트되는 "네임스페이스 패키지"가 생겼지만, 명시적으로 빈 파일을 두는 것이 여전히 표준 관행입니다.