
두 로더 모두 PDF를 LangChain `Document` 리스트로 바꿔 주지만, **추출 결과의 형태**가 다릅니다. 핵심 차이는 **일반 텍스트 vs 마크다운**입니다.

| 구분     | `PyPDFium2Loader`                   | `PyMuPDF4LLMLoader`                  |
| ------ | ----------------------------------- | ------------------------------------ |
| 패키지    | `langchain-community` + `pypdfium2` | `langchain-pymupdf4llm` (별도 파트너 패키지) |
| 엔진     | Google PDFium (Chrome의 PDF 엔진)      | PyMuPDF + `pymupdf4llm`              |
| 출력 형태  | 레이아웃 정보가 없는 **일반 텍스트**              | **마크다운** (제목 `#`, 굵게, 목록, 표)         |
| 표 처리   | 셀 내용이 줄글로 섞여서 나옴                    | 마크다운 표(`\| a \| b \|`)로 변환           |
| 제목/구조  | 인식하지 않음                             | 글자 크기를 기준으로 `#`, `##` 제목을 추정         |
| 속도     | 매우 빠름                               | 상대적으로 느림 (레이아웃 분석 때문)                |
| 라이선스   | Apache/BSD 계열 (상업적 사용 부담 적음)        | **AGPL** (PyMuPDF). 상업 서비스라면 확인 필요   |
| 적합한 용도 | 단순 텍스트 위주 PDF, 대량 처리                | RAG용 청킹, 표나 구조가 중요한 문서               |

**사용 예시**

```python
# 1) PyPDFium2Loader — 최신 import 경로는 langchain_community
from langchain_community.document_loaders import PyPDFium2Loader

docs = PyPDFium2Loader("sample.pdf").load()
print(docs[0].page_content[:500])   # 줄바꿈만 있는 평문
```

```python
# 2) PyMuPDF4LLMLoader
# pip install langchain-pymupdf4llm
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

loader = PyMuPDF4LLMLoader(
    "sample.pdf",
    mode="page",              # "page": 페이지별 Document / "single": 하나로 합침
    table_strategy="lines",   # 표 인식 방식
)
docs = loader.load()
print(docs[0].page_content[:500])   # # 제목, | 표 | 등 마크다운
```

**선택 기준**

- RAG에서 청킹 품질이 중요하다면 `PyMuPDF4LLMLoader`가 좋습니다. 마크다운 제목을 기준으로 `MarkdownHeaderTextSplitter`를 써서 의미 단위로 자를 수 있고, 표 구조도 유지되기 때문에 LLM이 수치를 더 정확히 읽습니다.
- 단순한 텍스트 문서를 많이 빠르게 처리해야 하거나 AGPL 라이선스가 부담된다면 `PyPDFium2Loader`가 낫습니다.
- `from langchain.document_loaders ...` 경로는 구버전 방식입니다. 최신 버전에서는 `langchain_community.document_loaders`를 사용하세요.
- 두 로더 모두 스캔한 이미지 PDF에서는 텍스트가 거의 나오지 않습니다. 이런 경우에는 OCR 옵션(`extract_images=True` 등)이나 별도의 OCR 도구가 필요합니다.