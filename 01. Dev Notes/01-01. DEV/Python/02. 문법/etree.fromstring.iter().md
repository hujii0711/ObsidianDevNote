
`root.iter()`는 XML 트리의 **모든 하위 요소(Element)를 순회**하는 메서드입니다. 자세히 설명드릴게요.

## 기본 동작: 트리 전체를 평평하게 순회

XML 트리는 계층 구조(부모-자식)인데, `root.iter()`는 이 계층을 무시하고 **모든 노드를 하나씩 순서대로** 훑어줍니다.

## 예시 XML

```xml
<법령검색>
  <법령>
    <법령명한글>주택임대차보호법</법령명한글>
    <법령일련번호>276291</법령일련번호>
  </법령>
  <법령>
    <법령명한글>상가건물임대차보호법</법령명한글>
    <법령일련번호>269797</법령일련번호>
  </법령>
</법령검색>
```

```python
root = etree.fromstring(xml_data.encode("utf-8"))

for row in root.iter():
    print(row.tag, "→", row.text)
```

**출력:**

```
법령검색 → None
법령 → None
법령명한글 → 주택임대차보호법
법령일련번호 → 276291
법령 → None
법령명한글 → 상가건물임대차보호법
법령일련번호 → 269797
```

## 순회 순서 (중요)

**DFS(깊이 우선 탐색), Pre-order 방식**입니다:

1. `root` 자기 자신부터 포함해서 시작
2. 첫 번째 자식으로 내려감
3. 그 자식의 자식들을 다 순회
4. 형제 노드로 이동
5. 반복

```
법령검색 (자기 자신도 포함!)
  └─ 법령 (1번째)
       ├─ 법령명한글
       └─ 법령일련번호
  └─ 법령 (2번째)
       ├─ 법령명한글
       └─ 법령일련번호
```

## ⚠️ 주의: `root` 자기 자신도 포함됨

`root.iter()`는 자식들만 도는 게 아니라 **`root` 자신도 첫 번째 결과로 포함**합니다. 그래서 위 예시에서 `법령검색`(최상위 태그)도 출력에 나온 겁니다.

## `root.iter(태그명)` — 특정 태그만 필터링

인자로 태그 이름을 넣으면 **그 태그만** 걸러서 순회합니다. 실무에서 훨씬 많이 쓰는 패턴입니다.

```python
for row in root.iter("법령"):
    이름 = row.findtext("법령명한글")
    mst = row.findtext("법령일련번호")
    print(이름, mst)
```

**출력:**

```
주택임대차보호법 276291
상가건물임대차보호법 269797
```

`법령` 태그만 쏙쏙 골라서 도는 거라, 실전에서 검색 결과 리스트를 순회할 때 자주 씁니다.

## 비교: 다른 순회 방법들과의 차이

|메서드|대상|특징|
|---|---|---|
|`root.iter()`|**모든 하위** 요소 (재귀적으로 전체)|깊이 상관없이 전부|
|`root` (for로 직접 순회)|**바로 아래 자식만**|1단계 깊이만|
|`root.findall(".//태그")`|특정 태그를 **리스트**로|iter와 비슷하지만 결과가 list|
|`root.iter("태그")`|특정 태그만 **순회(iterator)**|메모리 효율적|

### 코드로 비교

```python
# 1. root.iter() — 전체 재귀 순회
for el in root.iter():
    print(el.tag)  # 법령검색, 법령, 법령명한글, 법령일련번호, 법령, ...

# 2. for row in root — 바로 아래 자식만
for el in root:
    print(el.tag)  # 법령, 법령  (2번만 출력, 손자는 안 나옴)

# 3. root.findall(".//법령") — list 반환
법령_list = root.findall(".//법령")
print(type(법령_list))  # <class 'list'>

# 4. root.iter("법령") — iterator 반환 (제너레이터처럼 동작)
법령_iter = root.iter("법령")
print(type(법령_iter))  # <class 'generator'> 유사 (lxml에서는 iterator 객체)
```

## `iter()` vs `findall()` 언제 뭘 쓰나

- **`iter()`**: 큰 XML 파일을 **메모리 효율적으로** 순회할 때 (한 번에 하나씩 처리, 리스트로 다 안 만듦)
- **`findall()`**: 결과를 **리스트로 받아서** `len()`, 인덱싱, 슬라이싱 등을 하고 싶을 때

## 실전 패턴: 판례/법령 검색 결과 파싱

```python
root = etree.fromstring(search_xml.encode("utf-8"))

results = []
for row in root.iter("law"):   # 실제 태그명은 API 응답에 맞게
    results.append({
        "법령명": row.findtext("법령명한글"),
        "MST": row.findtext("법령일련번호"),
    })

print(results)
```

---

혹시 지금 다루시는 XML 응답의 실제 태그 구조를 보여주시면, `root.iter()`에 어떤 태그를 넣어야 원하는 데이터만 정확히 뽑을 수 있는지 구체적으로 짚어드릴 수 있어요.