
파이썬의 기본 코드가 동기/블로킹 방식으로 동작함에도 불구하고 `async/await`를 사용하는 결정적인 이유는 **I/O 작업(네트워크 요청, 데이터베이스 조회, 파일 읽기/쓰기 등)이 발생할 때 CPU가 낭비되는 것을 막고, 하나의 스레드에서 수많은 작업을 효율적으로 동시에 처리(동시성, Concurrency)하기 위해서**입니다.
### 1. 동기/블로킹 방식의 한계 (I/O 병목)

웹 서버나 스크립트가 외부 API를 호출하거나 DB에서 데이터를 가져올 때, 응답이 올 때까지 CPU는 아무 일도 하지 못하고 **대기(Blocking)** 상태가 됩니다.

- **예시:** 1개의 요청이 1초 동안 외부 서버의 응답을 기다려야 한다면, 동기 방식에서는 그 1초 동안 해당 스레드가 멈춰 버립니다. 만약 동시에 100명의 사용자가 요청을 보낸다면 순차적으로 처리되거나 수많은 스레드를 만들어야 하므로 시스템 자원이 고갈됩니다.

### 2. `async/await`가 해결하는 방식 (비동기/논블로킹)

`async/await`는 이벤트 루프(Event Loop)를 기반으로 동작합니다.

- **양보(Yielding):** 코드가 `await`를 만나면, 해당 작업이 완료될 때까지 기다리는 것이 아니라 제어권을 **이벤트 루프**에 다시 돌려줍니다.
- **다른 작업 수행:** 이벤트 루프는 그 사이에 대기 중이던 다른 작업(다른 사용자의 요청 등)을 처리합니다.
- **재개(Resuming):** 요청했던 I/O 작업이 끝나면 멈췄던 지점으로 돌아와 나머지 코드를 실행합니다.

### 3. 멀티스레딩(Multi-threading)과 다른 점

파이썬에는 멀티스레드 기능도 있지만, **GIL(Global Interpreter Lock)** 제약 때문에 진정한 의미의 병렬 CPU 연산은 어렵고, 스레드 전환(Context Switching)에 따른 오버헤드가 발생합니다.

반면 `asyncio`는 **단일 스레드(Single-thread)** 안에서 문맥 교환 비용 없이 코루틴 간의 전환만 이루어지기 때문에, 수천 개 이상의 동시 연결을 다루는 I/O Bound 작업(채팅 서버, 웹 크롤러, 고성능 Web API 등)에서 메모리를 적게 쓰면서도 압도적인 성능을 낼 수 있습니다.

### 요약: 언제 사용해야 할까?

|**구분**|**동기/블로킹**|**async/await (비동기)**|
|---|---|---|
|**주요 용도**|일반적인 스크립트, CPU 연산 중심 작업|네트워크 요청, DB 쿼리, 외부 API 연동 등 I/O가 빈번한 작업|
|**장점**|코드가 단순하고 직관적임|적은 자원으로 수많은 동시 요청을 효율적으로 처리 가능|
|**단점**|I/O 대기 시간 동안 자원이 낭비됨|코드가 복잡해지고 비동기 라이브러리를 일관되게 사용해야 함|

---


**파이썬의 일반적인 코드는 기본적으로 동기식(synchronous)이고 블로킹(blocking)**입니다.

그런데 `async/await`를 별도로 사용하는 이유는 **I/O 작업에서 기다리는 동안 CPU를 놀리지 않고 다른 작업을 처리하기 위해서**입니다.

### 1. 일반적인 동기 코드

예를 들어:

```python
response = requests.get("https://example.com")
print(response.text)

response = requests.get("https://python.org")
print(response.text)
```

첫 번째 요청을 보내면:

```text
요청 ────────────────┐
                    │ 서버 응답 기다림
                    │
                    └─ 응답
                       ↓
                    다음 요청
```

첫 번째 서버가 응답할 때까지 **현재 코드 실행이 멈춥니다.**

즉,

```text
요청 A → 기다림 → 완료
                  ↓
요청 B → 기다림 → 완료
                  ↓
요청 C → 기다림 → 완료
```

---

### 2. `async/await`를 사용하면

```python
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()
```

그리고:

```python
tasks = [
    fetch(session, "https://example.com"),
    fetch(session, "https://python.org"),
    fetch(session, "https://github.com"),
]

results = await asyncio.gather(*tasks)
```

실행 흐름은 대략:

```text
요청 A ──┐
         │ 기다리는 동안
요청 B ──┼──→ 다른 요청 처리
         │
요청 C ──┘

       ↓

A 응답
B 응답
C 응답
```

핵심은 **`await`에서 CPU가 멈춰 있는 것이 아니라 이벤트 루프에 제어권을 넘긴다는 것**입니다.

---

## 3. 그래서 `async/await`가 필요한 상황

특히 다음과 같은 작업에서 효과가 큽니다.

- HTTP API 호출
    
- DB 쿼리
    
- 파일 I/O
    
- 소켓 통신
    
- WebSocket
    
- 여러 외부 서비스 호출
    
- 크롤링
    

예를 들어 API 100개를 호출한다고 해보겠습니다.

동기 방식:

```text
API 1 → 1초 대기
API 2 → 1초 대기
API 3 → 1초 대기
...
API 100 → 1초 대기

약 100초
```

비동기 방식:

```text
API 1 ─┐
API 2 ─┤
API 3 ─┤
...    ├── 동시에 요청
API 100┘

약 1~몇 초
```

물론 실제 시간은 서버, 네트워크, 커넥션 풀 등에 따라 달라집니다.

---

## 4. 중요한 점: `async`가 CPU 작업을 빠르게 만드는 것은 아님

이 부분이 상당히 중요합니다.

`async/await`는 **I/O 대기 시간을 효율적으로 사용하는 기술**입니다.

예를 들어:

```python
for i in range(100000000):
    calculate(i)
```

이런 CPU 연산은 `async`로 만든다고 빨라지지 않습니다.

오히려 `async`가 특히 강한 것은:

```text
CPU 작업
████████████████████

I/O 대기
      ..................

      ↑
   이 시간을
   다른 작업에 사용
```

입니다.

---

## 5. Node.js와 비교하면 이해하기 쉽습니다

최근 질문하셨던 Node.js와 연결하면 거의 같은 개념입니다.

Node.js:

```javascript
const results = await Promise.all([
    fetch(url1),
    fetch(url2),
    fetch(url3)
]);
```

Python:

```python
results = await asyncio.gather(
    fetch(url1),
    fetch(url2),
    fetch(url3)
)
```

개념적으로 상당히 비슷합니다.

||Python|Node.js|
|---|---|---|
|비동기 함수|`async def`|`async function`|
|기다리기|`await`|`await`|
|여러 작업 동시 실행|`asyncio.gather()`|`Promise.all()`|
|이벤트 루프|`asyncio`|Node.js Event Loop|
|일반적인 I/O|blocking 가능|non-blocking 중심|

즉, 지난번에 말씀하신 것처럼 **`asyncio.gather()`는 JavaScript의 `Promise.all()`과 상당히 유사합니다.**

### 한 가지 더 중요한 차이

`async/await`라고 해서 **스레드를 여러 개 만드는 것은 아닙니다.**

일반적으로 Python의 `asyncio`는 **하나의 이벤트 루프에서 여러 I/O 작업을 번갈아 실행**합니다.

```text
          Event Loop
              │
     ┌────────┼────────┐
     ↓        ↓        ↓
    API A    API B    API C
     │        │        │
   대기       대기      대기
     │        │        │
     └────→ 다른 작업 처리
```

그래서 **"동기 → 비동기 = 멀티스레드"가 아닙니다.**

한 줄로 정리하면:

> **Python은 기본적으로 동기/블로킹이지만, 네트워크·DB 같은 I/O 작업에서 기다리는 시간을 낭비하지 않기 위해 `async/await`와 이벤트 루프를 사용하는 것입니다.**

그리고 이걸 이해하면 다음 단계로 **`asyncio`의 이벤트 루프가 실제로 `await`를 만났을 때 어떤 식으로 작업을 교대하는지**를 이해하는 게 좋습니다.