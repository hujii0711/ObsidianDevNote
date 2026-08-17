
파이썬 프로젝트를 생성하고 `pip3`를 통해 `requests` 라이브러리를 설치한 뒤, HTTP 통신을 테스트하는 전체 과정을 순서대로 안내해 드립니다.

## 1. 프로젝트 폴더 생성 및 이동

터미널(또는 명령 프롬프트)을 열고 프로젝트를 진행할 폴더를 생성한 뒤 이동합니다.

```
mkdir my_http_project
cd my_http_project
```

## 2. 가상 환경 생성 및 활성화 (권장)

전역 환경 오염을 방지하기 위해 가상 환경을 생성하고 활성화하는 것이 좋습니다.
전역 환경과 별개로 이 프로젝트에만 `requests`가 설치됩니다.

- **macOS / Linux:**
```
python3 -m venv pipProject
source pipProject/bin/activate
```

**`-m` 옵션의 역할**
`-m`은 module(모듈)의 약자로, Python이 설치된 라이브러리/모듈 중 하나를 실행하라는 뜻입니다. 즉, `python3 -m pip`는 "현재 실행 중인 `python3` 환경 안에 설치된 `pip` 모듈을 실행해라"라는 의미입니다.

- **Windows:**
```
python -m venv pipProject
pipProject\Scripts\activate
```

## 3. requests 라이브러리 설치

`pip3`를 사용하여 `requests` 패키지를 설치합니다.

```
python3 -m pip install requests
```

설치가 완료되었는지 확인하려면 아래 명령어로 버전을 확인해 볼 수 있습니다.

```
pip3 show requests
```

의존성 기록 (다른 사람과 공유할 때 핵심)

```bash
pip3 freeze > requirements.txt
```

다른 컴퓨터에서 그대로 재현
```shell
pip install -r requirements.txt
```

## 4. HTTP 통신 테스트 코드 작성

프로젝트 폴더 내에 `test.py` 파일을 생성하고, 공개 테스트 API(예: JSONPlaceholder)를 대상으로 GET 및 POST 요청을 테스트하는 코드를 작성합니다.

```python
import requests

def test_http_requests():
    # 1. GET 요청 테스트 (데이터 조회)
    print("--- GET 요청 테스트 ---")
    get_url = "https://jsonplaceholder.typicode.com/posts/1"
    
    response = requests.get(get_url)
    
    # 상태 코드 확인 (200은 성공)
    print(f"상태 코드: {response.status_code}")
    
    if response.status_code == 200:
        # JSON 응답 데이터 파싱
        data = response.json()
        print("응답 본문 (JSON):")
        print(data)
    else:
        print("요청 실패")

    print("\n" + "="*40 + "\n")

    # 2. POST 요청 테스트 (데이터 전송)
    print("--- POST 요청 테스트 ---")
    post_url = "https://jsonplaceholder.typicode.com/posts"
    
    payload = {
        "title": "파이썬 HTTP 통신 테스트",
        "body": "requests 라이브러리 테스트 중입니다.",
        "userId": 1
    }
    
    response = requests.post(post_url, json=payload)
    
    print(f"상태 코드: {response.status_code}")
    
    if response.status_code in [200, 201]:
        print("생성된 데이터 응답:")
        print(response.json())
    else:
        print("요청 실패")

if __name__ == "__main__":
    test_http_requests()
```

## 5. 테스트 실행

작성한 파이썬 파일을 실행하여 정상적으로 응답이 오는지 확인합니다.

```
python3 test.py
```

정상적으로 통신이 완료되면 API 서버로부터 받아온 데이터와 상태 코드(`200 OK`, `201 Created` 등)가 터미널에 출력됩니다.

---

## 핵심 차이

|명령|의미|
|---|---|
|`pip install requests`|`pip`라는 **독립된 실행 파일**을 직접 실행|
|`pip3 install requests`|`pip3`라는 실행 파일을 직접 실행 (파이썬 3 전용으로 명명된 것)|
|`python3 -m pip install requests`|**python3 인터프리터를 먼저 실행**하고, 그 안에서 `pip` 모듈을 실행|

## 왜 문제가 생길 수 있나

`pip`, `pip3`는 PATH에 등록된 **별도의 실행 파일**이라, 시스템에 파이썬 버전이 여러 개 있거나 가상환경이 여러 개 있으면 **엉뚱한 파이썬 환경에 설치될 위험**이 있어요.`pip` 파일의 첫 줄(shebang)에 어떤 파이썬을 쓸지 하드코딩돼 있는 게 보이시죠. 이 shebang 경로가 PATH 우선순위와 꼬이면 **의도치 않은 파이썬에 설치**되는 사고가 생깁니다. 예를 들어:

- 시스템에 파이썬 3.10, 3.12가 둘 다 있음
- `pip install`이 PATH상 먼저 잡히는 `pip`를 실행하는데, 그게 3.10용 pip일 수 있음
- 반면 `python3`은 3.12를 가리킬 수 있음
- → 설치는 3.10에 되는데 실행은 3.12로 해서 `ModuleNotFoundError` 발생

## 결론 및 권장 사항

**`python3 -m pip install requests`가 가장 안전합니다.**

이유: `python3 -m pip`는 "지금 내가 쓰는 이 `python3`에 연결된 pip를 실행해라"는 뜻이라, **어떤 파이썬에 설치되는지 명확하고 확실**합니다. 반면 `pip`/`pip3`는 PATH에 걸린 아무 실행 파일이나 부를 수 있어 모호합니다.

|상황|추천|
|---|---|
|가상환경 안에서 (activate된 상태)|셋 다 동일 (안전)|
|시스템에 파이썬 여러 버전 공존|`python3 -m pip` (안전)|
|스크립트/CI에서 확실성이 중요할 때|`python3 -m pip`|
|그냥 터미널에서 빠르게|`pip` (대개 문제 없음)|