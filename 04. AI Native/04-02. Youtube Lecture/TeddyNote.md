# 목차

1. 변수
2. 데이터타입
3. type
4. int(정수), float(실수)
5. str(문자열)
6. bool(참/거짓)
7. None
8. 타입변환
9. list 개요
10. list 생성
11. list 특징
12. list 추가
13. list 정렬
14. list 역정렬
15. list 삽입 및 삭제
16. list 카운트
17. list indexing
18. list slicing
19. list step //1(토)
20. tuple 개요
21. tuple 연산
22. tuple 기능
23. set 개요
24. set 교집합, 합집합, 차집합
25. dict 개요
26. dict 추가, 업데이트
27. dict 삭제
28. dict 길이, 초기화
29. 문자열 개요 및 생성
30. print()
31. % 문자열 포맷팅
32. .format() 문자열 포맷팅
33. f 문자열 포맷팅
34. 문자열 길이
35. 문자열 indexing
36. 문자열 slicing
37. 문자열 불변객체
38. 문자열 연산
39. 문자열 타입변환
40. 문자열 split()
41. 문자열 join()
42. 문자열 lower(), upper()
43. 문자열 startswith(), endswith()
44. 문자열 바꾸기
45. 문자열 여백 제거
46. 사칙연산
47. 연산 %, //, **
48. 비교연산자
49. 조건문 if, elif, else
50. 조건문 삼항연산자
51. 논리연산자 - AND
52. 논리연산자 - OR
53. 논리연산자 - NOT
54. for ~ in 구문
55. range()
56. 제어문 - continue, break
57. while문
58. list, set, dict //2(일)
59. 함수(function)
60. 인수, 매개변수
61. 가변매개변수 - tuple
62. 가변매개변수 - dict
63. lambda 함수
64. 내장함수 map()
65. 내장함수 filter()
66. 내장함수 zip()
67. 내장함수 enumerate()
68. 패키지, 모듈, 라이브러리
69. 파이썬 Decorator! 알면 인생이 편해지는 기능 //3(월)

## 강의 요약


데이터 타입 (Data Type)

자료형태(자료형), 자료구조라고도 불리웁니다.

---

**주요 데이터 타입(type)**

1. `int` (정수): Integer(정수)의 약어이며, 정수를 나타내는 자료형입니다.
2. `float` (실수): Floating point의 약어이며, 소수점이 있는 숫자를 나타내는 자료형입니다.
3. `str` (문자열): 문자를 나타내는 자료형입니다. 작은 따옴표 혹은 큰 따옴표로 감싸져 있습니다.
4. `bool` (참/거짓): 참 또는 거짓을 나타내는 자료형입니다. True, False에서 T와 F는 반드시 대문자로 표기해야 합니다.

![image.png|92](attachment:fcac0ffe-4344-4cfc-8b45-566080b0b27e:image.png)

## 아무것도 아닌 None 타입

---

말 그래도 아무 것도 아닌 흔히 Null 값을 넣는다고도 합니다.

사전상 의미는

- **Null: Nullify (무효화하다)** 라는 뜻을 가지고 있다네요~

python에서는 **None** 입니다!

**시퀀스, 집합형 자료구조**

|분류|타입|특징|예시|
|---|---|---|---|
|시퀀스(sequence)|리스트(list)|순서가 있고, 가변(mutable)|[1, 2, 3]|
|시퀀스(sequence)|튜플(tuple)|순서가 있고, 불변(immutable)|(1, 2, 3)|
|세트(set)|세트(set)|순서가 없고, 중복을 허용하지 않음|{1, 2, 3}|
|맵(map)|딕셔너리(dictionary)|순서가 없고, key/value 쌍으로 이루어짐|{'a': 1, 'b': 2, 'c': 3}|
|||||

## 리스트(list)

- 리스트는 데이터의 요소를 순차적으로 파악하는데 용이한 자료형 입니다.
- 리스트는 다양한 메서드(method) 혹은 함수를 지원하며 메서드를 활용하여 요소를 추가, 삭제 및 변경할 수 있습니다.

**메서드(method)**: 객체(object)가 포함하는 함수 혹은 기능입니다. 함수에 대한 내용은 추후에 다룹니다.

### 슬라이싱(Slicing): 범위 추출

- 슬라이싱의 활용: [start:stop:step] 을 명시하여 부분을 추출할 수 있습니다.

## indexing 에 step 활용하기

list[start:stop:step]

- step은 몇 칸씩 건너 뛰는지 지정합니다.