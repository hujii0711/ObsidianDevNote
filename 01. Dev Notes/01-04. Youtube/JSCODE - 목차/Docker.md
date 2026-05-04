**1.1. Docker를 왜 배우는 걸까?**

**1.2. Docker란 ? / 컨테이너(Container)란? / 이미지(Image)란?**

**1.3. [실습] Docker 전체 흐름 느껴보기 (Nginx 설치 및 실행)**

**2.1. 이미지(Image) 다운로드**

**2.2. 이미지(Image) 조회 / 삭제**

**2.3. 컨테이너(Container) 생성 / 실행 - 1**

**2.4. 컨테이너(Container) 생성 / 실행 - 2**

**2.5. 컨테이너(Container) 조회 / 중지 / 삭제**

**2.6. 컨테이너(Container) 로그 조회**

**2.7. 실행중인 컨테이너 내부에 접속하기 (exec -it)**

**2.8. [실습] Docker 전체 흐름 다시 느껴보기 (Nginx 설치 및 실행)**

**2.9. [실습] Docker로 Redis 실행시켜보기**

**3.1. Docker Volume(도커 볼륨)**

**3.2. [실습] Docker로 MySQL 실행시켜보기 - 1**

**3.3. [실습] Docker로 MySQL 실행시켜보기 - 2**

**3.4. [실습] Docker로 MySQL 실행시켜보기 - 3**

==============================================================

1. 도커 볼륨을 활용해 데이터 유실 방지하기

[보충 자료] 윈도우(Windows)에서 볼륨이 생성되지 않는 경우

[실습] Docker로 MySQL 실행시켜보기 - 4

[보충 자료] Docker로 PostgreSQL 실행시켜보기

[실습] Docker로 PostgreSQL 실행시켜보기

[실습] Docker로 MongoDB 실행시켜보기

1. Dockerfile 활용해 이미지 직접 만들기

Dockerfile이란?

FROM : 베이스 이미지 생성

[실습] FROM : 베이스 이미지 생성

종료된 컨테이너에 들어가서 디버깅하고 싶을 때

COPY : 파일 복사(이동)

ENTRYPOINT : 컨테이너가 시작할 때 실행되는 명령어

[실습] 백엔드 프로젝트(Spring Boot) 프로젝트를 Docker로 실행시키기

RUN : 이미지를 생성하는 과정에서 사용할 명령문 실행

WORKDIR : 작업 디렉토리를 지정

EXPOSE : 컨테이너 내부에서 사용 중인 포트를 문서화하기

[실습] 백엔드 프로젝트(Nest.js)를 Docker로 실행시키기

[실습] 웹 프론트엔드 프로젝트(Next.js)를 Docker로 배포하기

[실습] 웹 프론트엔드 프로젝트(HTML, CSS, Nginx)를 Docker로 배포하기

1. Docker Compose를 활용해 컨테이너 관리하기

Docker Compose를 사용하는 이유(강의 있음)

Docker Compose 전체 흐름 느껴보기(강의 있음)

자주 사용하는 Docker Compose CLI 명령어

[실습] Docker Compose로 Redis 실행시키기

[실습] Docker Compose로 MySQL 실행시키기

[실습] Docker Compose로 백엔드(Spring Boot) 실행시키기

[실습] Docker Compose로 백엔드(Nest.js) 실행시키기

[실습] Docker Compose로 프론트엔드(Next.js) 실행시키기

[실습] Docker Compose로 프론트엔드(HTML, CSS, Nginx) 실행시키기

Docker CLI ↔ Docker Compose 쉽게 작성하기

1. Docker Compose를 활용해 2개 이상의 컨테이너 관리하기

[실습] MySQL, Redis 컨테이너 동시에 띄워보기

[실습] Spring Boot, MySQL 컨테이너 동시에 띄워보기

컨테이너로 실행시킨 Spring Boot가 MySQL에 연결이 안 되는 이유

[실습] Spring Boot, MySQL, Redis 컨테이너 동시에 띄워보기