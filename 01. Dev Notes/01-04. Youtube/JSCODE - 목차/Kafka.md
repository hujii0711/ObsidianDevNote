1-1. Kafka를 왜 배워야 할까?

1-2. Kafka의 메시지큐란?

1-3. Kafka 설치할 환경 세팅하기

1-4. AWS EC2에 Kafka 설치하기

2-1. Kafka의 기본 구성(Topic, Producer, Consumer)

2-2. Topic 생성, 조회, 삭제하기

2-3. Kafka에 메시지 넣기, 조회하기

2-4. 메시지를 어디까지 읽었는지 기억하고, 그 다음 메시지부터 처리하기

2-5. Spring boot에 Kafka 연결을 위한 코드 추가하기(Producer)

2-6. Spring boot로 Kafka에 메시지 넣는 코드 작성하기

2-7. Spring boot가 Kafka에 메시지 잘 넣었는지 테스트하기

2-8. Spring boot로 Kafka에서 메시지 조회하기(Consumer)

2-9. Kafka의 비동기 처리로 인한 성능 이점 느껴보기

3-1. Spring boot로 Kafka에서 처리에 실패한 메시지를 재시도 하도록 만들기

3-2. Spring boot로 Kafka에서 재시도조차 실패한 메시지를 따로 보관하기

3-3. Spring boot로 재시도조차 실패한 메시지 사후 처리하기

==============================================================

1. Kafka 메시지 처리 성능 높이기(병렬 처리)

컨슈머가 메시지를 하나씩만 처리하는 현상

**파티션(Partition)이란? / 특징**

[실습] Spring Boot로 하나의 파티션에는 정말 하나의 컨슈머만 할당되는 지 확인해보기

특정 토픽의 파티션 수 조회하기 / 설정하기 / 변경하기

[실습] Spring Boot로 여러 개의 파티션에 메시지가 골고루 들어가는 지 확인해보기

[실습] Spring Boot에서 여러 개의 컨슈머로 메시지 병렬적으로 처리하기

[실습] Spring Boot에서 하나의 컨슈머로 메시지 병렬적으로 처리하기

적정 파티션 개수 계산하는 방법

컨슈머가 메시지를 지연 없이 잘 처리하고 있는 지 확인하는 방법 (Consumer Lag)

1. Kafka 장애 대비하기(고가용성)

노드(node), 브로커(broker), 컨트롤러(controller), 클러스터(cluster), 레플리케이션(replication)이란?

[실습] kafka 서버 총 3대 셋팅하기

[실습] Kafka 서버 3대가 서로 잘 연동됐는 지 확인하기

토픽 세부 정보 출력값 정보 해석하기 (Isr, Leader, Replicas 등)

[실습] 팔로워 파티션에 메시지를 넣으면 어떻게 될까?

[실습] 리더 파티션에 장애가 발생하면 어떻게 될까? / Kafka 서버 1대가 고장나면 어떻게 될까?

Kafka 서버는 몇 대를 운용하는 게 좋을까?

Spring Boot에 Kafka 서버 3대를 연결해서 사용하는 방법

1. MSA 프로젝트에서 Kafka 도입하기

프로젝트 설계

[실습] Spring Boot로 UserService 서버 초기 환경 설정하기

[실습] 회원가입 API 전체 뼈대 만들기

[실습] 회원 가입 비즈니스 로직 짜기

[실습] Spring Boot로 EmailService 서버 초기 환경 설정하기

[실습] 이메일 발송을 처리할 Consumer 로직 짜기

[실습] 프로젝트 구조에 맞게 Kafka 셋팅하기

[실습] 잘 작동하는 지 테스트해보기

AWS 비용 나가지 않게 리소스 종료하기 / 혹시나 비용 나가고 있는 지 체크하기