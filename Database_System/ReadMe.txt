* 목표: Bitmap Index를 지원하는 관계 DB 시스템의 설계 및 구현
* 주제: 온라인 티켓 예매처
* 사용언어 및 프로그램: Java, MySQL
* 조건
1. 테이블의 레코드 수는 대량
2. 기능은 테이블 생성, 레코드 삽입, bitmap index 생성, bitmap index를 이용한 질의 처리(multiple-key 질의 처리, 집계함수 count 처리)
3. DB 시스템의 API는 위의 기능을 지원하는 호출
4. 응용 UI는 개발한 DB 시스템 상에서 동작하는 프로그램으로 2번의 기능 테스트용으로 text기반
5. 기능 동작의 정확성 검증
6. bitmap index는 디스크 기반의 저장
