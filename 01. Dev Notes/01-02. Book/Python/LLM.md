- 아나콘다 실행 환경 구성

	가상 환경 생성: conda create -n 내환경이름 python=3.10
	
	새 환경을 만들 때 아나콘다의 기본 패키지들을 모두 포함: conda create -n 내환경이름 python=3.10 anaconda
	
	가상 환경 환경 활성화: conda activate 내환경이름
	
	현재 사용중인 가상 환경 비활성화: conda deactivate
	
	가상 환경 삭제: conda remove -n 내환경이름 --all
	
	환경 목록 확인: conda env list
	
	아나콘다 설치된 패키지 목록 확인: conda list
	
	특정 패키지 설치 여부 확인: conda list numpy
	
	파이토치 설치: conda install pytorch torchvision torchaudio -c pytorch
	
	tiktoken 설치: conda install -c conda-forge tiktoken

- 파이썬 가상 환경 구성
	1. python3 -m venv myenv: 가상 환경 생성
	2. source myenv/bin/activate: 가상 환경 활성화
	3. pip3 install xlwings: 가상 환경 내 패키지 설치
	4. pip3 freeze: 패키지 목록 출력
	5. pip3 freeze > requirement.txt: 설치 패키지 목록 파일로 생성
	6. pip3 install -r requirement.txt: 파일 기반으로 가상 환경에 필요한 패키지 설치
	7. python3 -m venv myenv --system-site-packages: 공용 패키지 함께 사용 가능
	8. pip3 list: 공용 패키지 설치 목록 확인
	9. pip3 list --local: 가상 환경에 설치된 패키지만 확인