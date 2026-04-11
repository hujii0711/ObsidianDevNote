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