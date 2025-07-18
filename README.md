# 배터리 데이터 이상치 탐지 

이 프로젝트는 배터리 사이클 데이터에서 이상치를 탐지하고 제거하는 머신러닝 기반 시스템입니다.

## 프로젝트 개요

- **목적**: 배터리 사이클 데이터의 이상치 탐지 및 제거
- **방법**: 다변량 오토인코더를 사용한 비지도 학습
- **데이터**: NASA 배터리 데이터셋 (B0005, B0006, B0007, B0018, B0033, B0034, B0036, B0038, B0039, B0040)

## 환경별 데이터 그룹

- **env1**: B0005, B0006, B0007, B0018
- **env2**: B0033, B0034, B0036  
- **env3**: B0038, B0039, B0040

## 주요 기능

1. **데이터 전처리**: 사이클별 데이터 정규화 및 패딩
2. **오토인코더 학습**: 다변량 시계열 데이터 재구성
3. **이상치 탐지**: 재구성 오차 기반 이상치 식별
4. **시각화**: 사이클별 비교 및 이상치 분포 시각화
5. **결과 저장**: 정제된 데이터 및 분석 결과 저장

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/` 폴더에 다음 CSV 파일들을 배치하세요:
- B0005.csv, B0006.csv, B0007.csv, B0018.csv
- B0033.csv, B0034.csv, B0036.csv
- B0038.csv, B0039.csv, B0040.csv

### 3. 실행
```bash
python main.py
```

## 출력 결과

실행 후 `outputs/` 폴더에 다음 결과물들이 생성됩니다:

### 환경별 폴더 구조
```
outputs/
├── env1/
│   ├── B0005/
│   │   └── B0005_cleaned_final.csv
│   ├── cyclewise_compare/
│   │   └── [시각화 파일들]
│   ├── outlier_scatter/
│   │   └── [이상치 산점도]
│   ├── recon_error_compare/
│   │   └── [재구성 오차 비교]
│   └── env1_outliers.csv
├── env2/
└── env3/
```

### 주요 출력 파일
- **`*_cleaned_final.csv`**: 이상치가 제거된 정제 데이터
- **`*_outliers.csv`**: 탐지된 이상치 목록
- **`*_outlier_scatter.png`**: 사이클별 이상치 분포 시각화
- **`*_cyclewise_overlapped.png`**: 사이클별 데이터 비교 시각화
- **`*_compare_recon_error.png`**: 이상치 제거 전후 재구성 오차 비교

## 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Matplotlib**: 시각화

## 매개변수 설정

`main.py`에서 다음 매개변수들을 조정할 수 있습니다:

```python
LEARNING_RATE = 0.001      # 학습률
BATCH_SIZE = 32           # 배치 크기
NUM_EPOCHS = 50           # 학습 에포크
QUANTILE_THRESHOLD = 0.95 # 이상치 탐지 임계값
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
