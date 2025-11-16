# FestM

# 🎉 AI 축제 추천 서비스
> **취향 기반 AI 추천 + 실시간 이벤트 동기화**
>
> FastAPI + Python으로 구현된 TF-IDF 기반 축제 추천/관리 서비스입니다.

---

## 🧩 프로젝트 개요
사용자의 검색·즐겨찾기 이력을 학습 데이터와 결합해 맞춤형 축제를 추천하고, 매일 자정 배치(`batch`)로 새로운 이벤트를 주입하면 즉시 CSV와 모델을 갱신하도록 설계되었습니다.

### 주요 기능
- 🎯 **AI 추천 엔진**: 사용자 히스토리를 TF-IDF 벡터로 변환해 가장 유사한 축제를 추천합니다.
- 🔄 **이벤트 동기화 + 자동 학습**: `/event-sync` API로 들어온 이벤트를 CSV에 병합하고 즉시 모델을 재학습합니다.
- 🕛 **스케줄 기반 클린업**: APScheduler가 매일 지정 시간에 CSV를 초기화하고 모델을 리프레시합니다.
- 🩺 **헬스 체크 & 로깅**: `/health` 엔드포인트와 다중 핸들러 로깅으로 운영 상태를 추적합니다.

---

## 🏗️ 시스템 구성
1. **이벤트 수집** → `event.csv`에 스키마를 맞춰 저장.
2. **모델 학습** → `FestivalRecommender`가 TF-IDF 행렬을 생성하고 `festival_recommender.pkl`로 직렬화.
3. **API 제공** → FastAPI가 `/recommend`, `/event-sync`, `/health` 등을 통해 데이터를 주고받음.
4. **운영 자동화** → APScheduler가 주기적으로 CSV 초기화 및 모델 재학습 작업 수행.

---

## ⚙️ 기술 스택
- **백엔드**: FastAPI, Pydantic
- **ML/데이터**: scikit-learn, pandas, numpy, joblib
- **스케줄링**: APScheduler
- **런타임**: Python 3.x, Uvicorn
