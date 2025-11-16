import pandas as pd
import joblib
import logging
import sys
import traceback
import os
from recommender import FestivalRecommender, ensure_event_schema, DEFAULT_EVENT_COLUMNS


logger = logging.getLogger('FestivalRecommender')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)
file_handler = logging.FileHandler('model_train.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)
EVENT_CSV_FILE = os.environ.get('EVENT_CSV_FILE', 'event.csv')
MODEL_FILE = os.environ.get('MODEL_FILE', 'festival_recommender.pkl')

def load_festival_data(file_path):
    logger.debug(f"축제 데이터 로드: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.info(f"CSV 파일 없음, 새로 생성: {file_path}")
            df = pd.DataFrame(columns=DEFAULT_EVENT_COLUMNS)
            df.to_csv(file_path, index=False, encoding='utf-8')
            return df
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
        logger.debug(f"데이터프레임 크기: {df.shape}")
        return ensure_event_schema(df)
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(file_path, model_file):
    logger.info(f"모델 학습 시작: {file_path}")
    try:
        festival_data = load_festival_data(file_path)
        recommender = FestivalRecommender()
        recommender.fit(festival_data)
        joblib.dump(recommender, model_file)
        logger.info(f"모델이 {model_file}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 생성 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    train_model(EVENT_CSV_FILE, MODEL_FILE)
