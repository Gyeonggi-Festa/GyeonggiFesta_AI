from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pandas as pd
import logging
import os
import traceback
from recommender import DEFAULT_EVENT_COLUMNS


logger = logging.getLogger('FestivalRecommender')
CSV_FILE = os.environ.get('EVENT_CSV_FILE', 'event.csv')
MODEL_FILE = os.environ.get('MODEL_FILE', 'festival_recommender.pkl')

def reset_csv():
    logger.debug("CSV 초기화 시작")
    try:
        df = pd.DataFrame(columns=DEFAULT_EVENT_COLUMNS)
        df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        logger.info(f"CSV 초기화 완료: {CSV_FILE}")
        # 모델 재학습
        from model_train import train_model
        train_model(CSV_FILE, MODEL_FILE)
    except Exception as e:
        logger.error(f"CSV 초기화 실패: {str(e)}")
        logger.error(traceback.format_exc())

def start_scheduler():
    logger.debug("스케줄러 시작 시도")
    try:
        scheduler = BackgroundScheduler(timezone="Asia/Seoul")
        # 운영용: 자정 00:00:05
        scheduler.add_job(
            reset_csv,
            trigger=CronTrigger(hour=22, minute=30, second=0, timezone="Asia/Seoul"),
            id='reset_csv_job',
            replace_existing=True
        )
        # 테스트용: 주석 처리 (10초 후 실행)
        # scheduler.add_job(
        #     reset_csv,
        #     trigger=IntervalTrigger(seconds=10),
        #     id='reset_csv_job',
        #     replace_existing=True
        # )
        scheduler.start()
        logger.info("스케줄러 시작됨: 자정 00:00:05 CSV 초기화")
        jobs = scheduler.get_jobs()
        logger.debug(f"등록된 작업: {jobs}")
    except Exception as e:
        logger.error(f"스케줄러 시작 실패: {str(e)}")
        logger.error(traceback.format_exc())
