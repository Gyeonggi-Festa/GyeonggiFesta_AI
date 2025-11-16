import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
import sys
import os


DEFAULT_EVENT_COLUMNS = ['eventid', 'title', 'category', 'isFree', 'location', 'description']
TEXT_FEATURE_COLUMNS = ['title', 'category', 'location', 'description', 'isFree']
COLUMN_ALIASES = {
    'eventid': 'eventid',
    'event_id': 'eventid',
    'title': 'title',
    'category': 'category',
    'isfree': 'isFree',
    'location': 'location',
    'description': 'description'
}
IS_FREE_POSITIVE = {'Y', 'YES', 'FREE', 'TRUE', '1'}
IS_FREE_NEGATIVE = {'N', 'NO', 'FALSE', 'PAID', '0'}


def ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming event data so the rest of the pipeline can assume the
    same column names even when optional fields (location/description) are
    missing.
    """
    if df is None:
        return pd.DataFrame(columns=DEFAULT_EVENT_COLUMNS)

    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    rename_map = {}
    for col in normalized.columns:
        alias = COLUMN_ALIASES.get(col.lower())
        if alias:
            rename_map[col] = alias
    normalized = normalized.rename(columns=rename_map)

    if 'eventid' not in normalized.columns:
        raise ValueError("event data must include an 'eventid' column")

    normalized['eventid'] = normalized['eventid'].astype(str)
    for column in ['title', 'category']:
        if column not in normalized.columns:
            normalized[column] = ''
    for column in DEFAULT_EVENT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ''

    ordered_columns = DEFAULT_EVENT_COLUMNS + [c for c in normalized.columns if c not in DEFAULT_EVENT_COLUMNS]
    normalized = normalized[ordered_columns]
    normalized = normalized.fillna('')
    return normalized


def _format_is_free(value):
    if value is None:
        return ''
    token = str(value).strip().upper()
    if not token:
        return ''
    if token in IS_FREE_POSITIVE:
        return '무료'
    if token in IS_FREE_NEGATIVE:
        return '유료'
    return str(value)

logger = logging.getLogger('FestivalRecommender')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

class FestivalRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model_file = os.environ.get('MODEL_FILE', 'festival_recommender.pkl')
        self.csv_file = os.environ.get('EVENT_CSV_FILE', 'event.csv')
        self.festival_data = None
        self.tfidf_matrix = None
        logger.debug("FestivalRecommender 초기화")

    def fit(self, festival_data):
        logger.debug("모델 학습 시작")
        try:
            normalized_df = ensure_event_schema(festival_data)
            if normalized_df.empty:
                logger.warning("No festival data available, skipping training")
                self.festival_data = normalized_df
                self.tfidf_matrix = None
                return
            self.festival_data = normalized_df
            text_df = normalized_df.copy()
            if 'isFree' in text_df.columns:
                text_df['isFree'] = text_df['isFree'].apply(_format_is_free)
            text_columns = [col for col in TEXT_FEATURE_COLUMNS if col in text_df.columns]
            combined_features = text_df[text_columns].fillna('').agg(' '.join, axis=1)
            logger.debug("combined_features 생성 완료")
            self.tfidf_matrix = self.vectorizer.fit_transform(combined_features)
            logger.debug("TF-IDF 벡터화 완료")
        except Exception as e:
            logger.error(f"모델 학습 실패: {str(e)}")
            raise

    def add_events(self, events):
        logger.debug("CSV에 이벤트 추가 시작")
        try:
            csv_file = self.csv_file
            if not os.path.exists(csv_file):
                logger.info(f"CSV file missing, creating: {csv_file}")
                pd.DataFrame(columns=DEFAULT_EVENT_COLUMNS).to_csv(csv_file, index=False, encoding='utf-8')
            existing_df = pd.read_csv(csv_file, encoding='utf-8')
            existing_df = ensure_event_schema(existing_df)
            new_df = ensure_event_schema(pd.DataFrame(events))
            if not existing_df.empty:
                existing_df = existing_df[~existing_df['eventid'].isin(new_df['eventid'])]
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df = updated_df.fillna('')
            updated_df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.debug(f"CSV에 {len(events)}개 이벤트 추가 완료, 총 행: {len(updated_df)}")
            self.fit(updated_df)
            joblib.dump(self, self.model_file)
            return len(events)
        except Exception as e:
            logger.error(f"이벤트 추가 실패: {str(e)}")
            raise

    def recommend(self, user_data, top_n=5):
        logger.debug(f"사용자 데이터: {user_data}")
        try:
            if self.tfidf_matrix is None or self.festival_data.empty:
                logger.warning("모델 또는 데이터 없음, 추천 불가")
                return [{"userid": user['userid'], "festivalRecommendations": [{"eventid": []}]} for user in user_data]
            recommendations = []
            for user in user_data:
                user_id = user.get('userid', 'unknown')
                search_history = ' '.join(user.get('searchHistory', []))
                favorites = ' '.join(user.get('favorites', []))
                user_features = f"{search_history} {favorites}".strip()
                if not user_features:
                    recommendations.append({"userid": user_id, "festivalRecommendations": [{"eventid": None}]})
                    continue
                user_tfidf = self.vectorizer.transform([user_features])
                similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
                top_indices = similarities.argsort()[-top_n:][::-1]
                top_event_ids = self.festival_data.iloc[top_indices]['eventid'].astype(str).tolist()
                recommendations.append({
                    "userid": user_id,
                    "festivalRecommendations": [{"eventid": top_event_ids}]
                })
            logger.debug(f"추천 결과: {recommendations}")
            return recommendations
        except Exception as e:
            logger.error(f"추천 실패: {str(e)}")
            raise
