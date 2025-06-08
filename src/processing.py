import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os


class ReviewProcessor:
    def __init__(self):
        self.stop_words = [
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но',
            'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня',
            'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни',
            'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя',
            'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
            'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто',
            'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
            'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два',
            'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
            'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
            'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        ]

    def load_data(self, filepath='data/raw/reviews.csv'):
        """Загружает данные из CSV файла"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"Загружено {len(df)} отзывов")
            return df
        except FileNotFoundError:
            print(f"Файл {filepath} не найден")
            return None

    def clean_text(self, text):
        """Очищает текст от лишних символов"""
        if pd.isna(text):
            return ""

        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', str(text))

        # Убираем специальные символы, оставляем только буквы, цифры и основные знаки
        text = re.sub(r'[^\w\s!?.,]', '', text)

        # Приводим к нижнему регистру
        text = text.lower().strip()

        return text

    def get_sentiment_score(self, text):
        """
        Простой анализ тональности на основе ключевых слов
        Возвращает число от -1 (негативно) до 1 (позитивно)
        """
        if not text:
            return 0

        # Позитивные слова
        positive_words = [
            'отличный', 'хороший', 'прекрасный', 'замечательный', 'великолепный',
            'качественный', 'рекомендую', 'доволен', 'довольна', 'нравится',
            'превосходный', 'идеальный', 'быстро', 'быстрый', 'быстрая'
        ]

        # Негативные слова
        negative_words = [
            'плохой', 'ужасный', 'плохо', 'не рекомендую', 'разочарование',
            'дефект', 'брак', 'медленно', 'долго', 'не работает', 'не стоит',
            'зря', 'хуже', 'проблема', 'недочет'
        ]

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Простая формула для расчета тональности
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0

        sentiment = (positive_count - negative_count) / max(total_words, 1)

        # Нормализуем в диапазон [-1, 1]
        return max(-1, min(1, sentiment * 10))

    def categorize_sentiment(self, sentiment_score):
        """Категоризует тональность на основе числового значения"""
        if sentiment_score > 0.1:
            return 'Позитивная'
        elif sentiment_score < -0.1:
            return 'Негативная'
        else:
            return 'Нейтральная'

    def extract_keywords(self, texts, max_features=20):
        """Извлекает ключевые слова из текстов"""
        # Очищаем тексты
        clean_texts = [self.clean_text(text) for text in texts if text]

        # Убираем стоп-слова
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Получаем средние значения TF-IDF для каждого слова
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Создаем словарь слово-важность
            keywords = dict(zip(feature_names, mean_scores))

            # Сортируем по важности
            keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

            return keywords

        except Exception as e:
            print(f"Ошибка при извлечении ключевых слов: {e}")
            return {}

    def process_reviews(self, df):
        """Основной метод обработки отзывов"""
        if df is None:
            return None

        print("Обработка отзывов...")

        # Создаем копию датафрейма
        processed_df = df.copy()

        # Очищаем тексты
        processed_df['clean_text'] = processed_df['text'].apply(self.clean_text)

        # Анализируем тональность
        processed_df['sentiment_score'] = processed_df['clean_text'].apply(self.get_sentiment_score)
        processed_df['sentiment_category'] = processed_df['sentiment_score'].apply(self.categorize_sentiment)

        # Добавляем длину отзыва
        processed_df['text_length'] = processed_df['text'].apply(lambda x: len(str(x)) if x else 0)
        processed_df['word_count'] = processed_df['text'].apply(lambda x: len(str(x).split()) if x else 0)

        # Конвертируем дату
        processed_df['date'] = pd.to_datetime(processed_df['date'])

        # Создаем категории по рейтингу
        def rating_category(rating):
            if rating >= 4:
                return 'Высокий'
            elif rating >= 3:
                return 'Средний'
            else:
                return 'Низкий'

        processed_df['rating_category'] = processed_df['rating'].apply(rating_category)

        print("Обработка завершена!")

        return processed_df

    def save_processed_data(self, df, filename='processed_reviews.csv'):
        """Сохраняет обработанные данные"""
        if df is None:
            return

        # Создаем директорию если её нет
        os.makedirs('data/processed', exist_ok=True)

        filepath = f'data/processed/{filename}'
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Обработанные данные сохранены в {filepath}")

    def get_summary_stats(self, df):
        """Получает основную статистику по данным"""
        if df is None:
            return {}

        stats = {
            'total_reviews': len(df),
            'avg_rating': df['rating'].mean(),
            'sentiment_distribution': df['sentiment_category'].value_counts().to_dict(),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean()
        }

        return stats


if __name__ == "__main__":
    processor = ReviewProcessor()

    # Загружаем данные
    df = processor.load_data()

    if df is not None:
        # Обрабатываем
        processed_df = processor.process_reviews(df)

        # Сохраняем
        processor.save_processed_data(processed_df)

        # Выводим статистику
        stats = processor.get_summary_stats(processed_df)
        print("\n=== СТАТИСТИКА ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\nПервые 3 обработанных отзыва:")
        print(processed_df[['text', 'rating', 'sentiment_category', 'sentiment_score']].head(3))