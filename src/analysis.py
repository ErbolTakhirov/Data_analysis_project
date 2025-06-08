import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Настройка русского языка для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'


class ReviewAnalyzer:
    def __init__(self):
        self.df = None

    def load_processed_data(self, filepath='data/processed/processed_reviews.csv'):
        """Загружает обработанные данные"""
        try:
            self.df = pd.read_csv(filepath, encoding='utf-8')
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Загружено {len(self.df)} обработанных отзывов")
            return True
        except FileNotFoundError:
            print(f"Файл {filepath} не найден")
            return False

    def basic_statistics(self):
        """Выводит базовую статистику"""
        if self.df is None:
            print("Данные не загружены")
            return

        print("=== БАЗОВАЯ СТАТИСТИКА ===")
        print(f"Общее количество отзывов: {len(self.df)}")
        print(f"Средний рейтинг: {self.df['rating'].mean():.2f}")
        print(f"Медианный рейтинг: {self.df['rating'].median()}")
        print(f"Средняя длина отзыва: {self.df['word_count'].mean():.1f} слов")

        print("\nРаспределение по рейтингам:")
        print(self.df['rating'].value_counts().sort_index())

        print("\nРаспределение по тональности:")
        print(self.df['sentiment_category'].value_counts())

        print("\nСтатистика по тональности:")
        print(f"Средний score тональности: {self.df['sentiment_score'].mean():.3f}")
        print(f"Стандартное отклонение: {self.df['sentiment_score'].std():.3f}")

    def correlation_analysis(self):
        """Анализ корреляций между рейтингом и тональностью"""
        if self.df is None:
            return

        correlation = self.df['rating'].corr(self.df['sentiment_score'])
        print(f"\nКорреляция между рейтингом и тональностью: {correlation:.3f}")

        # Группировка по рейтингам
        rating_sentiment = self.df.groupby('rating')['sentiment_score'].agg(['mean', 'std', 'count'])
        print("\nСредняя тональность по рейтингам:")
        print(rating_sentiment)

    def create_rating_distribution_plot(self):
        """Создает график распределения рейтингов"""
        if self.df is None:
            return

        plt.figure(figsize=(10, 6))

        # Основной график
        plt.subplot(1, 2, 1)
        rating_counts = self.df['rating'].value_counts().sort_index()
        plt.bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7)
        plt.title('Распределение рейтингов')
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество отзывов')
        plt.xticks(range(1, 6))

        # Круговая диаграмма
        plt.subplot(1, 2, 2)
        sentiment_counts = self.df['sentiment_category'].value_counts()
        colors = ['lightcoral', 'lightgray', 'lightgreen']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Распределение тональности')

        plt.tight_layout()
        plt.savefig('data/external/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_sentiment_analysis_plot(self):
        """Создает график анализа тональности"""
        if self.df is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # График 1: Scatter plot рейтинг vs тональность
        axes[0, 0].scatter(self.df['rating'], self.df['sentiment_score'], alpha=0.6, color='steelblue')
        axes[0, 0].set_xlabel('Рейтинг')
        axes[0, 0].set_ylabel('Score тональности')
        axes[0, 0].set_title('Рейтинг vs Тональность')
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Boxplot тональности по рейтингам
        self.df.boxplot(column='sentiment_score', by='rating', ax=axes[0, 1])
        axes[0, 1].set_title('Тональность по рейтингам')
        axes[0, 1].set_xlabel('Рейтинг')
        axes[0, 1].set_ylabel('Score тональности')

        # График 3: Гистограмма тональности
        axes[1, 0].hist(self.df['sentiment_score'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Score тональности')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Распределение тональности')
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Средняя тональность по рейтингам
        avg_sentiment = self.df.groupby('rating')['sentiment_score'].mean()
        axes[1, 1].bar(avg_sentiment.index, avg_sentiment.values, color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Рейтинг')
        axes[1, 1].set_ylabel('Средняя тональность')
        axes[1, 1].set_title('Средняя тональность по рейтингам')
        axes[1, 1].set_xticks(range(1, 6))

        plt.tight_layout()
        plt.savefig('data/external/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_word_cloud(self):
        """Создает облако слов"""
        if self.df is None:
            return

        # Объединяем все тексты
        all_text = ' '.join(self.df['clean_text'].dropna())

        if not all_text.strip():
            print("Нет текста для создания облака слов")
            return

        # Создаем облако слов
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate(all_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Облако слов из отзывов', fontsize=16)
        plt.tight_layout(pad=0)
        plt.savefig('data/external/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_time_series_plot(self):
        """Создает график динамики по времени"""
        if self.df is None:
            return

        # Группируем данные по месяцам
        monthly_data = self.df.set_index('date').resample('M').agg({
            'rating': 'mean',
            'sentiment_score': 'mean',
            'text': 'count'
        }).rename(columns={'text': 'review_count'})

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # График количества отзывов
        axes[0].plot(monthly_data.index, monthly_data['review_count'], marker='o', color='blue')
        axes[0].set_title('Количество отзывов по времени')
        axes[0].set_ylabel('Количество')
        axes[0].grid(True, alpha=0.3)

        # График среднего рейтинга
        axes[1].plot(monthly_data.index, monthly_data['rating'], marker='s', color='green')
        axes[1].set_title('Средний рейтинг по времени')
        axes[1].set_ylabel('Средний рейтинг')
        axes[1].grid(True, alpha=0.3)

        # График средней тональности
        axes[2].plot(monthly_data.index, monthly_data['sentiment_score'], marker='^', color='red')
        axes[2].set_title('Средняя тональность по времени')
        axes[2].set_ylabel('Средняя тональность')
        axes[2].set_xlabel('Дата')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/external/time_series.png', dpi=300, bbox_inches='tight')
        plt.show()

    def find_extreme_reviews(self):
        """Находит самые позитивные и негативные отзывы"""
        if self.df is None:
            return

        print("=== ЭКСТРЕМАЛЬНЫЕ ОТЗЫВЫ ===")

        # Самые позитивные отзывы
        print("\nСамые позитивные отзывы:")
        positive_reviews = self.df.nlargest(3, 'sentiment_score')[['text', 'rating', 'sentiment_score']]
        for i, (_, row) in enumerate(positive_reviews.iterrows(), 1):
            print(f"{i}. Рейтинг: {row['rating']}, Тональность: {row['sentiment_score']:.3f}")
            print(f"   Текст: {row['text'][:100]}...")
            print()

        # Самые негативные отзывы
        print("Самые негативные отзывы:")
        negative_reviews = self.df.nsmallest(3, 'sentiment_score')[['text', 'rating', 'sentiment_score']]
        for i, (_, row) in enumerate(negative_reviews.iterrows(), 1):
            print(f"{i}. Рейтинг: {row['rating']}, Тональность: {row['sentiment_score']:.3f}")
            print(f"   Текст: {row['text'][:100]}...")
            print()

    def analyze_rating_sentiment_mismatch(self):
        """Анализирует несоответствия между рейтингом и тональностью"""
        if self.df is None:
            return

        print("=== АНАЛИЗ НЕСООТВЕТСТВИЙ ===")

        # Высокий рейтинг, но негативная тональность
        high_rating_negative = self.df[(self.df['rating'] >= 4) & (self.df['sentiment_score'] < -0.1)]
        if len(high_rating_negative) > 0:
            print(f"\nВысокий рейтинг, но негативная тональность ({len(high_rating_negative)} отзывов):")
            for _, row in high_rating_negative.head(2).iterrows():
                print(f"Рейтинг: {row['rating']}, Тональность: {row['sentiment_score']:.3f}")
                print(f"Текст: {row['text']}")
                print()

        # Низкий рейтинг, но позитивная тональность
        low_rating_positive = self.df[(self.df['rating'] <= 2) & (self.df['sentiment_score'] > 0.1)]
        if len(low_rating_positive) > 0:
            print(f"Низкий рейтинг, но позитивная тональность ({len(low_rating_positive)} отзывов):")
            for _, row in low_rating_positive.head(2).iterrows():
                print(f"Рейтинг: {row['rating']}, Тональность: {row['sentiment_score']:.3f}")
                print(f"Текст: {row['text']}")
                print()

    def create_interactive_dashboard(self):
        """Создает интерактивный дашборд с Plotly"""
        if self.df is None:
            return

        # Создаем подграфики
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Распределение рейтингов', 'Тональность vs Рейтинг',
                            'Распределение тональности', 'Длина отзывов'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'box'}]]
        )

        # График 1: Распределение рейтингов
        rating_counts = self.df['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=rating_counts.index, y=rating_counts.values, name='Рейтинги'),
            row=1, col=1
        )

        # График 2: Scatter plot
        fig.add_trace(
            go.Scatter(
                x=self.df['rating'],
                y=self.df['sentiment_score'],
                mode='markers',
                name='Тональность vs Рейтинг',
                text=self.df['text'].str[:100],
                hovertemplate='Рейтинг: %{x}<br>Тональность: %{y:.3f}<br>%{text}...'
            ),
            row=1, col=2
        )

        # График 3: Гистограмма тональности
        fig.add_trace(
            go.Histogram(x=self.df['sentiment_score'], name='Тональность'),
            row=2, col=1
        )

        # График 4: Boxplot длины отзывов по рейтингам
        for rating in sorted(self.df['rating'].unique()):
            fig.add_trace(
                go.Box(
                    y=self.df[self.df['rating'] == rating]['word_count'],
                    name=f'Рейтинг {rating}',
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(height=800, title_text="Дашборд анализа отзывов")
        fig.write_html('data/external/dashboard.html')
        print("Интерактивный дашборд сохранен в data/external/dashboard.html")

        return fig

    def generate_insights(self):
        """Генерирует основные инсайты из анализа"""
        if self.df is None:
            return

        print("=== ОСНОВНЫЕ ИНСАЙТЫ ===")

        # Общая статистика
        total_reviews = len(self.df)
        avg_rating = self.df['rating'].mean()
        positive_ratio = len(self.df[self.df['sentiment_category'] == 'Позитивная']) / total_reviews * 100

        print(f"1. Общий анализ:")
        print(f"   - Проанализировано {total_reviews} отзывов")
        print(f"   - Средний рейтинг: {avg_rating:.2f} из 5")
        print(f"   - Позитивных отзывов: {positive_ratio:.1f}%")

        # Корреляция
        correlation = self.df['rating'].corr(self.df['sentiment_score'])
        print(f"\n2. Соответствие рейтинга и тональности:")
        print(f"   - Корреляция: {correlation:.3f}")
        if correlation > 0.5:
            print("   - Высокое соответствие между рейтингом и тональностью")
        elif correlation > 0.3:
            print("   - Умеренное соответствие между рейтингом и тональностью")
        else:
            print("   - Слабое соответствие между рейтингом и тональностью")

        # Распределение по рейтингам
        rating_dist = self.df['rating'].value_counts().sort_index()
        most_common_rating = rating_dist.idxmax()
        print(f"\n3. Распределение рейтингов:")
        print(f"   - Наиболее частый рейтинг: {most_common_rating}")
        print(f"   - Высокие рейтинги (4-5): {(rating_dist[4] + rating_dist[5]) / total_reviews * 100:.1f}%")
        print(f"   - Низкие рейтинги (1-2): {(rating_dist[1] + rating_dist[2]) / total_reviews * 100:.1f}%")

        # Длина отзывов
        avg_length = self.df['word_count'].mean()
        print(f"\n4. Характеристики отзывов:")
        print(f"   - Средняя длина: {avg_length:.1f} слов")

        # Топ проблемы (если есть негативные отзывы)
        negative_reviews = self.df[self.df['sentiment_category'] == 'Негативная']
        if len(negative_reviews) > 0:
            print(f"\n5. Негативные отзывы:")
            print(f"   - Количество: {len(negative_reviews)} ({len(negative_reviews) / total_reviews * 100:.1f}%)")
            print("   - Основные проблемы можно выявить из анализа негативных отзывов")

    def run_full_analysis(self):
        """Запускает полный анализ"""
        print("Запуск полного анализа отзывов...")

        if not self.load_processed_data():
            print("Не удалось загрузить данные")
            return

        # Создаем директорию для результатов
        import os
        os.makedirs('data/external', exist_ok=True)

        # Базовая статистика
        self.basic_statistics()

        # Корреляционный анализ
        self.correlation_analysis()

        # Графики
        print("\nСоздание графиков...")
        self.create_rating_distribution_plot()
        self.create_sentiment_analysis_plot()
        self.create_word_cloud()
        self.create_time_series_plot()

        # Экстремальные отзывы
        self.find_extreme_reviews()

        # Анализ несоответствий
        self.analyze_rating_sentiment_mismatch()

        # Интерактивный дашборд
        self.create_interactive_dashboard()

        # Инсайты
        self.generate_insights()

        print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")
        print("Результаты сохранены в папке data/external/")


if __name__ == "__main__":
    analyzer = ReviewAnalyzer()
    analyzer.run_full_analysis()