import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from processing import ReviewProcessor

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ отзывов покупателей",
    page_icon="📊",
    layout="wide"
)


class ReviewDashboard:
    def __init__(self):
        self.df = None

    def load_data(self):
        """Загружает данные"""
        try:
            self.df = pd.read_csv('data/processed/processed_reviews.csv')
            self.df['date'] = pd.to_datetime(self.df['date'])
            return True
        except FileNotFoundError:
            return False

    def show_header(self):
        """Показывает заголовок приложения"""
        st.title("📊 Анализ отзывов покупателей")
        st.markdown("---")

        if self.df is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всего отзывов", len(self.df))

            with col2:
                avg_rating = self.df['rating'].mean()
                st.metric("Средний рейтинг", f"{avg_rating:.2f}")

            with col3:
                positive_pct = len(self.df[self.df['sentiment_category'] == 'Позитивная']) / len(self.df) * 100
                st.metric("Позитивных отзывов", f"{positive_pct:.1f}%")

            with col4:
                avg_words = self.df['word_count'].mean()
                st.metric("Средняя длина", f"{avg_words:.0f} слов")

    def show_rating_analysis(self):
        """Показывает анализ рейтингов"""
        st.header("📈 Анализ рейтингов")

        col1, col2 = st.columns(2)

        with col1:
            # Распределение рейтингов
            rating_counts = self.df['rating'].value_counts().sort_index()
            fig_bar = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Рейтинг', 'y': 'Количество отзывов'},
                title='Распределение рейтингов',
                color=rating_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Круговая диаграмма по категориям рейтинга
            rating_categories = self.df['rating_category'].value_counts()
            fig_pie = px.pie(
                values=rating_categories.values,
                names=rating_categories.index,
                title='Категории рейтингов'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    def show_sentiment_analysis(self):
        """Показывает анализ тональности"""
        st.header("😊 Анализ тональности")

        col1, col2 = st.columns(2)

        with col1:
            # Распределение тональности
            sentiment_counts = self.df['sentiment_category'].value_counts()
            colors = ['#ff6b6b', '#feca57', '#48dbfb']
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Распределение тональности',
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Scatter plot рейтинг vs тональность
            fig_scatter = px.scatter(
                self.df,
                x='rating',
                y='sentiment_score',
                color='sentiment_category',
                title='Рейтинг vs Тональность',
                labels={'rating': 'Рейтинг', 'sentiment_score': 'Score тональности'},
                hover_data=['text']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Корреляция
        correlation = self.df['rating'].corr(self.df['sentiment_score'])
        st.info(f"Корреляция между рейтингом и тональностью: {correlation:.3f}")

    def show_text_analysis(self):
        """Показывает анализ текста"""
        st.header("📝 Анализ текста отзывов")

        col1, col2 = st.columns(2)

        with col1:
            # Гистограмма длины отзывов
            fig_hist = px.histogram(
                self.df,
                x='word_count',
                nbins=20,
                title='Распределение длины отзывов',
                labels={'word_count': 'Количество слов', 'count': 'Частота'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Box plot длины по рейтингам
            fig_box = px.box(
                self.df,
                x='rating',
                y='word_count',
                title='Длина отзывов по рейтингам',
                labels={'rating': 'Рейтинг', 'word_count': 'Количество слов'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Облако слов
        st.subheader("☁️ Облако слов")
        if st.button("Создать облако слов"):
            all_text = ' '.join(self.df['clean_text'].dropna())
            if all_text.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(all_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

    def show_time_analysis(self):
        """Показывает временной анализ"""
        st.header("📅 Временной анализ")

        # Группировка по месяцам
        monthly_data = self.df.set_index('date').resample('M').agg({
            'rating': 'mean',
            'sentiment_score': 'mean',
            'text': 'count'
        }).rename(columns={'text': 'review_count'})

        col1, col2 = st.columns(2)

        with col1:
            # Количество отзывов по времени
            fig_line1 = px.line(
                x=monthly_data.index,
                y=monthly_data['review_count'],
                title='Количество отзывов по времени',
                labels={'x': 'Дата', 'y': 'Количество отзывов'}
            )
            st.plotly_chart(fig_line1, use_container_width=True)

        with col2:
            # Средний рейтинг по времени
            fig_line2 = px.line(
                x=monthly_data.index,
                y=monthly_data['rating'],
                title='Средний рейтинг по времени',
                labels={'x': 'Дата', 'y': 'Средний рейтинг'}
            )
            st.plotly_chart(fig_line2, use_container_width=True)

    def show_detailed_reviews(self):
        """Показывает детальный анализ отзывов"""
        st.header("🔍 Детальный анализ отзывов")

        # Фильтры
        col1, col2, col3 = st.columns(3)

        with col1:
            rating_filter = st.selectbox(
                "Фильтр по рейтингу:",
                ["Все"] + list(sorted(self.df['rating'].unique()))
            )

        with col2:
            sentiment_filter = st.selectbox(
                "Фильтр по тональности:",
                ["Все"] + list(self.df['sentiment_category'].unique())
            )

        with col3:
            sort_by = st.selectbox(
                "Сортировать по:",
                ["Дате", "Рейтингу", "Тональности"]
            )

        # Применяем фильтры
        filtered_df = self.df.copy()

        if rating_filter != "Все":
            filtered_df = filtered_df[filtered_df['rating'] == rating_filter]

        if sentiment_filter != "Все":
            filtered_df = filtered_df[filtered_df['sentiment_category'] == sentiment_filter]

        # Сортировка
        if sort_by == "Дате":
            filtered_df = filtered_df.sort_values('date', ascending=False)
        elif sort_by == "Рейтингу":
            filtered_df = filtered_df.sort_values('rating', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('sentiment_score', ascending=False)

        # Показываем отзывы
        st.write(f"Найдено отзывов: {len(filtered_df)}")

        for idx, row in filtered_df.head(10).iterrows():
            with st.expander(
                    f"Рейтинг: {row['rating']} | Тональность: {row['sentiment_category']} | {row['date'].strftime('%Y-%m-%d')}"):
                st.write(f"**Автор:** {row['author']}")
                st.write(f"**Текст:** {row['text']}")
                st.write(f"**Score тональности:** {row['sentiment_score']:.3f}")
                st.write(f"**Количество слов:** {row['word_count']}")

    def show_insights(self):
        """Показывает основные инсайты"""
        st.header("💡 Основные инсайты")

        # Общая статистика
        total_reviews = len(self.df)
        avg_rating = self.df['rating'].mean()
        positive_ratio = len(self.df[self.df['sentiment_category'] == 'Позитивная']) / total_reviews * 100
        correlation = self.df['rating'].corr(self.df['sentiment_score'])

        insights = [
            f"📊 Проанализировано **{total_reviews}** отзывов",
            f"⭐ Средний рейтинг составляет **{avg_rating:.2f}** из 5",
            f"😊 **{positive_ratio:.1f}%** отзывов имеют позитивную тональность",
            f"🔗 Корреляция между рейтингом и тональностью: **{correlation:.3f}**"
        ]

        if correlation > 0.5:
            insights.append("✅ Высокое соответствие между рейтингом и тональностью")
        elif correlation > 0.3:
            insights.append("⚠️ Умеренное соответствие между рейтингом и тональностью")
        else:
            insights.append("❌ Слабое соответствие между рейтингом и тональностью")

        # Распределение по рейтингам
        rating_dist = self.df['rating'].value_counts().sort_index()
        most_common_rating = rating_dist.idxmax()
        high_ratings = (rating_dist.get(4, 0) + rating_dist.get(5, 0)) / total_reviews * 100
        low_ratings = (rating_dist.get(1, 0) + rating_dist.get(2, 0)) / total_reviews * 100

        insights.extend([
            f"🏆 Наиболее частый рейтинг: **{most_common_rating}**",
            f"👍 Высокие рейтинги (4-5): **{high_ratings:.1f}%**",
            f"👎 Низкие рейтинги (1-2): **{low_ratings:.1f}%**"
        ])

        for insight in insights:
            st.write(insight)

        # Экстремальные отзывы
        st.subheader("🎯 Экстремальные отзывы")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Самый позитивный отзыв:**")
            most_positive = self.df.loc[self.df['sentiment_score'].idxmax()]
            st.info(f"Рейтинг: {most_positive['rating']}, Score: {most_positive['sentiment_score']:.3f}")
            st.write(f"*{most_positive['text']}*")

        with col2:
            st.write("**Самый негативный отзыв:**")
            most_negative = self.df.loc[self.df['sentiment_score'].idxmin()]
            st.error(f"Рейтинг: {most_negative['rating']}, Score: {most_negative['sentiment_score']:.3f}")
            st.write(f"*{most_negative['text']}*")

    def run_dashboard(self):
        """Запускает дашборд"""
        # Пытаемся загрузить данные
        if not self.load_data():
            st.error("Данные не найдены! Сначала запустите скрипты сбора и обработки данных.")
            st.code("""
# Для запуска анализа выполните:
python src/scraping.py
python src/processing.py
python src/analysis.py
            """)
            return

        # Сайдбар с навигацией
        st.sidebar.title("Навигация")
        pages = {
            "Главная": self.show_header,
            "Анализ рейтингов": self.show_rating_analysis,
            "Анализ тональности": self.show_sentiment_analysis,
            "Анализ текста": self.show_text_analysis,
            "Временной анализ": self.show_time_analysis,
            "Детальные отзывы": self.show_detailed_reviews,
            "Инсайты": self.show_insights
        }

        selected_page = st.sidebar.selectbox("Выберите раздел:", list(pages.keys()))

        # Показываем заголовок всегда
        if selected_page == "Главная":
            self.show_header()
            st.markdown("### Добро пожаловать в систему анализа отзывов!")
            st.write("Используйте боковое меню для навигации по разделам анализа.")
        else:
            pages[selected_page]()


def main():
    dashboard = ReviewDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()