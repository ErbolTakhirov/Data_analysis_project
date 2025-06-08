import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin, urlparse
import os


class ReviewScraper:
    def __init__(self):
        self.session = requests.Session()
        # Добавляем заголовки чтобы выглядеть как обычный браузер
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_reviews_from_text(self, sample_reviews_text):
        """
        Создает образцы отзывов для демонстрации
        В реальном проекте здесь был бы код для парсинга конкретного сайта
        """
        reviews = []

        # Примеры отзывов на русском языке
        sample_reviews = [
            {
                'rating': 5,
                'text': 'Отличный товар! Качество превзошло все ожидания. Доставка была быстрой.',
                'date': '2024-01-15',
                'author': 'Анна К.'
            },
            {
                'rating': 4,
                'text': 'Хороший продукт, но упаковка могла быть лучше. В целом доволен покупкой.',
                'date': '2024-01-18',
                'author': 'Михаил П.'
            },
            {
                'rating': 3,
                'text': 'Средний товар. Есть как плюсы, так и минусы. За эту цену нормально.',
                'date': '2024-01-20',
                'author': 'Елена В.'
            },
            {
                'rating': 2,
                'text': 'Не очень качественно сделано. Не соответствует описанию на сайте.',
                'date': '2024-01-22',
                'author': 'Дмитрий Л.'
            },
            {
                'rating': 1,
                'text': 'Ужасное качество! Деньги потрачены зря. Не рекомендую никому.',
                'date': '2024-01-25',
                'author': 'Ольга С.'
            },
            {
                'rating': 5,
                'text': 'Просто великолепно! Быстрая доставка, отличное качество, рекомендую всем!',
                'date': '2024-01-28',
                'author': 'Алексей Н.'
            },
            {
                'rating': 4,
                'text': 'Качественный товар. Небольшие недочеты, но в целом очень хорошо.',
                'date': '2024-02-01',
                'author': 'Мария Т.'
            },
            {
                'rating': 3,
                'text': 'Обычный товар. Ничего особенного, но свою функцию выполняет.',
                'date': '2024-02-03',
                'author': 'Владимир К.'
            },
            {
                'rating': 5,
                'text': 'Превосходное качество! Уже второй раз заказываю. Очень доволен!',
                'date': '2024-02-05',
                'author': 'Ирина М.'
            },
            {
                'rating': 2,
                'text': 'Товар пришел с дефектами. Служба поддержки отвечает медленно.',
                'date': '2024-02-08',
                'author': 'Сергей Б.'
            }
        ]

        return sample_reviews

    def save_reviews_to_csv(self, reviews, filename='reviews.csv'):
        """Сохраняет отзывы в CSV файл"""
        df = pd.DataFrame(reviews)

        # Создаем директорию если её нет
        os.makedirs('data/raw', exist_ok=True)

        filepath = f'data/raw/{filename}'
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Сохранено {len(reviews)} отзывов в файл {filepath}")

        return df

    def get_sample_data(self):
        """Получает образцы данных для анализа"""
        print("Получение образцов отзывов...")

        # В реальном проекте здесь был бы код для скрапинга
        reviews = self.scrape_reviews_from_text("")

        # Добавляем больше разнообразных отзывов
        additional_reviews = [
            {'rating': 4, 'text': 'Хорошее соотношение цена-качество. Доставили вовремя.', 'date': '2024-02-10',
             'author': 'Андрей Ф.'},
            {'rating': 5, 'text': 'Отличный сервис! Товар соответствует всем ожиданиям.', 'date': '2024-02-12',
             'author': 'Наталья Р.'},
            {'rating': 3, 'text': 'Нормальный товар, но есть нюансы в качестве материала.', 'date': '2024-02-14',
             'author': 'Павел Ж.'},
            {'rating': 1, 'text': 'Полное разочарование. Товар не работает как заявлено.', 'date': '2024-02-16',
             'author': 'Татьяна Д.'},
            {'rating': 5, 'text': 'Замечательная покупка! Буду рекомендовать друзьям.', 'date': '2024-02-18',
             'author': 'Игорь В.'},
            {'rating': 4, 'text': 'Почти идеально. Единственный минус - долгая доставка.', 'date': '2024-02-20',
             'author': 'Юлия К.'},
            {'rating': 2, 'text': 'Качество хуже ожидаемого. Не стоит своих денег.', 'date': '2024-02-22',
             'author': 'Роман М.'},
            {'rating': 5, 'text': 'Прекрасный товар! Качество на высшем уровне.', 'date': '2024-02-24',
             'author': 'Светлана П.'},
        ]

        reviews.extend(additional_reviews)

        # Сохраняем в CSV
        df = self.save_reviews_to_csv(reviews)

        return df


if __name__ == "__main__":
    scraper = ReviewScraper()
    data = scraper.get_sample_data()
    print("Скрапинг завершен!")
    print(f"Собрано отзывов: {len(data)}")
    print("\nПервые 3 отзыва:")
    print(data.head(3))