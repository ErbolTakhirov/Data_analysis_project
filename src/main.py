#!/usr/bin/env python3
"""
Главный скрипт для пет-проекта анализа отзывов покупателей
Автор: Junior Data Analyst
"""

import os
import sys

# Добавляем src в путь для импорта модулей
sys.path.append('')

from scraping import ReviewScraper
from processing import ReviewProcessor
from analysis import ReviewAnalyzer


def create_directories():
    """Создает необходимые директории для проекта"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Создана директория: {directory}")


def run_full_pipeline():
    """Запускает полный пайплайн анализа отзывов"""
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА АНАЛИЗА ОТЗЫВОВ")
    print("=" * 60)

    # 1. Создаем структуру проекта
    print("\n1️⃣ Создание структуры проекта...")
    create_directories()

    # 2. Сбор данных (скрапинг)
    print("\n2️⃣ Сбор данных...")
    scraper = ReviewScraper()
    try:
        raw_data = scraper.get_sample_data()
        print(f"✓ Собрано {len(raw_data)} отзывов")
    except Exception as e:
        print(f"❌ Ошибка при сборе данных: {e}")
        return False

    # 3. Обработка данных
    print("\n3️⃣ Обработка данных...")
    processor = ReviewProcessor()
    try:
        # Загружаем сырые данные
        df = processor.load_data()
        if df is not None:
            # Обрабатываем
            processed_df = processor.process_reviews(df)
            if processed_df is not None:
                # Сохраняем
                processor.save_processed_data(processed_df)

                # Выводим статистику
                stats = processor.get_summary_stats(processed_df)
                print("✓ Статистика обработанных данных:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("❌ Ошибка при обработке данных")
                return False
        else:
            print("❌ Не удалось загрузить сырые данные")
            return False
    except Exception as e:
        print(f"❌ Ошибка при обработке данных: {e}")
        return False

    # 4. Анализ данных
    print("\n4️⃣ Анализ данных...")
    analyzer = ReviewAnalyzer()
    try:
        analyzer.run_full_analysis()
        print("✓ Анализ данных завершен")
    except Exception as e:
        print(f"❌ Ошибка при анализе данных: {e}")
        return False

    # 5.