

# Data Analysis Project

## Описание
Этот проект посвящён анализу данных с использованием Python. В рамках проекта выполняется загрузка, очистка и визуализация данных, а также получение основных инсайтов.  
Цель — продемонстрировать навыки работы с данными и построения простого аналитического пайплайна.

## Структура проекта
- `src/` — исходные скрипты анализа и визуализации  
- `requirements.txt` — список необходимых библиотек для запуска проекта  
- `data` - для хранения данных

## Используемые данные
Формат данных — CSV или другой табличный формат.

## Основные этапы анализа
1. Загрузка и предварительный просмотр данных  
2. Очистка и подготовка данных к анализу  
3. Визуализация ключевых метрик и закономерностей  
4. Аналитика и выводы  

## Как запустить проект

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/ErbolTakhirov/Data_analysis_project.git
   cd Data_analysis_project
   ```

2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

3. Собрать данные:
   ```
   cd src 
   python scraping.py   
   ```
4. Анализ данных:
   ```
   python analysis.py
   python processing.py
   ```
5. Визуализация:   (streamlit приложение)
   ```
   streamlit run visualisation.py
   ```


## Используемые технологии

- Python 3.7+  
- pandas  
- numpy  
- matplotlib / seaborn
- scikit-learn
- wordcloud
- beautifulsoup4
- requests
- textblob
- plotly
- streamlit
- lxml

## Контакты

Автор: Erbol Takhirov  
GitHub: [https://github.com/ErbolTakhirov](https://github.com/ErbolTakhirov)  
