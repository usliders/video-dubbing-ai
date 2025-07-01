# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копируем только requirements.txt для кэширования слоёв
COPY requirements.txt ./

# Установка Python-зависимостей
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт для Jupyter
EXPOSE 8888

# По умолчанию — Jupyter, но можно запускать любой скрипт через docker run ...
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["if [ \"$1\" = 'python' ]; then shift; exec python \"$@\"; else exec jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/app/notebooks; fi", "--"] 