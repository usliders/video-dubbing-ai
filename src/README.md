# src/

Весь основной код проекта. Структура модульная, каждый компонент — в своей папке.

- `vision_audio/` — обработка аудио и видео (извлечение, замена, сборка дорожек)
- `asr/` — распознавание речи (ASR)
- `mt/` — машинный перевод
- `tts/` — синтез речи и клонирование голоса (zero/few-shot)
- `eval/` — оценка качества (similarity, сравнение эмбеддингов)
- `utils/` — утилиты, логирование, вспомогательные функции

Дополнительные папки (агенты, пайплайны, интеграция с LLM и др.) — для расширения функционала.

**Рекомендуется начать знакомство с проектом с основного [README.md](../README.md).** 