"""
Пайплайн для дубляжа видео:
1. Распознавание речи (ASR)
2. Машинный перевод (MT)
3. Клонирование голоса и генерация аудио (TTS)
4. Сборка финального видео
"""

from asr.speech_recognizer import SpeechRecognizer
from mt.translator import Translator
from tts.voice_cloner import VoiceCloner
from vision_audio.video_processor import VideoProcessor, assemble_audio_by_segments
import glob
import os
from datetime import datetime
from shutil import copyfile
from pathlib import Path
import re
import numpy as np, wave
from main import PipelineFatalError


def clean_temp_files(temp_dir, prefix):
    temp_patterns = [
        os.path.join(temp_dir, f"{prefix}_tts_segment_*.wav"),
        os.path.join(temp_dir, f"{prefix}_assembled_audio.wav"),
        os.path.join(temp_dir, f"{prefix}_asr_segments.txt"),
        os.path.join(temp_dir, f"{prefix}_asr_segments_ru.txt"),
    ]
    for pattern in temp_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"[CLEANUP] Удалён временный файл: {file}")
            except Exception as e:
                print(f"[CLEANUP] Не удалось удалить {file}: {e}")


def main(video_path, reference_audio_path, output_path, use_gpu=False, temp_dir="data/temp", mode="zero-shot"):
    # Получаем timestamp из output_path, если он есть
    m = re.search(r'(\d{8}_\d{4})', output_path)
    if m:
        timestamp = m.group(1)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    prefix = "run"
    os.makedirs(temp_dir, exist_ok=True)
    clean_temp_files(temp_dir, prefix)

    # 1. Извлечение аудио из видео (и сохраняем в temp_dir)
    extracted_audio_path = os.path.join(temp_dir, f"{prefix}_extracted_audio.wav")
    audio_path = VideoProcessor.extract_audio(video_path, output_path=extracted_audio_path)

    # 2. Распознавание речи (ASR)
    asr = SpeechRecognizer()
    asr_segments_path = os.path.join(temp_dir, f"{prefix}_asr_segments.txt")
    segments = asr.transcribe(audio_path, use_gpu=use_gpu, output_path=asr_segments_path)
    if not segments or len(segments) == 0:
        raise PipelineFatalError("ASR не удалось выполнить ни на одном устройстве. Нет сегментов для обработки.")

    # 3. Перевод всех сегментов (MT) — сохраняем txt в output, если уже есть, не переводим заново
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    mt_segments_path = os.path.join(temp_dir, f"{prefix}_asr_segments_ru.txt")
    mt_segments_output = os.path.join(output_dir, f"asr_segments_ru_{timestamp}.txt")
    if os.path.exists(mt_segments_output):
        print(f"[MT] Файл перевода уже существует: {mt_segments_output}, пропускаю перевод.")
    else:
        mt = Translator()
        mt_segments = []
        for seg in segments:
            src_text = seg['text']
            start = seg['start']
            end = seg['end']
            try:
                translated = mt.translate(src_text, src_lang='en', tgt_lang='ru', use_gpu=use_gpu)
            except Exception as e:
                raise PipelineFatalError(f"MT не удалось выполнить ни на GPU, ни на CPU: {e}")
            mt_segments.append({'start': start, 'end': end, 'text': translated})
        with open(mt_segments_path, "w", encoding="utf-8") as f:
            for seg in mt_segments:
                f.write(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{seg['text']}\n")
        print(f"[MT] Сегменты с переводом сохранены: {mt_segments_path}")
        # Копируем в output
        copyfile(mt_segments_path, mt_segments_output)
        print(f"[MT] Копия перевода сохранена в output: {mt_segments_output}")
    # Если файл уже был, используем его для TTS
    if os.path.exists(mt_segments_output):
        mt_segments_path = mt_segments_output

    # 4. Генерация аудио по сегментам (TTS)
    tts = VoiceCloner(reference_audio_path)
    processed_segments = []
    with open(mt_segments_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                print(f"[PIPELINE][WARNING] Строка сегмента {i} некорректна: {line.strip()}")
                continue
            start, end, text = float(parts[0]), float(parts[1]), parts[2]
            tts_path = os.path.join(temp_dir, f"{prefix}_tts_segment_{i}.wav")
            try:
                segment_audio_path = tts.synthesize(text, output_path=tts_path, use_gpu=use_gpu)
                print(f"[PIPELINE][TTS] Сегмент {i} успешно сгенерирован: {segment_audio_path}")
            except Exception as e:
                raise PipelineFatalError(f"Ошибка при генерации сегмента {i}: {e}. Пайплайн остановлен, выберите режим заново.")
            processed_segments.append({
                'start': start,
                'end': end,
                'audio_path': segment_audio_path
            })
    if not processed_segments:
        raise PipelineFatalError("TTS не удалось выполнить ни на одном устройстве. Нет сгенерированных сегментов.")

    # 5. Сборка итоговой аудиодорожки
    final_audio_path = os.path.join(temp_dir, f"{prefix}_assembled_audio.wav")
    final_audio_path = assemble_audio_by_segments(processed_segments, base_audio_path=extracted_audio_path, output_path=final_audio_path)

    # 6. Сборка видео с новым аудио
    VideoProcessor.replace_audio(video_path, final_audio_path, output_path)
    print(f"Готово! Видео сохранено: {output_path}")


if __name__ == "__main__":
    # TODO: добавить парсинг аргументов командной строки
    main("data/raw/input.mp4", "data/raw/reference_audio.wav", "data/processed/output.mp4") 