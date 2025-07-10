import os
from pydub import AudioSegment
# from pyannote.audio import Pipeline  # если потребуется VAD

# Пример VAD с pyannote.audio (требует pip install pyannote.audio)
def run_vad(audio_path, output_path, model_name='pyannote/voice-activity-detection', min_duration=0.5):
    """
    Выделяет голосовые сегменты из аудио с помощью pyannote.audio и сохраняет результат в текстовый файл.
    Каждый сегмент: start (сек), end (сек)
    Требует HuggingFace Token для доступа к модели (гейтед).
    """
    from pyannote.audio import Pipeline
    # Токен пользователя для HuggingFace (гейтед модель)
    HF_TOKEN = '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'
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    vad = pipeline({'audio': audio_path})
    with open(output_path, 'w') as f:
        for speech in vad.get_timeline():
            # Фильтруем слишком короткие сегменты
            if (speech.end - speech.start) >= min_duration:
                f.write(f"{speech.start:.2f}\t{speech.end:.2f}\n")
    print(f"[VAD] Сегменты речи сохранены: {output_path}")

# Нарезка аудио по сегментам (start, end)
def split_audio_by_segments(audio_path, segments, output_dir):
    """
    Нарезает аудио на фрагменты по сегментам (start, end).
    """
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(os.path.join(output_dir, f'segment_{i}.wav'), format='wav')

# Конвертация аудио в нужный формат и sample rate
def convert_audio_format(input_path, output_path, sr=16000):
    """
    Конвертирует аудиофайл в нужный формат и sample rate.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio.export(output_path, format='wav') 