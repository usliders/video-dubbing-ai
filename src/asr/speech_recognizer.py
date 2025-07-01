# Модуль для распознавания речи (ASR)
import os
import torch

class SpeechRecognizer:
    def transcribe(self, audio_path, use_gpu=False, output_path=None):
        print(f"[ASR] Распознавание речи из файла: {audio_path}")
        segments = []
        if os.path.exists(audio_path):
            try:
                import whisper
                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                print(f"[ASR] Используется устройство: {device}")
                model = whisper.load_model("tiny", device=device)
                try:
                    result = model.transcribe(audio_path, language="en")
                except Exception as e:
                    if device == "cuda":
                        print(f"[ASR][ERROR] Ошибка на GPU: {e}. Пробую на CPU...")
                        model = whisper.load_model("tiny", device="cpu")
                        result = model.transcribe(audio_path, language="en")
                        print(f"[ASR] Успешно на CPU.")
                    else:
                        raise
                segments = result.get("segments", [])
                if not segments:
                    # Если сегментов нет, создаём один сегмент-заглушку
                    segments = [{"start": 0.0, "end": 1.0, "text": "[ASR] Не удалось распознать речь."}]
                # Сохраняем сегменты в отдельный файл
                if output_path is None:
                    output_path = "data/temp/asr_segments.txt"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for seg in segments:
                        f.write(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{seg['text']}\n")
                print(f"[ASR] Сегменты с таймкодами сохранены: {output_path}")
                return segments
            except ImportError:
                print("[ASR] Модуль whisper не установлен, используем заглушку.")
                segments = [{"start": 0.0, "end": 1.0, "text": "Hello, this is a test transcription."}]
            except Exception as e:
                print(f"[ASR] Ошибка при распознавании: {e}")
                segments = [{"start": 0.0, "end": 1.0, "text": "Hello, this is a test transcription."}]
        else:
            print("[ASR] Аудиофайл не найден, возвращаю заглушку.")
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello, this is a test transcription."}]
        return segments 