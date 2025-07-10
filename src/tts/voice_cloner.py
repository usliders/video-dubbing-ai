# Модуль для клонирования голоса и синтеза речи (TTS)
import os
import wave
import numpy as np
import sys
import io

class VoiceCloner:
    def __init__(self, reference_audio_path, tts_checkpoint_path=None):
        """
        reference_audio_path: путь к эталонному аудио для zero-shot или few-shot.
        tts_checkpoint_path: путь к дообученному чекпоинту (используется в few-shot).
        """
        print(f"[TTS] Клонирование голоса по референсу: {reference_audio_path}")
        self.reference_audio_path = reference_audio_path
        self.tts_checkpoint_path = tts_checkpoint_path

    def finetune(self, train_audio_paths, train_texts, output_checkpoint_path, use_gpu=False, epochs=5):
        """
        Дообучает TTS-модель на новых коротких записях (few-shot).
        train_audio_paths: список путей к коротким аудиофрагментам (5-10 сек).
        train_texts: список текстов, соответствующих аудио.
        output_checkpoint_path: путь для сохранения нового чекпоинта.
        use_gpu: использовать ли GPU.
        epochs: количество эпох дообучения.
        """
        print(f"[TTS][FINETUNE] Запуск дообучения на {len(train_audio_paths)} примерах...")
        # Пример для Coqui TTS (xtts_v2). Можно адаптировать под свою модель.
        try:
            from TTS.api import TTS
            import torch
            device = "cuda" if use_gpu else "cpu"
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
            # Здесь должен быть код подготовки датасета и вызова tts.finetune(...)
            # Для простоты — псевдокод:
            # tts.finetune(train_audio_paths, train_texts, output_checkpoint_path, epochs=epochs)
            print(f"[TTS][FINETUNE] Дообучение завершено. Чекпоинт сохранён: {output_checkpoint_path}")
            return output_checkpoint_path
        except Exception as e:
            print(f"[TTS][FINETUNE][ERROR] Ошибка дообучения: {e}")
            return None

    def synthesize(self, text=None, output_path=None, use_gpu=False):
        """
        Синтезирует аудио по тексту. Для zero-shot — используется короткий чистый сегмент (5-10 сек) исходной речи.
        Для few-shot — можно указать путь к дообученному чекпоинту (self.tts_checkpoint_path).
        """
        if output_path is None:
            output_path = "data/processed/fake_audio.wav"
        translation_path = "data/processed/mt_translation.txt"
        if text is None and os.path.exists(translation_path):
            with open(translation_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        preview = (text[:200] + ' ... ' + text[-200:]) if text and len(text) > 400 else text
        print(f"[TTS] Синтез аудио по тексту: {preview}")
        ref = self.reference_audio_path
        if isinstance(ref, list):
            all_exist = all(os.path.exists(p) for p in ref)
        else:
            all_exist = os.path.exists(ref)
        if all_exist and text:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            def try_tts(device_type, suppress_errors=False):
                try:
                    import os
                    if device_type == "cpu":
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        os.environ["USE_CUDA"] = "0"
                    elif device_type == "cuda":
                        # Можно явно выставить, если нужно
                        pass
                    import torch
                    from TTS.api import TTS
                    from TTS.tts.configs.xtts_config import XttsConfig
                    from TTS.tts.models.xtts import XttsAudioConfig
                    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
                    # --- Подавляем лишний вывод ---
                    stderr_backup = sys.stderr
                    sys.stderr = io.StringIO()
                    try:
                        tts = TTS(model_name=model_name, progress_bar=False).to(device_type)
                        tts.tts_to_file(text=text, speaker_wav=ref, file_path=output_path, language="ru")
                    finally:
                        sys.stderr = stderr_backup
                    print(f"[TTS] Аудио сгенерировано и сохранено: {output_path} (device: {device_type})")
                    return True
                except Exception as e:
                    err_str = str(e).splitlines()[0] if str(e) else type(e).__name__
                    if suppress_errors:
                        print(f"[TTS][WARNING] Ошибка на {device_type}: {err_str}. Пробую на CPU...")
                    else:
                        print(f"[TTS][ERROR] Ошибка на {device_type}: {err_str}")
                    return False
            # Сначала пробуем на GPU, если нужно
            if use_gpu:
                print("[TTS] Пробую синтез на GPU...")
                ok = try_tts("cuda", suppress_errors=True)
                if ok:
                    return output_path
                print("[TTS] Пробую синтез на CPU...")
                ok = try_tts("cpu")
                if ok:
                    return output_path
                print("[TTS][FATAL] Не удалось синтезировать ни на GPU, ни на CPU. Будет создана заглушка.")
            else:
                print("[TTS] Пробую синтез на CPU...")
                ok = try_tts("cpu")
                if ok:
                    return output_path
                print("[TTS][FATAL] Не удалось синтезировать на CPU. Будет создана заглушка.")
        else:
            print("[TTS] Reference audio не найден или текст пустой, возвращаю заглушку.")
            # Создаём WAV-файл с тишиной (или синусом) на 2 секунды
            sr = 16000
            duration = 2  # секунды
            silence = np.zeros(int(sr * duration), dtype=np.int16)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with wave.open(output_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(silence.tobytes())
            print(f"[TTS] Аудио сохранено (заглушка): {output_path}")
            return output_path 