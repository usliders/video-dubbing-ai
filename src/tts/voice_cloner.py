# Модуль для клонирования голоса и синтеза речи (TTS)
import os
import wave
import numpy as np
import sys
import io

class VoiceCloner:
    def __init__(self, reference_audio_path):
        print(f"[TTS] Клонирование голоса по референсу: {reference_audio_path}")
        self.reference_audio_path = reference_audio_path

    def synthesize(self, text=None, output_path=None, use_gpu=False):
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