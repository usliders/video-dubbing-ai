import sys
import os
from datetime import datetime

# Добавляем src в PYTHONPATH для корректного импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def get_timestamp():
    # Формат: YYYYMMDD_HHMM
    return datetime.now().strftime('%Y%m%d_%H%M')

class PipelineFatalError(Exception):
    """Фатальная ошибка пайплайна, для возврата в меню без завершения скрипта."""
    pass

def main():
    while True:
        print("=== Video Dubbing AI 2025 ===")
        print("Выберите режим устройства:")
        print("1. Использовать GPU (если доступно)")
        print("2. Использовать только CPU")
        device_choice = input("Введите номер режима: ").strip()
        use_gpu = False
        if device_choice == "1":
            use_gpu = True
        elif device_choice == "2":
            use_gpu = False
        else:
            print("Неизвестный выбор. По умолчанию используется CPU.")
            use_gpu = False

        print("Выберите действие:")
        print("1. Запустить пайплайн дубляжа видео (zero-shot, короткий сегмент)")
        print("2. Запустить пайплайн дубляжа видео (few-shot/fine-tune, дообучение)")
        print("0. Выйти")

        choice = input("Введите номер действия: ").strip()

        input_dir = os.path.join("data", "input")
        output_dir = os.path.join("data", "output")
        temp_dir = os.path.join("data", "temp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = get_timestamp()

        try:
            if choice == "1":
                from pipelines.video_dubbing_pipeline import main as dubbing_main
                video_path = os.path.join(input_dir, "input.mp4")
                # Автоматический выбор reference audio для zero-shot
                zero_shot_ref = os.path.join(input_dir, "zero_shot_reference.wav")
                temp_ref = os.path.join(temp_dir, "run_extracted_audio.wav")
                # Если нет zero_shot_reference.wav, используем run_extracted_audio.wav
                if not os.path.exists(zero_shot_ref):
                    if os.path.exists(temp_ref):
                        reference_audio_path = temp_ref
                        print(f"[INFO] zero_shot_reference.wav не найден, используем оригинальное аудио из видео: {temp_ref}")
                    else:
                        # fallback: первый reference_audio*.wav
                        import glob
                        ref_candidates = glob.glob(os.path.join(input_dir, "reference_audio*.wav"))
                        if ref_candidates:
                            reference_audio_path = ref_candidates[0]
                            print(f"[INFO] zero_shot_reference.wav и run_extracted_audio.wav не найдены, используем: {reference_audio_path}")
                        else:
                            print("[ERROR] Не найден ни один reference audio для zero-shot. Добавьте zero_shot_reference.wav или reference_audio*.wav в папку input.")
                            return
                else:
                    reference_audio_path = zero_shot_ref
                output_path = os.path.join(output_dir, f"output_zero_shot_{timestamp}.mp4")
                vad_enabled = input("Включить VAD-сегментацию аудио перед ASR? (y/n): ").strip().lower() == 'y'
                dubbing_main(
                    video_path=video_path,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    use_gpu=use_gpu,
                    temp_dir=temp_dir,
                    mode="zero-shot",
                    vad_enabled=vad_enabled
                )
            elif choice == "2":
                from pipelines.video_dubbing_pipeline import main as dubbing_main
                video_path = os.path.join(input_dir, "input.mp4")
                # Автоматический выбор reference audio для few-shot
                few_shot_ref = os.path.join(input_dir, "few_shot_reference.wav")
                import glob
                ref_candidates = glob.glob(os.path.join(input_dir, "reference_audio*.wav"))
                if os.path.exists(few_shot_ref):
                    reference_audio_path = few_shot_ref
                elif ref_candidates:
                    reference_audio_path = ref_candidates[0]
                    print(f"[INFO] few_shot_reference.wav не найден, используем: {reference_audio_path}")
                else:
                    print("[ERROR] Не найден ни один reference audio для few-shot. Добавьте few_shot_reference.wav или reference_audio*.wav в папку input.")
                    return
                # Собираем все reference_audio*.wav для дообучения
                finetune_audio_paths = ref_candidates if ref_candidates else [reference_audio_path]
                # Генерируем тексты-заглушки для дообучения (или можно загрузить из файла)
                finetune_texts = [f"Текст для аудио {i+1}" for i in range(len(finetune_audio_paths))]
                finetune_data = {'audio_paths': finetune_audio_paths, 'texts': finetune_texts}
                output_path = os.path.join(output_dir, f"output_few_shot_{timestamp}.mp4")
                vad_enabled = input("Включить VAD-сегментацию аудио перед ASR? (y/n): ").strip().lower() == 'y'
                dubbing_main(
                    video_path=video_path,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    use_gpu=use_gpu,
                    temp_dir=temp_dir,
                    mode="few-shot",
                    vad_enabled=vad_enabled,
                    finetune_data=finetune_data,
                    finetune_epochs=5
                )
            elif choice == "0":
                print("Выход.")
                sys.exit(0)
            else:
                print("Неизвестный выбор. Попробуйте снова.")
        except PipelineFatalError as e:
            print(f"[FATAL] {e}")
            print("Возврат в главное меню.\n")
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Выполнение прервано пользователем. Возврат в меню.\n")

if __name__ == "__main__":
    main() 