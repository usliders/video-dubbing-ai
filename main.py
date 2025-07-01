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
        print("1. Запустить пайплайн дубляжа видео (zero-shot)")
        print("2. Запустить пайплайн дубляжа видео (few-shot/fine-tune)")
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
                reference_audio_path = os.path.join(temp_dir, "run_extracted_audio.wav")  # будет извлечён из видео
                output_path = os.path.join(output_dir, f"output_zero_shot_{timestamp}.mp4")
                dubbing_main(
                    video_path=video_path,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    use_gpu=use_gpu,
                    temp_dir=temp_dir,
                    mode="zero-shot"
                )
            elif choice == "2":
                from pipelines.video_dubbing_pipeline import main as dubbing_main
                video_path = os.path.join(input_dir, "input.mp4")
                reference_audio_path = os.path.join(input_dir, "reference_audio1.wav")
                output_path = os.path.join(output_dir, f"output_few_shot_{timestamp}.mp4")
                dubbing_main(
                    video_path=video_path,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    use_gpu=use_gpu,
                    temp_dir=temp_dir,
                    mode="few-shot"
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