# Модуль для обработки видео и аудио дорожек
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

class VideoProcessor:
    @staticmethod
    def extract_audio(video_path, output_path=None):
        print(f"[Video] Извлечение аудио из видео: {video_path}")
        if output_path is None:
            output_path = "data/temp/extracted_audio.wav"
        if os.path.exists(video_path):
            try:
                video = VideoFileClip(video_path)
                audio = video.audio
                if audio is not None:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    audio.write_audiofile(output_path, fps=16000)
                    print(f"[Video] Аудио сохранено: {output_path}")
                    return output_path
                else:
                    print("[Video] В видео нет аудиодорожки!")
            except Exception as e:
                print(f"[Video] Ошибка при извлечении аудио: {e}")
        else:
            print("[Video] Файл видео не найден, возвращаю заглушку.")
        return output_path

    @staticmethod
    def replace_audio(video_path, audio_path, output_path):
        print(f"[Video] Замена аудио в видео: {video_path} -> {audio_path}, результат: {output_path}")
        if os.path.exists(video_path) and os.path.exists(audio_path):
            try:
                video = VideoFileClip(video_path)
                audio = AudioFileClip(audio_path)
                video = video.set_audio(audio)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                video.write_videofile(output_path, fps=30)
                print(f"[Video] Видео с новым аудио сохранено: {output_path}")
            except Exception as e:
                print(f"[Video] Ошибка при замене аудио: {e}")
        else:
            print("[Video] Не найден видеофайл или аудиофайл, создаю пустой файл-результат (заглушка).")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(b"")

def assemble_audio_by_segments(segments, base_audio_path=None, sr=16000, output_path="data/temp/assembled_audio.wav"):
    """
    Склеивает аудиофайлы сегментов в одну дорожку с учётом таймингов.
    segments: список словарей {'start': float, 'end': float, 'audio_path': str}
    base_audio_path: путь к оригинальному аудио (для длины)
    sr: sample rate
    output_path: куда сохранить результат
    """
    print(f"[AUDIO] Сборка аудио по сегментам, всего сегментов: {len(segments)}")
    if len(segments) < 2:
        print("[AUDIO][WARNING] Сегментов для склейки меньше 2! Возможно, что-то пошло не так.")
    if base_audio_path and os.path.exists(base_audio_path):
        base_audio = AudioSegment.from_file(base_audio_path)
        total_duration = len(base_audio)
    else:
        # Если нет оригинала, считаем по последнему сегменту
        total_duration = int(1000 * max(seg['end'] for seg in segments))
    result = AudioSegment.silent(duration=total_duration)
    for idx, seg in enumerate(segments):
        seg_audio = AudioSegment.from_file(seg['audio_path'])
        seg_start = int(seg['start'] * 1000)
        seg_end = int(seg['end'] * 1000)
        seg_duration = seg_end - seg_start
        print(f"[AUDIO][SEGMENT] #{idx}: {seg['audio_path']} | start={seg['start']:.2f}s end={seg['end']:.2f}s dur={seg_duration/1000:.2f}s audio_len={len(seg_audio)/1000:.2f}s")
        # Если сгенерированное аудио длиннее — обрезаем, если короче — добавляем тишину
        if len(seg_audio) > seg_duration:
            seg_audio = seg_audio[:seg_duration]
        elif len(seg_audio) < seg_duration:
            seg_audio += AudioSegment.silent(duration=seg_duration - len(seg_audio))
        result = result.overlay(seg_audio, position=seg_start)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.export(output_path, format="wav")
    print(f"[AUDIO] Итоговое аудио сохранено: {output_path}")
    print(f"[AUDIO] Итоговая длительность: {len(result)/1000:.2f} секунд")
    if len(result) < 2000:
        print("[AUDIO][WARNING] Итоговое аудио подозрительно короткое! Проверьте входные сегменты.")
    return output_path 