import numpy as np
from scipy.spatial.distance import cosine

# Пример: загрузка speaker encoder из SpeechBrain
# pip install speechbrain
from speechbrain.inference import EncoderClassifier
import torchaudio

# Загрузка модели (один раз)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def extract_embedding(audio_path):
    """
    Извлекает speaker embedding из аудиофайла с помощью SpeechBrain.
    Возвращает numpy-массив эмбеддинга.
    """
    signal, fs = torchaudio.load(audio_path)
    emb = classifier.encode_batch(signal).detach().cpu().numpy()[0][0]
    return emb

def compute_similarity(embedding1, embedding2):
    """
    Вычисляет косинусное сходство между двумя эмбеддингами.
    Возвращает значение similarity (1 - cosine).
    """
    return 1 - cosine(embedding1, embedding2)

# Пример пакетной оценки
# def batch_evaluate(ref_paths, gen_paths):
#     sims = []
#     for ref, gen in zip(ref_paths, gen_paths):
#         emb1 = extract_embedding(ref)
#         emb2 = extract_embedding(gen)
#         sims.append(compute_similarity(emb1, emb2))
#     return sims 