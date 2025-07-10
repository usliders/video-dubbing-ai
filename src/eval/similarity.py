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

def similarity_upper_bound(segment_paths):
    """
    Вычисляет среднее значение similarity между всеми парами оригинальных сегментов одного спикера.
    Используется для оценки верхней границы метрики (upper bound).
    """
    embeddings = [extract_embedding(p) for p in segment_paths]
    n = len(embeddings)
    if n < 2:
        return None
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(compute_similarity(embeddings[i], embeddings[j]))
    if sims:
        return np.mean(sims)
    return None

# Пример пакетной оценки
# def batch_evaluate(ref_paths, gen_paths):
#     sims = []
#     for ref, gen in zip(ref_paths, gen_paths):
#         emb1 = extract_embedding(ref)
#         emb2 = extract_embedding(gen)
#         sims.append(compute_similarity(emb1, emb2))
#     return sims 