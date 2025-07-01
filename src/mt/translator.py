# Модуль для машинного перевода (MT)
import os

class Translator:
    def translate(self, text=None, src_lang='en', tgt_lang='ru', use_gpu=False, output_path=None):
        translation = "Привет, это тестовая транскрипция."
        preview = (text[:200] + ' ... ' + text[-200:]) if text and len(text) > 400 else text
        print(f"[MT] Перевод текста с {src_lang} на {tgt_lang}: {preview}")
        if text:
            try:
                from transformers import MarianMTModel, MarianTokenizer, logging
                import torch
                
                # Подавляем лишние логи от transformers
                logging.set_verbosity_error()

                model_name = "Helsinki-NLP/opus-mt-en-ru"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
                print(f"[MT] Используется устройство: {device}")
                try:
                    model = model.to(device)
                    max_length = 512
                    sentences = text.split('. ')
                    chunks = []
                    current = ''
                    for sent in sentences:
                        candidate = (current + sent + '. ').strip()
                        num_tokens = len(tokenizer(candidate, return_tensors="pt")["input_ids"][0])
                        if num_tokens < max_length:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = sent + '. '
                    if current:
                        chunks.append(current.strip())
                    translations = []
                    for chunk in chunks:
                        batch = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                        batch = {k: v.to(device) for k, v in batch.items()}
                        gen = model.generate(**batch)
                        translated = tokenizer.decode(gen[0], skip_special_tokens=True)
                        translations.append(translated)
                    translation = '\n'.join(translations)
                except Exception as e:
                    if device.type == "cuda":
                        print(f"[MT][ERROR] Ошибка на GPU: {e}. Пробую на CPU...")
                        model = model.to("cpu")
                        translations = []
                        for chunk in chunks:
                            batch = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                            batch = {k: v.to("cpu") for k, v in batch.items()}
                            gen = model.generate(**batch)
                            translated = tokenizer.decode(gen[0], skip_special_tokens=True)
                            translations.append(translated)
                        translation = '\n'.join(translations)
                        print(f"[MT] Успешно на CPU.")
                    else:
                        raise
            except ImportError:
                print("[MT] Модуль transformers не установлен, используем заглушку.")
            except Exception as e:
                print(f"[MT] Ошибка при переводе: {e}")
        if text:
            preview_tr = (translation[:200] + ' ... ' + translation[-200:]) if translation and len(translation) > 400 else translation
            print(f"[MT] Превью перевода: {preview_tr}")
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translation)
            return translation
        else:
            print("[MT] Пустой текст, возвращаю заглушку.")
            return translation 