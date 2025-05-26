from google.cloud import translate_v3
import json, os, glob

# point at your JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"translation-for-diakg-7d84427b6e13.json"

client = translate_v3.TranslationServiceClient()
project_id = "translation-for-diakg"
parent     = f"projects/900014886527/locations/global"


def translate_text(text: str, target_language: str = "en") -> str:
    #Translate Chinese text to English using Google Cloud Translation API v3.

    try:
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "zh",
                "target_language_code": target_language.lower(),
            }
        )
        # take the first (and only) translation
        return response.translations[0].translated_text
    except Exception as e:
        print(f"[Translate ERROR] {e} -- text: {text[:30]}…")
        return "ERROR"


def process_json_file(file_path, output_dir):
    with open(file_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    for paragraph in document.get("paragraphs", []):
        paragraph_text = paragraph.get("paragraph")
        if paragraph_text:
            paragraph["paragraph_en"] = translate_text(paragraph_text, target_language="en")

        for sentence in paragraph.get("sentences", []):
            sentence_text = sentence.get("sentence")
            if sentence_text:
                sentence["sentence_en"] = translate_text(sentence_text, target_language="en")

            for entity in sentence.get("entities", []):
                chinese_text = entity.get("entity")
                if chinese_text:
                    entity["entity_en"] = translate_text(chinese_text, target_language="en")

    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(document, f_out, ensure_ascii=False, indent=2)
    print(f"Processed and saved: {output_file}")


def process_all_json_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(file_list)} JSON files.")
    for file_path in file_list:
        process_json_file(file_path, output_dir)

if __name__ == "__main__":
    input_directory = "diaKG/0521_new_format"
    output_directory = "diaKG/diakg/0521_new_format_translated"
    process_all_json_files(input_directory, output_directory)


"""
import requests
import json
import os
import glob
import csv

# ========== DeepL API Configuration ==========
API_KEY = 'bbbf45b2-9a83-46a9-a1f2-45a4bb42b9c8'
API_URL = 'https://api.deepl.com/v2/translate'

# ========== Translate One Text Block ==========
def translate_text(text, target_language='EN'):
    data = {
        'auth_key': API_KEY,
        'text': text,
        'source_lang': 'ZH',
        'target_lang': target_language.upper()
    }
    response = requests.post(API_URL, data=data)
    if response.status_code == 200:
        return response.json()["translations"][0]["text"]
    else:
        print(f"Translation error ({response.status_code}): {text}")
        return "ERROR"

import os
from mistralai import Mistral

# Initialize client once at module level
_api_key = os.environ["MISTRAL_API_KEY"]
_client  = Mistral(api_key=_api_key)
_model   = "mistral-large-latest"

def mistral_translate(text: str) -> str:
    #Translate Chinese medical text into domain-specific English
    #using Mistral’s chat completion API.
    
    prompt = (
        "You are a professional medical translator specializing in endocrinology and chronic disease management.\n"
        "Your task is to accurately translate the following Chinese medical text into fluent, natural English, ensuring precise use of diabetes-related terminology.\n\n"
        "Guidelines:\n"
        "- Maintain the original meaning and tone (e.g., clinical, instructional, patient-facing).\n"
        "- Accurately convert all medical terms, lab results, and treatment concepts.\n"
        "- Preserve units of measurement (e.g., mmol/L, mmHg) without conversion.\n"
        "- Use appropriate clinical English terms (e.g., 'HbA1c', 'Insulin secretagogues').\n"
        "- If a term is ambiguous, choose the most medically accurate interpretation.\n"
        "- Do not add, omit, or generalize content.\n\n"
        f"Chinese Text:\n{text}\n\nTranslation:"
    )

    response = _client.chat.complete(
        model       = _model,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.2,
    )

    # Return just the generated translation
    return response.choices[0].message.content.strip()



# ========== Step 1: Extract Unique Entities ==========
def extract_unique_entities(input_dir):
    unique_entities = set()
    file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)
            for paragraph in document.get("paragraphs", []):
                for sentence in paragraph.get("sentences", []):
                    for entity in sentence.get("entities", []):
                        entity_text = entity.get("entity")
                        if entity_text:
                            unique_entities.add(entity_text)
    return list(unique_entities)

# ========== Step 2: Translate and Export to CSV ==========
def translate_entities_to_csv(entities, csv_output_path):
    with open(csv_output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Chinese", "English"])
        for entity in entities:
            translated = mistral_translate(entity)
            writer.writerow([entity, translated])
    print(f"Saved translated entities to: {csv_output_path}")

# ========== Step 4: Load Corrected Dictionary ==========
def load_translation_dict(csv_file):
    translation_dict = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zh = row["Chinese"].strip()
            en = row["English"].strip()
            if en and en != "ERROR":
                translation_dict[zh] = en
    return translation_dict

# ========== Step 5: Reprocess JSONs Using the Dictionary ==========
def process_json_file(file_path, output_dir, translation_dict):
    with open(file_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    for paragraph in document.get("paragraphs", []):
        paragraph_text = paragraph.get("paragraph")
        if paragraph_text:
            paragraph["paragraph_en"] = mistral_translate(paragraph_text)

        for sentence in paragraph.get("sentences", []):
            sentence_text = sentence.get("sentence")
            if sentence_text:
                sentence["sentence_en"] = mistral_translate(sentence_text)

            for entity in sentence.get("entities", []):
                chinese_text = entity.get("entity")
                if chinese_text:
                    entity["entity_en"] = translation_dict.get(chinese_text, mistral_translate(chinese_text))

    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(document, f_out, ensure_ascii=False, indent=2)
    print(f"Processed and saved: {output_file}")

def process_all_json_files(input_dir, output_dir, translation_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(file_list)} JSON files.")
    for file_path in file_list:
        process_json_file(file_path, output_dir, translation_dict)


if __name__ == "__main__":
    input_directory = "diaKG/0521_new_format"
    output_directory = "diaKG/0521_new_format_translated"
    entity_csv_file = "translated_entities.csv"

    # Step 1: Extract + translate unique entities to CSV
    entities = extract_unique_entities(input_directory)
    translate_entities_to_csv(entities, entity_csv_file)

    # Step 2: After manual correction in CSV, run this:
    #translation_dict = load_translation_dict(entity_csv_file)
    #process_all_json_files(input_directory, output_directory, translation_dict)


import os
import json
import csv
import glob

def extract_entities_to_csv(input_dir, output_csv_path):
    seen = set()
    rows = []

    file_list = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(file_list)} files to scan for entities.")

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)
            for paragraph in document.get("paragraphs", []):
                for sentence in paragraph.get("sentences", []):
                    for entity in sentence.get("entities", []):
                        zh = entity.get("entity", "").strip()
                        en = entity.get("entity_en", "").strip()
                        if zh and en and (zh, en) not in seen:
                            seen.add((zh, en))
                            rows.append([zh, en])

    # Sort alphabetically by Chinese term
    rows.sort(key=lambda x: x[0])

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chinese", "English"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} unique entity pairs to: {output_csv_path}")

if __name__ == "__main__":
    input_dir = "diaKG/diakg/0521_new_format_translated"
    output_csv = "translated_entity_glossary.csv"
    extract_entities_to_csv(input_dir, output_csv)
"""