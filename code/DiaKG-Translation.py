from google.cloud import translate_v3
import json, os, glob

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"translation-for-diakg-7d84427b6e13.json"

client = translate_v3.TranslationServiceClient()
project_id = "translation-for-diakg"
parent     = f"projects/900014886527/locations/global"


def translate_text(text: str, target_language: str = "en") -> str:
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
        # take the first translation
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
    input_directory = "0521_new_format"
    output_directory = "0521_new_format_translated"
    process_all_json_files(input_directory, output_directory)
