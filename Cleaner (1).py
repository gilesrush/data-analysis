import os
import glob
import re

# Set the folder with your .txt files
FOLDER_PATH = "./corpus"  # change as needed

# Regex: match only Sino-Japanese characters (Kanji, Hiragana, Katakana)
sinojapanese_regex = re.compile(r'[^\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]')

# Get all .txt files
txt_files = glob.glob(os.path.join(FOLDER_PATH, "*.txt"))

for filepath in txt_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove non Sino-Japanese characters
    cleaned = sinojapanese_regex.sub('', content)

    # Overwrite the file with cleaned content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"Cleaned: {os.path.basename(filepath)}")

print("✅ All files cleaned.")