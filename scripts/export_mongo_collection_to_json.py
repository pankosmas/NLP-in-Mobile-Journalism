import json

# Load the file with individual JSON objects
input_file1 = "articles_speeches.json"
output_file1 = "articles_speeches_fixed.json"

# Load the file with individual JSON objects
input_file2 = "news_articles.json"
output_file2 = "news_articles_fixed.json"

# Open with UTF-8 encoding
with open(input_file1, "r", encoding="utf-8") as infile:
    data = [json.loads(line) for line in infile]

# Save as a valid JSON array
with open(output_file1, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, ensure_ascii=False)

# Open with UTF-8 encoding
with open(input_file2, "r", encoding="utf-8") as infile:
    data = [json.loads(line) for line in infile]

# Save as a valid JSON array
with open(output_file2, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, ensure_ascii=False)

print("File converted to JSON array format!")
