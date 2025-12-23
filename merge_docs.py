import os

# 1. Docs folder ka path (rag-api ke relative path se)
docs_folder = "../docs"

# 2. Output file
output_file = "book.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    # Saare files ko alphabetically read karo
    for filename in sorted(os.listdir(docs_folder)):
        # Sirf .md ya .txt files
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(docs_folder, filename)
            with open(filepath, "r", encoding="utf-8") as infile:
                # Chapter ka header
                outfile.write(f"# {filename}\n\n")
                # File ka content
                outfile.write(infile.read())
                outfile.write("\n\n")

print(f"✅ book.txt successfully created with all chapters from {docs_folder}")
