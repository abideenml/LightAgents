import pymupdf4llm


md_text = pymupdf4llm.to_markdown("documents/human-nutrition-text.pdf" ,page_chunks=True,image_format="jpeg",write_images=True , image_path="images" , pages=[42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60])
# tokenizer = Tokenizer.from_pretrained("BAAI/bge-en-icl")

print(md_text[0]['metadata'])
print(md_text[0]['metadata']['title'])
print(md_text[0]['metadata']['author'])
print(md_text[0]['metadata']['producer'])

# max_tokens = 400
# splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, max_tokens , overlap=30)
# all_pages_text = []
# for page in md_text:
#     page_text = page['text']
#     all_pages_text.append(page_text)
# chunks = splitter.chunks(str(all_pages_text))
# for chunk in chunks:
#     print(chunk)
# pageNo = 0
# for chunks in md_text:
#     pageNo += 1
#     print(f"Page {pageNo} : ")
#     print(chunks['metadata'])
#     print("\nText : ")
#     print(chunks['text'])
#     print("\nImages : ")
#     print(chunks['images'])
    


# pathlib.Path("documents/output.md").write_bytes(md_text.encode())



