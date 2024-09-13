import pymupdf4llm

# Extract text and pdf from the given document
def extract_text_and_images_from_docs(obj_path,  image_output_folder, pages):
    md_text = pymupdf4llm.to_markdown(
        obj_path,
        image_format="jpeg",
        write_images=True,
        image_path=image_output_folder,
        pages=pages
    )
    return md_text

# Extract metadata from the given document
def collect_metadata(pdf_path):
    extracted_metadata = pymupdf4llm.to_markdown(pdf_path, pages=[0], page_chunks=True)
    metadata = extracted_metadata[0]['metadata']
    # remove page number from metadata
    metadata.pop('page', None)
    return metadata