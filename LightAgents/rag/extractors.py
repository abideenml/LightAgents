import pymupdf4llm

def extract_text_and_images_from_docs(obj_path,  image_output_folder, pages):
    md_text = pymupdf4llm.to_markdown(
        obj_path,
        image_format="jpeg",
        write_images=True,
        image_path=image_output_folder,
        pages=pages
    )
    return md_text

def collect_metadata(pdf_path):
    extracted_metadata = pymupdf4llm.to_markdown(pdf_path, pages=[0], page_chunks=True)
    metadata = extracted_metadata[0]['metadata']
    return metadata