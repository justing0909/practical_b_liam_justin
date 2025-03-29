import os
import fitz

def remove_whitespace_from_pdf(pdf_path, output_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Loop through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract text to identify whitespace bounding box
        text_instances = page.search_for(' ')  # Find space characters
        
        for inst in text_instances:
            # Remove the identified whitespace (optional, depends on your use case)
            page.delete_text(inst)

        # Alternatively, you could use cropping to remove whitespace:
        rect = page.rect
        new_rect = fitz.Rect(rect.x0 + 10, rect.y0 + 10, rect.x1 - 10, rect.y1 - 10)  # Example cropping
        page.set_cropbox(new_rect)

    # Save the modified PDF to the output path
    doc.save(output_path)

def process_pdfs_in_directory(directory_path):
    # Loop through the directory and process each PDF file
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            output_path = os.path.join(directory_path, "modified_" + filename)
            
            # Remove whitespace from each PDF
            remove_whitespace_from_pdf(pdf_path, output_path)
            print(f"Processed {filename}")

# Set the directory containing the PDFs
directory_path = "PDFs"

# Process all PDFs in the directory
process_pdfs_in_directory(directory_path)