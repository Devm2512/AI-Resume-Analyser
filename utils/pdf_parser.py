import PyPDF2


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Extracted text
    """

    text = ""

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Check if PDF is empty
            if not pdf_reader.pages:
                raise ValueError("PDF has no readable pages")

            for page in pdf_reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + " "

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    # Clean unwanted characters (basic level)
    text = text.replace("\x0c", " ")  # remove page breaks

    return text.strip()