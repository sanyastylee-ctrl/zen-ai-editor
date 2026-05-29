from app.controller import PDFCombinerController

def main():
    controller = PDFCombinerController()
    pdf_files = ["file1.pdf", "file2.pdf"]  # Example list of PDF files
    combined_pdf = controller.combine_pdfs(pdf_files)
    print(f"Combined PDF saved as: {combined_pdf}")

if __name__ == "__main__":
    main()
