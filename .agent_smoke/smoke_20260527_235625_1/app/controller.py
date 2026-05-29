class PDFCombinerController:
    import PyPDF2

    def combine_pdfs(self, pdf_files):
        merger = PyPDF2.PdfMerger()
        
        for file in pdf_files:
            with open(file, 'rb') as f:
                merger.append(f)
        
        combined_pdf_path = "combined.pdf"
        merger.write(combined_pdf_path)
        merger.close()
        
        return combined_pdf_path
