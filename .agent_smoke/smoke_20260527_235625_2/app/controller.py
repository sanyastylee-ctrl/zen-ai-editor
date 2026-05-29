import PyPDF2

def combine_pdfs(pdf_files, output_path):
    pdf_writer = PyPDF2.PdfFileWriter()
    
    for file in pdf_files:
        with open(file, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                pdf_writer.addPage(page)
    
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

def main():
    pdf_files = ['file1.pdf', 'file2.pdf']  # Пример списка PDF файлов
    output_path = 'combined_output.pdf'     # Путь для сохранения объединенного файла
    
    combine_pdfs(pdf_files, output_path)
