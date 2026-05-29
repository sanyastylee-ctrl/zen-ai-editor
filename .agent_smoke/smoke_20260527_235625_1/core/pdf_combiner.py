import PyPDF2

def combine_pdfs(pdf_list, output_path):
    pdf_writer = PyPDF2.PdfFileWriter()
    
    for pdf in pdf_list:
        with open(pdf, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            for page_num in range(pdf_reader.numPages):
                pdf_writer.addPage(pdf_reader.getPage(page_num))
    
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

