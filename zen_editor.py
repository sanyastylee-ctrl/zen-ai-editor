import os
from PyQt6.QtWidgets import QTreeView
from PyQt6.QtGui import QFileSystemModel

class ZenEditor:
    def __init__(self):
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(os.getcwd()))
        self.tree_view.hideColumn(1)  # Hide size column
        self.tree_view.hideColumn(2)  # Hide type column
        self.tree_view.hideColumn(3)  # Hide date column
        
        # Clear sidebar_layout and add tree_view to it
        self.sidebar_layout.removeWidget(self.sidebar_stretch)
        self.sidebar_layout.addWidget(self.tree_view)
        
        # Set transparent background and white text for the tree view
        self.tree_view.setStyleSheet("QTreeView {background-color: rgba(0, 0, 0, 0); color: white;}")
        
        # Connect double click event to open file method
        self.tree_view.doubleClicked.connect(self.open_file)

    def open_file(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.code_editor.setPlainText(content)
