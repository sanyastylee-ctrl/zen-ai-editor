import sys
from PyQt6.QtWidgets import QApplication, QStyleFactory
from ui.main_window import ZenEditor

def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    window = ZenEditor()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()