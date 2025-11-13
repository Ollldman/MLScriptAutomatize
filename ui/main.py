import sys
from PyQt5.QtWidgets import QApplication
from ui.ui_main_window import MainWindow
from ui.ui_styles import apply_dark_glass_style

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_glass_style(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())