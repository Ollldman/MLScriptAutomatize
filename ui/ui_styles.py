from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush
from PyQt5.QtCore import Qt

def set_glass_style(app):
    """
    Применяет современный стеклянный (glassmorphism) стиль к приложению.
    Поддерживает полупрозрачность, тени, градиенты и чистый шрифт.
    """
    # Устанавливаем шрифт
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Создаем палитру с прозрачным фоном
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255, 180))  # Белый с прозрачностью
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))      # Темный текст
    palette.setColor(QPalette.Base, QColor(240, 240, 240, 160))    # Фон полей
    palette.setColor(QPalette.AlternateBase, QColor(250, 250, 250, 170))
    palette.setColor(QPalette.Text, QColor(30, 30, 30))
    palette.setColor(QPalette.Button, QColor(255, 255, 255, 190))  # Кнопки — стекло
    palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218, 180))  # Подсветка — синий с прозрачностью
    palette.setColor(QPalette.HighlightedText, Qt.white)

    app.setPalette(palette)

    # Включаем антиалиасинг и прозрачность
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app.setAttribute(Qt.WA_TranslucentBackground, True)
    app.setStyleSheet("""
        QWidget {
            background-color: rgba(255, 255, 255, 0);
            border: none;
        }
        QMainWindow {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0f7ff, stop:1 #f0f0ff);
            border-radius: 12px;
        }
        QPushButton {
            background-color: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(150, 150, 150, 0.3);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 500;
            color: #1e1e1e;
            min-height: 40px;
        }
        QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.95);
            border-color: rgba(100, 100, 100, 0.5);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        QPushButton:pressed {
            background-color: rgba(240, 240, 240, 0.9);
            transform: translateY(1px);
        }
        QLabel {
            color: #1e1e1e;
            font-size: 14px;
            padding: 5px;
        }
        QFrame {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            border: 1px solid rgba(200, 200, 200, 0.4);
            padding: 15px;
            margin: 10px;
        }
        QGroupBox {
            font-weight: bold;
            color: #1e1e1e;
            border: 1px solid rgba(180, 180, 180, 0.5);
            border-radius: 10px;
            margin-top: 15px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
        }
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background: rgba(240, 240, 240, 0.5);
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: rgba(150, 150, 150, 0.7);
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: rgba(120, 120, 120, 0.8);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)