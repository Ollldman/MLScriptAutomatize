import logging
from typing import Dict, Any
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

# Цветовая палитра (можно легко изменить)
COLORS = {
    "background": "#1e1e2a",  # Тёмно-серо-синий
    "surface": "#2d2d44",     # Сlightly lighter
    "card": "rgba(50, 50, 70, 180)",  # Стеклянная карточка
    "text_primary": "#f0f0f0",
    "text_secondary": "#c0c0d0",
    "accent": "#6a6acc",      # Светло-синий акцент
    "button_bg": "rgba(80, 80, 100, 180)",
    "button_hover": "rgba(100, 100, 130, 200)",
    "border": "rgba(120, 120, 140, 0.4)",
    "highlight": "rgba(106, 106, 204, 0.3)"
}


def apply_dark_glass_style(app):
    """
    Применяет тёмную тему с эффектом стекла ко всему приложению.
    """
    # Шрифт
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Палитра
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 42))        # #1e1e2a
    palette.setColor(QPalette.WindowText, QColor(240, 240, 240)) # #f0f0f0
    palette.setColor(QPalette.Base, QColor(45, 45, 68))          # #2d2d44
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 70))
    palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 42))
    palette.setColor(QPalette.ToolTipText, QColor(240, 240, 240))
    palette.setColor(QPalette.Text, QColor(240, 240, 240))
    palette.setColor(QPalette.Button, QColor(80, 80, 100))       # #505064
    palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    palette.setColor(QPalette.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.Link, QColor(106, 106, 204))       # #6a6acc
    palette.setColor(QPalette.Highlight, QColor(106, 106, 204, 50))  # #6a6acc с прозрачностью
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)

    # ✅ CSS-стили
    app.setStyleSheet(f"""
        /* GLOBAL */
        * {{
            font-family: "Segoe UI";
            font-size: 10pt;
        }}
        QWidget {{
            background-color: {COLORS["background"]};
            color: {COLORS["text_primary"]};
        }}
        QMainWindow {{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {COLORS["background"]}, stop:1 {COLORS["surface"]});
            border-radius: 12px;
        }}
        /* BUTTONS */
        QPushButton {{
            background-color: {COLORS["button_bg"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
            padding: 8px 16px;
            color: {COLORS["text_primary"]};
            min-height: 36px;
        }}
        QPushButton:hover {{
            background-color: {COLORS["button_hover"]};
            border-color: {COLORS["accent"]};
        }}
        QPushButton:pressed {{
            background-color: rgba(70, 70, 90, 180);
        }}
        /* LABELS */
        QLabel {{
            color: {COLORS["text_primary"]};
            padding: 4px;
        }}
        /* FRAMES / CARDS */
        QFrame {{
            background-color: {COLORS["card"]};
            border-radius: 12px;
            border: 1px solid {COLORS["border"]};
            padding: 12px;
            margin: 8px;
        }}
        /* GROUPBOXES */
        QGroupBox {{
            font-weight: bold;
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
            margin-top: 1.2ex;
            padding: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: rgba(45, 45, 68, 150);
            border-radius: 6px;
        }}
        /* SCROLLBAR */
        QScrollBar:vertical {{
            border: none;
            background: rgba(60, 60, 80, 150);
            width: 10px;
            margin: 0px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(100, 100, 130, 0.7);
            border-radius: 5px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(120, 120, 150, 0.9);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        /* INPUTS */
        QLineEdit {{
            background-color: rgba(60, 60, 80, 150);
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 6px;
            color: {COLORS["text_primary"]};
        }}
        QLineEdit:focus {{
            border-color: {COLORS["accent"]};
        }}
        QComboBox {{
            background-color: rgba(60, 60, 80, 150);
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 6px;
            color: {COLORS["text_primary"]};
        }}
        QComboBox:focus {{
            border-color: {COLORS["accent"]};
        }}
        /* PROGRESS BAR */
        QProgressBar {{
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            background-color: rgba(50, 50, 70, 150);
            text-align: center;
            color: {COLORS["text_secondary"]};
        }}
        QProgressBar::chunk {{
            background-color: {COLORS["accent"]};
            border-radius: 6px;
        }}
        QTextEdit {{
            background-color: rgba(50, 50, 60, 180);
            border: 1px solid rgba(100, 100, 120, 0.5);
            border-radius: 8px;
            padding: 10px;
            color: #f0f0f0;
            font-size: 13px;
        }}
        QTextEdit:focus {{
            border-color: #6a6acc;
        }}
    """)

    logger.info("Dark glass style applied successfully.")