import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


def set_glass_style(app):
    """
    Применяет современный тёмный стеклянный (glassmorphism) стиль к приложению.
    Поддерживает полупрозрачность, тени, градиенты и чистый шрифт.
    """
    # Устанавливаем шрифт
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Создаем палитру с тёмными цветами
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 40, 220))  # Тёмно-серый с прозрачностью
    palette.setColor(QPalette.WindowText, QColor(240, 240, 240))  # Светлый текст
    palette.setColor(QPalette.Base, QColor(40, 40, 50, 180))      # Фон полей
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 60, 180))
    palette.setColor(QPalette.Text, QColor(240, 240, 240))
    palette.setColor(QPalette.Button, QColor(60, 60, 70, 200))    # Кнопки — тёмно-серые с прозрачностью
    palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    palette.setColor(QPalette.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.Link, QColor(100, 200, 255))
    palette.setColor(QPalette.Highlight, QColor(70, 130, 180, 180))  # Подсветка — синий с прозрачностью
    palette.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)

    app.setPalette(palette)

    # ✅ УБРАЛИ вызов app.setAttribute(Qt.WA_TranslucentBackground, True) — он вызывает предупреждения в CSS
    # ✅ УБРАЛИ вызов AA_EnableHighDpiScaling — он должен быть установлен ДО создания QApplication

    app.setStyleSheet("""
        /* Global */
        QWidget {
            background-color: rgba(30, 30, 40, 220);
            color: #f0f0f0;
            border-radius: 12px;
        }
        QMainWindow {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e1e2a, stop:1 #2d2d44);
            border-radius: 15px;
        }
        /* Buttons */
        QPushButton {
            background-color: rgba(60, 60, 70, 200);
            border: 1px solid rgba(100, 100, 120, 0.5);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 500;
            color: #f0f0f0;
            min-height: 40px;
        }
        QPushButton:hover {
            background-color: rgba(80, 80, 90, 220);
            border-color: rgba(130, 130, 150, 0.7);
            /* box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); */
        }
        QPushButton:pressed {
            background-color: rgba(50, 50, 60, 200);
            /* transform: translateY(1px); */
        }
        /* Labels */
        QLabel {
            color: #e0e0e0;
            font-size: 14px;
            padding: 5px;
        }
        /* Frames */
        QFrame {
            background-color: rgba(40, 40, 50, 180);
            border-radius: 12px;
            border: 1px solid rgba(100, 100, 120, 0.4);
            padding: 15px;
            margin: 10px;
        }
        /* GroupBoxes */
        QGroupBox {
            font-weight: bold;
            color: #e0e0e0;
            border: 1px solid rgba(120, 120, 140, 0.5);
            border-radius: 10px;
            margin-top: 15px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            background-color: rgba(50, 50, 60, 180);
            border-radius: 8px;
        }
        /* Scroll Areas */
        QScrollArea {
            border: none;
            background: transparent;
        }
        /* Scrollbars */
        QScrollBar:vertical {
            border: none;
            background: rgba(60, 60, 70, 150);
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: rgba(120, 120, 140, 0.7);
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: rgba(140, 140, 160, 0.8);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)