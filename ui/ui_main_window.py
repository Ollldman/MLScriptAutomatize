from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QFrame, QSizePolicy, QScrollArea, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import sys
import os

# Импортируем шаги
from ui.crisp_steps.step_1_business_understanding import Step1BusinessUnderstanding
from ui.crisp_steps.step_2_data_understanding import Step2DataUnderstanding
from ui.crisp_steps.step_3_data_preparation import Step3DataPreparation
from ui.crisp_steps.step_4_modeling import Step4Modeling
from ui.crisp_steps.step_5_evaluation import Step5Evaluation
from ui.crisp_steps.step_6_deployment import Step6Deployment

from ModelForge.modules.report.report_data import ReportData


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shared_data = ReportData()

        self.setWindowTitle("📊 ModelForge — Автоматизированный ML-анализ")
        self.setGeometry(100, 100, 1200, 750)
        self.setMinimumSize(900, 600)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Флаг состояния навигации
        self.nav_collapsed = False

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Горизонтальный контейнер для навигации и контента
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # === Боковая панель навигации ===
        self.nav_frame = QFrame()
        self.nav_frame.setFixedWidth(320)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.nav_frame.setObjectName("nav_frame")

        nav_layout = QVBoxLayout(self.nav_frame)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(8)

        # Кнопка сворачивания
        self.toggle_nav_btn = QPushButton("◀▶")
        self.toggle_nav_btn.setFixedHeight(60)
        self.toggle_nav_btn.setFixedWidth(60)
        self.toggle_nav_btn.setText("◀" if not self.nav_collapsed else "▶")
        self.toggle_nav_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(138, 43, 226, 0.8);
                color: white;
                border-radius: 8px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
            }
        """)
        self.toggle_nav_btn.clicked.connect(self.toggle_navigation)
        nav_layout.addWidget(self.toggle_nav_btn)

        # Кнопки навигации
        self.nav_buttons = {}
        steps = [
            ("1. Business Understanding", Step1BusinessUnderstanding),
            ("2. Data Understanding", Step2DataUnderstanding),
            ("3. Data Preparation", Step3DataPreparation),
            ("4. Modeling", Step4Modeling),
            ("5. Evaluation", Step5Evaluation),
            ("6. Deployment", Step6Deployment),
        ]
        self.nav_button_texts = [
            "1. Business Understanding",
            "2. Data Understanding",
            "3. Data Preparation", 
            "4. Modeling",
            "5. Evaluation",
            "6. Deployment",
        ]

        for step_name, step_class in steps:
            btn = QPushButton(step_name)
            btn.setFixedHeight(40)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, name=step_name: self.on_nav_click(name))
            nav_layout.addWidget(btn)
            self.nav_buttons[step_name] = btn

        # Заполнитель внизу
        nav_layout.addStretch()

        content_layout.addWidget(self.nav_frame)

        # === Рабочая область с прокруткой ===
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content_scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                border: none;
                background: rgba(60, 60, 70, 150);
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(138, 43, 226, 0.7),
                    stop:1 rgba(255, 20, 147, 0.7));
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(138, 43, 226, 1),
                    stop:1 rgba(255, 20, 147, 1));
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background: transparent;")
        for step_name, step_class in steps:
            step_instance = step_class(self.shared_data)
            self.stacked_widget.addWidget(step_instance)
            break

        self.content_scroll.setWidget(self.stacked_widget)
        content_layout.addWidget(self.content_scroll)

        # === Заголовок ===
        title_label = QLabel("📊 ModelForge — Автоматизированный ML-анализ")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 23, QFont.Bold))
        title_label.setStyleSheet("""
            color: #f0f0ff;
            padding: 15px;
            background-color: rgba(50, 50, 60, 0.8);
            border-radius: 12px;
            margin: 10px;
        """)

        main_layout.insertWidget(0, title_label)

        # === Кнопка сохранения отчёта ===
        save_btn = QPushButton("💾 Сохранить отчёт (PDF/HTML)")
        save_btn.setFixedHeight(40)
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8B00FF, stop:1 #FF1493);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                margin: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 105, 180, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7A00E6, stop:1 #DB00AA);
            }
        """)
        save_btn.clicked.connect(self.on_save_report)
        main_layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # === Инициализация текущего шага ===
        self.current_step = 0
        self.stacked_widget.setCurrentIndex(self.current_step)
        self.nav_buttons["1. Business Understanding"].setChecked(True)

    def toggle_navigation(self):
        if self.nav_collapsed:
            self.nav_frame.setFixedWidth(320)
            self.toggle_nav_btn.setText("◀▶")
            for i, (step_name, btn) in enumerate(self.nav_buttons.items()):
                btn.setText(self.nav_button_texts[i])
                btn.setFixedWidth(260)  # Вернём ширину кнопки
        else:
            self.nav_frame.setFixedWidth(120)
            self.toggle_nav_btn.setText("▶")
            for i, (step_name, btn) in enumerate(self.nav_buttons.items()):
                btn.setText(f"{i + 1}.")
                btn.setFixedWidth(60)  # Узкая кнопка
        self.nav_collapsed = not self.nav_collapsed

    def on_nav_click(self, step_name):
        # Отключаем все кнопки
        for btn in self.nav_buttons.values():
            btn.setChecked(False)

        # Включаем текущую
        self.nav_buttons[step_name].setChecked(True)

        # Получаем индекс шага
        step_index = list(self.nav_buttons.keys()).index(step_name)

        # Проверяем, можно ли перейти на этот шаг
        if step_index > self.current_step:
            # Только если это следующий шаг
            if step_index == self.current_step + 1:
                self.current_step = step_index
                self.stacked_widget.setCurrentIndex(step_index)
            else:
                # Нельзя пропускать шаги
                self.nav_buttons[list(self.nav_buttons.keys())[self.current_step]].setChecked(True)
                return
        else:
            # Переход назад — разрешён всегда
            self.current_step = step_index
            self.stacked_widget.setCurrentIndex(step_index)

    def on_save_report(self):
        """
        Открывает диалог выбора директории для сохранения.
        В будущем вызовет generate_automl_report.
        """
        from PyQt5.QtWidgets import QMessageBox
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения отчёта")
        if not folder:
            QMessageBox.information(self, "Сохранение", "Папка не выбрана.")
            return

        # ⚠️ Здесь будет вызов report_generator.generate_automl_report(...)
        # Сейчас заглушка
        QMessageBox.information(
            self,
            "Отчёт сохранён",
            f"Файл отчёта будет сохранён в:\n{folder}\n\n"
            "В реальной версии здесь будет генерация HTML/PDF."
        )