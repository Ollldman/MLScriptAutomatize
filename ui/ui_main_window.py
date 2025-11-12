# UI/ui_main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import sys
import os

# Импортируем шаги
from ui.crisp_steps.step_1_business_understanding import Step1BusinessUnderstanding
from ui.crisp_steps.step_2_data_understanding import Step2DataUnderstanding
from ui.crisp_steps.step_3_data_preparation import Step3DataPreparation
from ui.crisp_steps.step_4_modeling import Step4Modeling
from ui.crisp_steps.step_5_evaluation import Step5Evaluation
from ui.crisp_steps.step_6_deployment import Step6Deployment


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ModelForge — Автоматизированный ML-анализ")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)

        # ✅ Устанавливаем стеклянный фон только для этого окна, если нужно
        # self.setAttribute(Qt.WA_TranslucentBackground, True)  # ❌ Убрали — вызывает CSS-предупреждения

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Отступы от краёв
        main_layout.setSpacing(10)

        # Заголовок
        title_label = QLabel("📊 ModelForge — Автоматизированный ML-анализ")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet("""
            color: #f0f0f0;
            padding: 20px;
            background-color: rgba(50, 50, 60, 180);
            border-radius: 12px;
            margin: 0;
        """)
        main_layout.addWidget(title_label)

        # Горизонтальный контейнер для навигации и контента
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Боковая панель навигации
        nav_frame = QFrame()
        nav_frame.setFixedWidth(310)
        nav_frame.setStyleSheet("""
            background-color: rgba(40, 40, 50, 180);
            border-right: 1px solid rgba(100, 100, 120, 0.5);
            border-radius: 0 12px 12px 0;
        """)
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(15, 20, 15, 20)

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

        for step_name, step_class in steps:
            btn = QPushButton(step_name)
            btn.setFixedHeight(40)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, name=step_name: self.on_nav_click(name))
            nav_layout.addWidget(btn)
            self.nav_buttons[step_name] = btn

        content_layout.addWidget(nav_frame)

        # Стек для содержимого шагов
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            background: transparent;
            border-radius: 12px;
        """)

        # Создаем экземпляры шагов
        self.steps = {}
        for step_name, step_class in steps:
            step_instance = step_class()
            self.steps[step_name] = step_instance
            self.stacked_widget.addWidget(step_instance)

        content_layout.addWidget(self.stacked_widget)

        # Устанавливаем первый шаг как активный
        self.current_step = 0
        self.stacked_widget.setCurrentIndex(self.current_step)
        self.nav_buttons["1. Business Understanding"].setChecked(True)

        # Добавляем кнопку "Сохранить отчет" внизу
        save_btn = QPushButton("💾 Сохранить отчет (PDF/HTML)")
        save_btn.setFixedHeight(40)
        save_btn.setStyleSheet("""
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            margin: 10px;
        """)
        save_btn.clicked.connect(self.on_save_report)
        main_layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def on_nav_click(self, step_name):
        """Обработчик клика по кнопке навигации."""
        # Отключаем все кнопки
        for btn in self.nav_buttons.values():
            btn.setChecked(False)

        # Включаем текущую
        self.nav_buttons[step_name].setChecked(True)

        # Получаем индекс шага
        step_index = list(self.steps.keys()).index(step_name)

        # Проверяем, можно ли перейти на этот шаг
        if step_index > self.current_step:
            # Пользователь пытается перейти вперёд — проверяем, выполнен ли предыдущий шаг
            if step_index == self.current_step + 1:
                self.current_step = step_index
                self.stacked_widget.setCurrentIndex(step_index)
            else:
                # Попытка пропустить шаг — игнорируем
                self.nav_buttons[list(self.steps.keys())[self.current_step]].setChecked(True)
                return
        else:
            # Переход назад — разрешён всегда
            self.current_step = step_index
            self.stacked_widget.setCurrentIndex(step_index)

    def on_save_report(self):
        """Открывает диалог сохранения отчёта."""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Сохранение отчёта",
            "Функция сохранения отчёта будет реализована на этапе 6.\n"
            "Пока вы можете сохранить отчёт через файловый диалог.\n"
            "Выберите папку для сохранения HTML и PDF файлов.",
            QMessageBox.Ok
        )