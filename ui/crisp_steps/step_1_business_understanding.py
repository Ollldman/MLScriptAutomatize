from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QFrame, QHBoxLayout
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from ModelForge.modules.report.report_data import ReportData


class Step1BusinessUnderstanding(QWidget):
    # Signal to notify parent that step is complete (optional)
    step_completed = pyqtSignal()

    def __init__(self, shared_data: ReportData):
        super().__init__()
        self.shared_data = shared_data  # Единый объект данных
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("1. Business Understanding")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #f0f0f0; padding: 10px;")
        layout.addWidget(title)

        # Описание
        desc = QLabel("Define the goal of the ML task and success metrics.")
        desc.setStyleSheet("color: #c0c0d0; font-size: 12px; padding: 5px;")
        layout.addWidget(desc)

        # Поле для цели
        goal_label = QLabel("Goal:")
        goal_label.setStyleSheet("color: #e0e0e0; padding: 5px;")
        layout.addWidget(goal_label)

        self.goal_text = QTextEdit()
        self.goal_text.setPlaceholderText("Describe the main objective of this experiment...")
        self.goal_text.setMaximumHeight(100)
        self.goal_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(50, 50, 60, 180);
                border: 1px solid rgba(100, 100, 120, 0.5);
                border-radius: 8px;
                padding: 10px;
                color: #f0f0f0;
                font-size: 13px;
            }
            QTextEdit:focus {
                border-color: #6a6acc;
            }
        """)
        layout.addWidget(self.goal_text)

        # Поле для метрики успеха
        metric_label = QLabel("Success Metric:")
        metric_label.setStyleSheet("color: #e0e0e0; padding: 5px;")
        layout.addWidget(metric_label)

        self.metric_text = QTextEdit()
        self.metric_text.setPlaceholderText("e.g. Accuracy > 0.9, F1-score > 0.85, etc.")
        self.metric_text.setMaximumHeight(100)
        self.metric_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(50, 50, 60, 180);
                border: 1px solid rgba(100, 100, 120, 0.5);
                border-radius: 8px;
                padding: 10px;
                color: #f0f0f0;
                font-size: 13px;
            }
            QTextEdit:focus {
                border-color: #6a6acc;
            }
        """)
        layout.addWidget(self.metric_text)

        # Кнопки
        button_layout = QHBoxLayout()

        save_btn = QPushButton("✅ Save")
        save_btn.setFixedHeight(40)
        save_btn.clicked.connect(self.on_save)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(46, 204, 113, 0.8);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(39, 174, 96, 0.9);
            }
        """)
        button_layout.addWidget(save_btn)

        reset_btn = QPushButton("🗑️ Reset")
        reset_btn.setFixedHeight(40)
        reset_btn.clicked.connect(self.on_reset)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(231, 76, 60, 0.8);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(192, 57, 43, 0.9);
            }
        """)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()  # Отодвигаем кнопки влево

        layout.addLayout(button_layout)

        # Заполняем поля, если данные уже есть
        self.load_existing_data()

        self.setLayout(layout)

    def load_existing_data(self):
        """Загружает существующие данные из shared_data в поля."""
        if self.shared_data.business_understanding:
            self.goal_text.setPlainText(self.shared_data.business_understanding.get("goal", ""))
            self.metric_text.setPlainText(self.shared_data.business_understanding.get("success_metric", ""))

    def on_save(self):
        """Сохраняет данные из полей в shared_data."""
        goal = self.goal_text.toPlainText().strip()
        metric = self.metric_text.toPlainText().strip()

        # Обновляем shared_data
        if not self.shared_data.business_understanding:
            self.shared_data.business_understanding = {}
        self.shared_data.business_understanding["goal"] = goal
        self.shared_data.business_understanding["success_metric"] = metric

        print(f"Saved Business Understanding: Goal='{goal}', Success Metric='{metric}'")

    def on_reset(self):
        """Очищает поля и удаляет данные из shared_data."""
        self.goal_text.clear()
        self.metric_text.clear()

        if self.shared_data.business_understanding:
            self.shared_data.business_understanding = {}