import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class Step2DataUnderstanding(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        label = QLabel("Step 2: Data Understanding")
        btn_next = QPushButton("Next →")
        btn_next.clicked.connect(self.go_to_step_3)

        layout.addWidget(label)
        layout.addWidget(btn_next)
        self.setLayout(layout)

    def go_to_step_3(self):
        # Заглушка: вызов переключения на следующий шаг
        print("Switching to Step 3...")