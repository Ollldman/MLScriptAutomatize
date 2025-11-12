import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class Step6Deployment(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        label = QLabel("Step 5: Deployment")
        btn_next = QPushButton("Next →")
        # btn_next.clicked.connect(self.go_to_step_2)

        layout.addWidget(label)
        layout.addWidget(btn_next)
        self.setLayout(layout)
