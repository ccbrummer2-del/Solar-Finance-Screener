"""
Solar Terminal Launcher
Choose between FX Screener or Stock Screener
"""
import sys
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
import os

class LauncherDialog(QDialog):
    """Launcher dialog to choose screener type"""
    
    def __init__(self):
        super().__init__()
        self.choice = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Solar Terminal - Select Screener")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🌞 Solar Terminal")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Select Market Type")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # FX Button
        fx_btn = QPushButton("💱 FX & Commodities Screener")
        fx_btn.setObjectName("fxButton")
        fx_btn.setMinimumHeight(80)
        fx_btn_font = QFont()
        fx_btn_font.setPointSize(14)
        fx_btn.setFont(fx_btn_font)
        fx_btn.clicked.connect(lambda: self.select_screener('fx'))
        layout.addWidget(fx_btn)
        
        # Stock Button
        stock_btn = QPushButton("📈 US 500 Stocks Screener")
        stock_btn.setObjectName("stockButton")
        stock_btn.setMinimumHeight(80)
        stock_btn_font = QFont()
        stock_btn_font.setPointSize(14)
        stock_btn.setFont(stock_btn_font)
        stock_btn.clicked.connect(lambda: self.select_screener('stocks'))
        layout.addWidget(stock_btn)
        
        layout.addStretch()
        
        # Info label
        info = QLabel("FX: 5m, 15m, 4h, 1D, 1W updates\nStocks: 1h, 1D, 1W updates (optimized)")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                padding: 20px;
                background-color: #333;
                border: 2px solid #555;
                border-radius: 8px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444;
                border: 2px solid #4CAF50;
            }
            QPushButton#fxButton {
                background-color: #1a237e;
            }
            QPushButton#fxButton:hover {
                background-color: #283593;
            }
            QPushButton#stockButton {
                background-color: #1b5e20;
            }
            QPushButton#stockButton:hover {
                background-color: #2e7d32;
            }
        """)
    
    def select_screener(self, choice):
        """User selected a screener type"""
        self.choice = choice
        self.accept()
    
    def get_choice(self):
        """Return user's choice"""
        return self.choice


def main():
    """Run the launcher"""
    app = QApplication(sys.argv)
    app.setApplicationName("Solar Terminal")
    app.setStyle('Fusion')
    
    # Show launcher
    launcher = LauncherDialog()
    if launcher.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    
    choice = launcher.get_choice()
    
    if choice == 'fx':
        # Launch FX screener
        from ui.main_window import MainWindow
        window = MainWindow(screener_type='fx')
        window.show()
    elif choice == 'stocks':
        # Launch Stock screener
        from ui.stock_window import StockWindow
        window = StockWindow()
        window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
