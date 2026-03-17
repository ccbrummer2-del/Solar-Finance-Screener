"""
API Key Setup Dialog
Shows on first launch or when API key needs to be entered
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class APIKeyDialog(QDialog):
    """Dialog for entering API key"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.api_key = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Solar Terminal - API Key Required")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔐 API Key Required")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setMaximumHeight(120)
        instructions.setHtml("""
            <p>Solar Terminal requires an API key to access live market data.</p>
            <p><b>If you received an API key:</b><br>
            Enter it below and click "Authenticate".</p>
            <p><b>If you don't have a key:</b><br>
            Contact the server administrator to request access.</p>
        """)
        layout.addWidget(instructions)
        
        # API Key input
        layout.addWidget(QLabel("Enter API Key:"))
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("sk_your_name_xxxxx")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.key_input)
        
        # Show/Hide key checkbox
        show_layout = QHBoxLayout()
        self.show_key_btn = QPushButton("Show Key")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self.toggle_key_visibility)
        show_layout.addWidget(self.show_key_btn)
        show_layout.addStretch()
        layout.addLayout(show_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.auth_btn = QPushButton("Authenticate")
        self.auth_btn.setObjectName("authButton")
        self.auth_btn.clicked.connect(self.authenticate)
        button_layout.addWidget(self.auth_btn)
        
        cancel_btn = QPushButton("Exit Application")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QLineEdit {
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
                font-size: 12pt;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                padding: 10px 20px;
                background-color: #333;
                border: none;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton#authButton {
                background-color: #4CAF50;
            }
            QPushButton#authButton:hover {
                background-color: #45a049;
            }
            QPushButton#authButton:disabled {
                background-color: #666;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                color: #e0e0e0;
                padding: 10px;
            }
        """)
    
    def toggle_key_visibility(self, checked):
        """Toggle visibility of API key"""
        if checked:
            self.key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("Hide Key")
        else:
            self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("Show Key")
    
    def authenticate(self):
        """Verify and save API key"""
        key = self.key_input.text().strip()
        
        if not key:
            self.status_label.setText("⚠️ Please enter an API key")
            self.status_label.setStyleSheet("color: #ff9800;")
            return
        
        # Disable button during verification
        self.auth_btn.setEnabled(False)
        self.auth_btn.setText("Verifying...")
        self.status_label.setText("🔄 Verifying API key...")
        self.status_label.setStyleSheet("color: #2196F3;")
        
        # Process events to update UI
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Verify key with server
        success, message = self.api_client.verify_api_key(key)
        
        if success:
            # Save key
            self.api_client.save_api_key(key)
            self.api_key = key
            
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: #4CAF50;")
            
            # Close dialog after short delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1000, self.accept)
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.auth_btn.setEnabled(True)
            self.auth_btn.setText("Authenticate")
    
    def get_api_key(self):
        """Get the entered API key"""
        return self.api_key
