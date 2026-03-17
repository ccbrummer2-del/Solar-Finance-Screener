"""
Stock Screener Window - US 500 Stocks
Simplified version with 1h, 1D, 1W timeframes only
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QMessageBox, QHeaderView, QTextEdit, 
                             QFileDialog, QDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
import sys
import os
from core.api_client import APIClient
from ui.api_key_dialog import APIKeyDialog
from datetime import datetime

class StockWindow(QMainWindow):
    """Stock screener window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize API client
        self.api_client = APIClient()
        
        # Check if API key is configured
        if not self.api_client.has_api_key():
            self.show_api_key_dialog()
        
        self.results = []
        
        self.init_ui()
        self.load_stylesheet()
        
        # Auto-refresh every hour
        self.setup_auto_refresh()
    
    def show_api_key_dialog(self):
        """Show API key entry dialog on first launch"""
        dialog = APIKeyDialog(self.api_client, self)
        result = dialog.exec()
        
        if result != QDialog.DialogCode.Accepted:
            reply = QMessageBox.question(
                self,
                "Authentication Required",
                "Solar Terminal requires an API key to access live data.\n\n"
                "Exit application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                sys.exit(0)
            else:
                self.show_api_key_dialog()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Solar Terminal - Stock Screener (US 500)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar
        top_bar = self.create_top_bar()
        main_layout.addLayout(top_bar)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            'Symbol', 'Name', 'Price', '24h %', 'Signal', 
            '1h', '1D', '1W', 'Sentiment', 'Strength'
        ])
        
        # Table settings
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Symbol
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # Price
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # 24h %
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # Signal
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)  # 1h
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)  # 1D
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)  # 1W
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Fixed)  # Sentiment
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Fixed)  # Strength
        
        self.table.setColumnWidth(0, 80)   # Symbol
        self.table.setColumnWidth(2, 80)   # Price
        self.table.setColumnWidth(3, 70)   # 24h %
        self.table.setColumnWidth(4, 120)  # Signal
        self.table.setColumnWidth(5, 100)  # 1h
        self.table.setColumnWidth(6, 100)  # 1D
        self.table.setColumnWidth(7, 100)  # 1W
        self.table.setColumnWidth(8, 90)   # Sentiment
        self.table.setColumnWidth(9, 70)   # Strength
        
        main_layout.addWidget(self.table)
        
        # Status bar
        self.timestamp_label = QLabel("No data loaded")
        self.statusBar().addPermanentWidget(self.timestamp_label)
        self.statusBar().showMessage("Ready")
    
    def create_top_bar(self):
        """Create top bar with buttons"""
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("📈 Solar Terminal - Stock Screener")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        # Refresh button
        self.refresh_btn = QPushButton("Load Data")
        self.refresh_btn.setObjectName("refreshButton")
        self.refresh_btn.clicked.connect(self.load_from_server)
        layout.addWidget(self.refresh_btn)
        
        return layout
    
    def setup_auto_refresh(self):
        """Setup automatic refresh every hour"""
        # Auto-refresh timer (every hour)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.refresh_timer.start(3600000)  # 1 hour in milliseconds
    
    def auto_refresh(self):
        """Auto refresh data"""
        print("🔄 Auto-refreshing stock data...")
        self.load_from_server()
    
    def load_from_server(self):
        """Load latest stock results from API server"""
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Loading...")
        self.statusBar().showMessage("Fetching stock data from server...")
        
        # Fetch from API - use STOCK endpoint
        data, error = self.api_client.get_latest_stock_results()
        
        if error:
            QMessageBox.critical(
                self,
                "Server Error",
                f"Could not load stock data:\n\n{error}\n\n"
                "Make sure the stock scanner service is running."
            )
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setText("Load Data")
            self.statusBar().showMessage(f"Error: {error}", 5000)
            return
        
        # Update results
        self.results = data['results']
        self.populate_table()
        
        # Update UI
        timestamp = data.get('timestamp', 'Unknown')
        self.timestamp_label.setText(f"Last updated: {timestamp}")
        
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("Load Data")
        self.statusBar().showMessage(
            f"✅ Loaded {len(self.results)} stocks from server",
            5000
        )
    
    def populate_table(self):
        """Populate table with stock results"""
        self.table.setRowCount(len(self.results))
        
        for row_idx, result in enumerate(self.results):
            # Symbol
            symbol_item = QTableWidgetItem(result.get('Pair', ''))
            symbol_item.setFont(QFont('Courier New', 10, QFont.Weight.Bold))
            self.table.setItem(row_idx, 0, symbol_item)
            
            # Name
            name_item = QTableWidgetItem(result.get('Name', ''))
            self.table.setItem(row_idx, 1, name_item)
            
            # Price
            price = result.get('Price', 0)
            price_item = QTableWidgetItem(f"${price:.2f}" if price else "-")
            self.table.setItem(row_idx, 2, price_item)
            
            # 24h Change
            change = result.get('Lookback1', 0)
            change_item = QTableWidgetItem(f"{change:+.2f}%" if change else "-")
            if change > 0:
                change_item.setForeground(QColor('#4CAF50'))
            elif change < 0:
                change_item.setForeground(QColor('#f44336'))
            self.table.setItem(row_idx, 3, change_item)
            
            # Signal
            signal = result.get('Signal', '')
            signal_item = QTableWidgetItem(signal)
            if 'LONG' in signal:
                signal_item.setForeground(QColor('#4CAF50'))
            elif 'SHORT' in signal:
                signal_item.setForeground(QColor('#f44336'))
            self.table.setItem(row_idx, 4, signal_item)
            
            # 1h state
            state_1h = result.get('1h', '-')
            self.table.setItem(row_idx, 5, self.create_state_item(state_1h))
            
            # 1D state
            state_1d = result.get('1D', '-')
            self.table.setItem(row_idx, 6, self.create_state_item(state_1d))
            
            # 1W state
            state_1w = result.get('1W', '-')
            self.table.setItem(row_idx, 7, self.create_state_item(state_1w))
            
            # Sentiment
            sentiment = result.get('Sentiment', '-')
            self.table.setItem(row_idx, 8, QTableWidgetItem(sentiment))
            
            # Strength
            strength = result.get('Strength', 0)
            self.table.setItem(row_idx, 9, QTableWidgetItem(str(strength)))
    
    def create_state_item(self, state):
        """Create colored table item for market state"""
        item = QTableWidgetItem(state)
        
        if state == 'accumulation':
            item.setForeground(QColor('#4CAF50'))
        elif state == 're-accumulation':
            item.setForeground(QColor('#8BC34A'))
        elif state == 'distribution':
            item.setForeground(QColor('#f44336'))
        elif state == 're-distribution':
            item.setForeground(QColor('#FF9800'))
        
        return item
    
    def export_results(self):
        """Export results to CSV"""
        if not self.results:
            QMessageBox.warning(self, "No Data", "No results to export. Load data first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Stock Results",
            f"stock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='') as f:
                    if self.results:
                        keys = self.results[0].keys()
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(self.results)
                
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
    
    def load_stylesheet(self):
        """Load dark theme stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial;
                font-size: 10pt;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton#refreshButton {
                background-color: #4CAF50;
            }
            QPushButton#refreshButton:hover {
                background-color: #45a049;
            }
            QTableWidget {
                background-color: #2a2a2a;
                gridline-color: #444;
                border: 1px solid #444;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
            }
            QHeaderView::section {
                background-color: #333;
                color: white;
                padding: 8px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QStatusBar {
                background-color: #2a2a2a;
                color: #e0e0e0;
            }
        """)
