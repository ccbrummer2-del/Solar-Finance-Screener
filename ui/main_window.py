"""
Main application window - Hybrid version with customizable timeframes
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QProgressBar, QMessageBox, QListWidget,
                             QAbstractItemView, QHeaderView, QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
import sys
import os
from ui.settings_dialog import SettingsDialog
from core.market_analyzer import MarketAnalyzer, FX_PAIRS
from core.data_fetcher import DataFetcher
from core.data_manager import DataManager
from datetime import datetime

class ScanWorker(QThread):
    """Background thread for scanning markets"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    
    def __init__(self, pairs, settings):
        super().__init__()
        self.pairs = pairs
        self.settings = settings
        self.analyzer = MarketAnalyzer()
        self.data_fetcher = DataFetcher()
    
    def run(self):
        results = []
        total = len(self.pairs)
        
        for idx, (pair_name, symbol) in enumerate(self.pairs):
            self.progress.emit(int((idx / total) * 100), f"Analyzing {pair_name}...")
            
            try:
                result = self.analyzer.analyze_pair(pair_name, symbol, self.data_fetcher, self.settings)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {pair_name}: {e}")
        
        self.finished.emit(results)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Load saved settings or use defaults
        saved_settings = self.data_manager.load_settings()
        if saved_settings:
            self.settings = saved_settings
        else:
            # Default settings for hybrid system
            self.settings = {
                # Timeframes
                'timeframes': ['5m', '15m', '4h', '1D', '1W'],
                
                # Sentiment
                'sentiment_enabled': True,
                'sentiment_timeframe': '1D',
                
                # Lookbacks
                'lookbacks': [
                    {'enabled': True, 'timeframe': '1D', 'periods': 30, 'label': 'Monthly'},
                    {'enabled': True, 'timeframe': '1D', 'periods': 7, 'label': 'Weekly'},
                    {'enabled': False, 'timeframe': '4h', 'periods': 20, 'label': 'Custom'}
                ],
                
                # Signal thresholds
                'signal_thresholds': {
                    'bullish': 5,
                    'bearish': 5
                },
                
                # Legacy/Recommendations
                'sort_by': ['Largest Mover'],
                'recommend_types': ['Full Bull', 'Full Bear'],
                'recommend_percent': 5.0,
                'recommend_sentiment': 70
            }
        
        self.selected_pairs = []
        self.results = []
        
        self.init_ui()
        self.load_stylesheet()
    
    def _set_app_icon(self):
        """Set the application icon"""
        # Handle PyInstaller bundled path
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_dir = sys._MEIPASS
        else:
            # Running as script
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        possible_paths = [
            os.path.join(base_dir, 'assets', 'solar_icon.ico'),
            os.path.join(base_dir, 'assets', 'icon.ico'),
            os.path.join(base_dir, 'assets', 'solar_icon.png'),
            os.path.join(base_dir, 'assets', 'icon.png'),
            os.path.join('assets', 'solar_icon.ico'),
            os.path.join('assets', 'icon.ico'),
            'solar_icon.ico',
            'icon.ico'
        ]
        
        for icon_path in possible_paths:
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print(f"✓ Window icon loaded: {icon_path}")
                return
        
        print(f"⚠ Warning: No icon file found in window. Base dir: {base_dir}")
        print(f"⚠ Frozen state: {getattr(sys, 'frozen', False)}")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Solar Terminal")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application icon
        self._set_app_icon()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar with buttons
        top_bar = self.create_top_bar()
        main_layout.addLayout(top_bar)
        
        # Content area (split into sidebar and main)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        
        # Sidebar for market selection
        sidebar = self.create_sidebar()
        content_layout.addWidget(sidebar, 1)
        
        # Main content area
        main_content = self.create_main_content()
        content_layout.addWidget(main_content, 4)
        
        main_layout.addLayout(content_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select markets and click Scan")
    
    def create_top_bar(self):
        """Create top navigation bar"""
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Solar Terminal")
        title.setObjectName("title")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Buttons
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-sort and update recommendations based on current settings")
        self.refresh_btn.clicked.connect(self.refresh_display)
        self.refresh_btn.setEnabled(False)  # Disabled until first scan
        layout.addWidget(self.refresh_btn)
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(settings_btn)
        
        self.scan_btn = QPushButton("Scan Markets")
        self.scan_btn.setObjectName("scanButton")
        self.scan_btn.clicked.connect(self.scan_markets)
        layout.addWidget(self.scan_btn)
        
        return layout
    
    def create_sidebar(self):
        """Create sidebar for market selection with category filters"""
        from core.market_analyzer import MARKET_CATEGORIES
        
        widget = QWidget()
        widget.setObjectName("sidebar")
        widget.setMaximumWidth(300)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Select Markets")
        title.setObjectName("sidebarTitle")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Category Filter Buttons
        filter_label = QLabel("Filter by Category:")
        filter_label_font = QFont()
        filter_label_font.setBold(True)
        filter_label.setFont(filter_label_font)
        layout.addWidget(filter_label)
        
        # Create filter buttons
        self.filter_buttons = {}
        for category in MARKET_CATEGORIES.keys():
            btn = QPushButton(f"✓ {category}")
            btn.setCheckable(True)
            btn.setChecked(True)  # All enabled by default
            btn.clicked.connect(lambda checked, cat=category: self.filter_by_category(cat, checked))
            self.filter_buttons[category] = btn
            layout.addWidget(btn)
        
        # Select All / Deselect All
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(5)
        
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all_markets)
        btn_layout.addWidget(select_all)
        
        deselect_all = QPushButton("Deselect All")
        deselect_all.clicked.connect(self.deselect_all_markets)
        btn_layout.addWidget(deselect_all)
        
        layout.addLayout(btn_layout)
        
        # Market list
        self.market_list = QListWidget()
        self.market_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.populate_market_list()
        layout.addWidget(self.market_list)
        
        # Count label
        self.selected_count_label = QLabel("0 markets selected")
        self.selected_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.selected_count_label)
        
        # Connect selection changed
        self.market_list.itemSelectionChanged.connect(self.update_selected_count)
        
        return widget
    
    def populate_market_list(self):
        """Populate market list based on active category filters"""
        from core.market_analyzer import MARKET_CATEGORIES
        
        self.market_list.clear()
        
        # Get active categories
        active_categories = [
            cat for cat, btn in self.filter_buttons.items() 
            if btn.isChecked()
        ]
        
        # Collect all markets from active categories
        visible_markets = []
        for category in active_categories:
            visible_markets.extend(MARKET_CATEGORIES[category])
        
        # Remove duplicates and sort
        visible_markets = sorted(set(visible_markets))
        
        # Add to list
        self.market_list.addItems(visible_markets)
        
        # Update count
        self.update_selected_count()
    
    def filter_by_category(self, category, checked):
        """Handle category filter toggle"""
        # Update button text
        btn = self.filter_buttons[category]
        if checked:
            btn.setText(f"✓ {category}")
        else:
            btn.setText(f"  {category}")
        
        # Repopulate list
        self.populate_market_list()
    
    def create_main_content(self):
        """Create main content area"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Summary metrics
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)
        
        self.metric_longs = QLabel("Perfect Longs: 0")
        self.metric_longs.setObjectName("metric")
        metrics_layout.addWidget(self.metric_longs)
        
        self.metric_shorts = QLabel("Perfect Shorts: 0")
        self.metric_shorts.setObjectName("metric")
        metrics_layout.addWidget(self.metric_shorts)
        
        self.metric_watch = QLabel("Watch List: 0")
        self.metric_watch.setObjectName("metric")
        metrics_layout.addWidget(self.metric_watch)
        
        self.metric_mixed = QLabel("Mixed: 0")
        self.metric_mixed.setObjectName("metric")
        metrics_layout.addWidget(self.metric_mixed)
        
        metrics_layout.addStretch()
        
        layout.addLayout(metrics_layout)
        
        # Results table (will be configured dynamically)
        self.table = QTableWidget()
        self.setup_table_columns()
        
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        layout.addWidget(self.table)
        
        # Last updated timestamp
        self.timestamp_label = QLabel("")
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timestamp_label)
        
        # Recommendations area
        recommendations_widget = QWidget()
        recommendations_widget.setObjectName("recommendationsWidget")
        recommendations_widget.setMaximumHeight(220)
        rec_layout = QVBoxLayout(recommendations_widget)
        rec_layout.setContentsMargins(5, 5, 5, 5)
        rec_layout.setSpacing(2)
        
        rec_title = QLabel("Trade Recommendations")
        rec_title_font = QFont()
        rec_title_font.setPointSize(10)
        rec_title_font.setBold(True)
        rec_title.setFont(rec_title_font)
        rec_layout.addWidget(rec_title)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(140)
        self.recommendations_text.setHtml("<p>Trade recommendations will appear here after scanning</p>")
        rec_layout.addWidget(self.recommendations_text)
        
        layout.addWidget(recommendations_widget)
        
        return widget
    
    def setup_table_columns(self):
        """Setup table columns based on current settings"""
        headers = ['Pair', 'Signal']
        
        # Add timeframe columns
        selected_tfs = self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W'])
        headers.extend(selected_tfs)
        
        # Add lookback columns if enabled
        lookbacks = self.settings.get('lookbacks', [])
        for idx, lb in enumerate(lookbacks):
            if lb.get('enabled', False):
                label = lb.get('label', f'LB{idx+1}')
                headers.append(label)
        
        # Add sentiment if enabled
        if self.settings.get('sentiment_enabled', True):
            headers.append('Sentiment')
        
        # Add notes
        headers.append('Notes')
        
        # Configure table
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set column resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Pair
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Signal
        
        # Timeframe columns
        for i in range(2, 2 + len(selected_tfs)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        # Lookback columns
        num_lookbacks = sum(1 for lb in lookbacks if lb.get('enabled', False))
        for i in range(2 + len(selected_tfs), 2 + len(selected_tfs) + num_lookbacks):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        # Sentiment and Notes
        if self.settings.get('sentiment_enabled', True):
            header.setSectionResizeMode(len(headers) - 2, QHeaderView.ResizeMode.ResizeToContents)  # Sentiment
        header.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)  # Notes
    
    def update_selected_count(self):
        """Update the selected markets count"""
        count = len(self.market_list.selectedItems())
        self.selected_count_label.setText(f"{count} market{'s' if count != 1 else ''} selected")
    
    def select_all_markets(self):
        """Select all markets"""
        for i in range(self.market_list.count()):
            self.market_list.item(i).setSelected(True)
    
    def deselect_all_markets(self):
        """Deselect all markets"""
        self.market_list.clearSelection()
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.get_settings()
            
            # Save settings to disk
            self.data_manager.save_settings(self.settings)
            
            self.statusBar().showMessage("Settings saved successfully", 3000)
            
            # Reconfigure table columns based on new settings
            self.setup_table_columns()
            
            # Refresh display if we have results
            if self.results:
                self.refresh_display()
    
    def refresh_display(self):
        """Refresh the display based on current settings without re-scanning"""
        if not self.results:
            return
        
        # Re-sort results based on new settings
        self.sort_results()
        
        # Update table display
        self.populate_table()
        
        # Update recommendations based on new criteria
        self.update_recommendations()
        
        # Update metrics
        self.update_metrics()
        
        self.statusBar().showMessage("Display refreshed with current settings", 3000)
    
    def scan_markets(self):
        """Start market scanning"""
        # Get selected markets
        selected_items = self.market_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "⚠️ Please select at least one market to scan")
            return
        
        self.selected_pairs = [(item.text(), FX_PAIRS[item.text()]) for item in selected_items]
        
        # Disable scan button and show progress
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Starting scan...")
        
        # Start background scan
        self.scan_worker = ScanWorker(self.selected_pairs, self.settings)
        self.scan_worker.progress.connect(self.update_progress)
        self.scan_worker.finished.connect(self.scan_finished)
        self.scan_worker.start()
    
    def update_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
        self.statusBar().showMessage(message)
    
    def scan_finished(self, results):
        """Handle scan completion"""
        self.results = results
        self.sort_results()
        self.populate_table()
        self.update_recommendations()
        self.update_metrics()
        
        # Enable refresh button now that we have results
        self.refresh_btn.setEnabled(True)
        
        # Update timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tfs_str = ', '.join(self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W']))
        self.timestamp_label.setText(
            f"Last updated: {timestamp} | Timeframes: {tfs_str}"
        )
        
        # Reset UI
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan Markets")
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"✅ Scan complete - {len(results)} pairs analyzed", 5000)
    
    def sort_results(self):
        """Sort results based on settings"""
        if not self.results:
            return
        
        sort_by = self.settings.get('sort_by', ['Largest Mover'])
        
        # Helper function to get absolute change value from lookbacks
        def get_change(r):
            # Try first enabled lookback
            for i in range(1, 4):
                lb_key = f'Lookback{i}'
                if lb_key in r and r[lb_key] is not None:
                    return r[lb_key]
            return 0
        
        def get_abs_change(r):
            return abs(get_change(r))
        
        has_bullish = 'Fully Bullish' in sort_by
        has_bearish = 'Fully Bearish' in sort_by
        has_largest = 'Largest Mover' in sort_by
        
        # Get total timeframes for determining "full" signals
        total_tfs = len(self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W']))
        
        # If both Fully Bullish and Fully Bearish are selected
        if has_bullish and has_bearish:
            long_signals = []
            short_signals = []
            others = []
            
            for r in self.results:
                if r['Strength'] == total_tfs:  # Full bullish
                    long_signals.append(r)
                elif r['Strength'] == -total_tfs:  # Full bearish
                    short_signals.append(r)
                else:
                    others.append(r)
            
            long_signals.sort(key=get_change, reverse=True)
            short_signals.sort(key=get_change)
            others.sort(key=get_abs_change, reverse=True)
            
            self.results = long_signals + short_signals + others
        
        elif has_bullish and has_largest:
            self.results.sort(key=lambda r: (
                1 if r['Strength'] == total_tfs else 0,
                get_abs_change(r)
            ), reverse=True)
        
        elif has_bearish and has_largest:
            self.results.sort(key=lambda r: (
                1 if r['Strength'] == -total_tfs else 0,
                get_abs_change(r)
            ), reverse=True)
        
        elif has_bullish:
            self.results.sort(key=lambda r: 1 if r['Strength'] == total_tfs else 0, reverse=True)
        
        elif has_bearish:
            self.results.sort(key=lambda r: 1 if r['Strength'] == -total_tfs else 0, reverse=True)
        
        else:
            # Default: sort by largest mover
            self.results.sort(key=get_abs_change, reverse=True)
    
    def update_metrics(self):
        """Update summary metrics"""
        total_tfs = len(self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W']))
        
        perfect_longs = len([r for r in self.results if r['Strength'] == total_tfs])
        perfect_shorts = len([r for r in self.results if r['Strength'] == -total_tfs])
        watch_list = len([r for r in self.results if abs(r['Strength']) == total_tfs - 1])
        mixed = len([r for r in self.results if abs(r['Strength']) < total_tfs - 1])
        
        self.metric_longs.setText(f"Perfect Longs: {perfect_longs}")
        self.metric_shorts.setText(f"Perfect Shorts: {perfect_shorts}")
        self.metric_watch.setText(f"Watch List: {watch_list}")
        self.metric_mixed.setText(f"Mixed: {mixed}")
    
    def populate_table(self):
        """Populate results table"""
        self.table.setRowCount(len(self.results))
        
        selected_tfs = self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W'])
        lookbacks = self.settings.get('lookbacks', [])
        sentiment_enabled = self.settings.get('sentiment_enabled', True)
        
        for row, result in enumerate(self.results):
            col = 0
            
            # Pair
            pair_item = QTableWidgetItem(result['Pair'])
            pair_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self.table.setItem(row, col, pair_item)
            col += 1
            
            # Signal
            signal_item = QTableWidgetItem(result['Signal'])
            if '🟢' in result['Signal']:
                signal_item.setBackground(QColor(76, 175, 80, 50))
            elif '🔴' in result['Signal']:
                signal_item.setBackground(QColor(244, 67, 54, 50))
            elif '🟡' in result['Signal']:
                signal_item.setBackground(QColor(255, 193, 7, 50))
            elif '🟠' in result['Signal']:
                signal_item.setBackground(QColor(255, 152, 0, 50))
            self.table.setItem(row, col, signal_item)
            col += 1
            
            # Timeframe states
            for tf in selected_tfs:
                state = result.get(tf, '-')
                self.table.setItem(row, col, QTableWidgetItem(self.format_state(state)))
                col += 1
            
            # Lookback periods
            for idx, lb in enumerate(lookbacks):
                if not lb.get('enabled', False):
                    continue
                
                lb_value = result.get(f'Lookback{idx+1}')
                if lb_value is not None:
                    lb_text = f"{'🟢' if lb_value > 0 else '🔴'} {lb_value:+.2f}%"
                    lb_item = QTableWidgetItem(lb_text)
                    if lb_value > 0:
                        lb_item.setForeground(QColor(76, 175, 80))
                    else:
                        lb_item.setForeground(QColor(244, 67, 54))
                    self.table.setItem(row, col, lb_item)
                else:
                    self.table.setItem(row, col, QTableWidgetItem("-"))
                col += 1
            
            # Sentiment
            if sentiment_enabled:
                sentiment_text = result.get('Sentiment_Text', '-')
                sentiment_val = result.get('Sentiment', '-')
                sent_display = f"{sentiment_text} ({sentiment_val})"
                sent_item = QTableWidgetItem(sent_display)
                
                if 'Bull' in sentiment_text:
                    sent_item.setForeground(QColor(76, 175, 80))
                elif 'Bear' in sentiment_text:
                    sent_item.setForeground(QColor(244, 67, 54))
                
                self.table.setItem(row, col, sent_item)
                col += 1
            
            # Notes placeholder
            self.table.setItem(row, col, QTableWidgetItem(""))
    
    def format_state(self, state):
        """Format market state for display"""
        if state == 'accumulation':
            return '🟩 ACC'
        elif state == 're-accumulation':
            return '🟢 R-ACC'
        elif state == 'distribution':
            return '🟥 DIS'
        elif state == 're-distribution':
            return '🔴 R-DIS'
        else:
            return '⚪ -'
    
    def update_recommendations(self):
        """Update recommendations based on results with sequential AND filtering"""
        recommend_types = self.settings.get('recommend_types', ['Full Bull', 'Full Bear'])
        recommend_percent = self.settings.get('recommend_percent', 5.0)
        recommend_sentiment = self.settings.get('recommend_sentiment', 70)
        
        total_tfs = len(self.settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W']))
        
        # Start with all results
        recommended = list(self.results)
        
        # Track filter stages for debug message
        filter_stages = []
        initial_count = len(recommended)
        
        # FILTER 1: % Change WITH DIRECTION ALIGNMENT (PRIMARY FILTER)
        if 'Based on % Change' in recommend_types or 'Based on % Gained' in recommend_types:
            before = len(recommended)
            new_recommended = []
            
            for r in recommended:
                # Get the signal direction
                strength = r.get('Strength', 0)
                is_bullish = strength > 0
                is_bearish = strength < 0
                
                # Check if ANY lookback meets the criteria AND matches direction
                meets_criteria = False
                for i in range(1, 4):
                    lookback_value = r.get(f'Lookback{i}', 0) or 0
                    
                    # Check if lookback is significant enough
                    if abs(lookback_value) >= recommend_percent:
                        # Check if direction matches signal
                        if is_bullish and lookback_value > 0:
                            # Bullish signal + positive movement = match
                            meets_criteria = True
                            break
                        elif is_bearish and lookback_value < 0:
                            # Bearish signal + negative movement = match
                            meets_criteria = True
                            break
                
                if meets_criteria:
                    new_recommended.append(r)
            
            recommended = new_recommended
            after = len(recommended)
            filter_stages.append(f"Minimum {recommend_percent}% change (direction-aligned): {before} → {after} markets")
        
        # FILTER 2: Sentiment (SECONDARY FILTER)
        if 'Based on Sentiment' in recommend_types:
            before = len(recommended)
            min_sent = recommend_sentiment
            max_sent = 100 - recommend_sentiment
            recommended = [
                r for r in recommended
                if r.get('Sentiment_Value', 0) >= min_sent 
                or r.get('Sentiment_Value', 0) <= max_sent
            ]
            after = len(recommended)
            filter_stages.append(f"Sentiment ≥{min_sent}% or ≤{max_sent}%: {before} → {after} markets")
        
        # FILTER 3: Signal Type (FINAL FILTER)
        has_bull = 'Full Bull' in recommend_types
        has_bear = 'Full Bear' in recommend_types
        
        if has_bull and has_bear:
            # Both selected: show full bulls OR full bears (that passed previous filters)
            before = len(recommended)
            recommended = [r for r in recommended if abs(r['Strength']) == total_tfs]
            after = len(recommended)
            filter_stages.append(f"Full Bull or Full Bear ({total_tfs}/{total_tfs}): {before} → {after} markets")
        elif has_bull:
            # Only bulls (that passed previous filters)
            before = len(recommended)
            recommended = [r for r in recommended if r['Strength'] == total_tfs]
            after = len(recommended)
            filter_stages.append(f"Full Bull ({total_tfs}/{total_tfs}): {before} → {after} markets")
        elif has_bear:
            # Only bears (that passed previous filters)
            before = len(recommended)
            recommended = [r for r in recommended if r['Strength'] == -total_tfs]
            after = len(recommended)
            filter_stages.append(f"Full Bear ({total_tfs}/{total_tfs}): {before} → {after} markets")
        
        # Build HTML for recommendations
        if recommended:
            html = f"<h3 style='color: #4CAF50;'>✅ {len(recommended)} RECOMMENDED SETUP(S)</h3>"
            html += f"<p><i>Filters applied (AND logic): {', '.join(recommend_types)}</i></p>"
            
            # Show filter breakdown if multiple filters
            if len(filter_stages) > 1:
                html += "<p style='font-size: 10px; color: #888;'>"
                html += f"Started with {initial_count} markets → "
                html += " → ".join([stage.split(": ")[1] for stage in filter_stages])
                html += "</p>"
            
            html += "<ul>"
            
            for r in recommended[:10]:  # Limit to 10
                pair = r['Pair']
                signal = r['Signal']
                
                # Build detail line
                details = []
                
                # Add % change if it was a filter
                if 'Based on % Change' in recommend_types or 'Based on % Gained' in recommend_types:
                    for i in range(1, 4):
                        lb_value = r.get(f'Lookback{i}')
                        if lb_value is not None:
                            lb_label = r.get(f'Lookback{i}_Label', f'LB{i}')
                            details.append(f"{lb_label}: {lb_value:+.2f}%")
                            break  # Just show first lookback
                
                # Add sentiment if it was a filter
                if 'Based on Sentiment' in recommend_types:
                    sent_val = r.get('Sentiment_Value', 0)
                    details.append(f"Sentiment: {sent_val}%")
                
                detail_str = " | ".join(details) if details else ""
                
                if r['Strength'] > 0:
                    html += f"<li><b>{pair}</b> - {signal}"
                    if detail_str:
                        html += f" | {detail_str}"
                    html += f" | <span style='color: #4CAF50;'>🟢 Look for LONG entries</span></li>"
                elif r['Strength'] < 0:
                    html += f"<li><b>{pair}</b> - {signal}"
                    if detail_str:
                        html += f" | {detail_str}"
                    html += f" | <span style='color: #f44336;'>🔴 Look for SHORT entries</span></li>"
                else:
                    html += f"<li><b>{pair}</b> - {signal}"
                    if detail_str:
                        html += f" | {detail_str}"
                    html += "</li>"
            
            html += "</ul>"
            
            if len(recommended) > 10:
                html += f"<p><i>...and {len(recommended) - 10} more</i></p>"
        else:
            html = "<h3 style='color: #f44336;'>⏳ No markets meet ALL selected criteria</h3>"
            html += f"<p><i>Applied filters: {', '.join(recommend_types)}</i></p>"
            
            # Show what failed
            if filter_stages:
                html += "<p style='font-size: 11px; color: #888;'><b>Filter breakdown:</b><br>"
                for stage in filter_stages:
                    html += f"• {stage}<br>"
                html += "</p>"
                html += "<p style='font-size: 11px;'>💡 Tip: Try lowering the % threshold or adjusting other criteria</p>"
            else:
                html += "<p>No recommendation criteria selected.</p>"
            
            # Show watch list (one TF away from full)
            watch_list = [r for r in self.results if abs(r['Strength']) == total_tfs - 1]
            if watch_list:
                html += f"<hr><h4>📋 Watch List ({total_tfs-1}/{total_tfs} alignment):</h4><ul>"
                for r in watch_list[:5]:
                    pair = r['Pair']
                    signal = r['Signal']
                    direction = "bullish" if r['Strength'] > 0 else "bearish"
                    html += f"<li><b>{pair}</b> - {signal} - Watch for final timeframe to align {direction}</li>"
                html += "</ul>"
        
        self.recommendations_text.setHtml(html)
    
    def export_results(self):
        """Export current scan results to CSV"""
        if not self.results:
            QMessageBox.warning(self, "No Data", "No scan results to export. Please run a scan first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scan Results",
            f"solar_terminal_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if filename:
            if self.data_manager.export_scan_to_csv(self.results, filename):
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to:\n{filename}"
                )
                self.statusBar().showMessage(f"Exported {len(self.results)} results to CSV", 5000)
            else:
                QMessageBox.critical(self, "Export Failed", "Failed to export results to CSV")
    
    def load_stylesheet(self):
        """Load application stylesheet"""
        style = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        #title {
            color: #4CAF50;
        }
        #scanButton {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-weight: bold;
            font-size: 13px;
            border-radius: 5px;
        }
        #scanButton:hover {
            background-color: #45a049;
        }
        #scanButton:disabled {
            background-color: #666;
        }
        #sidebar {
            background-color: #252525;
            border-radius: 8px;
            padding: 15px;
        }
        #sidebarTitle {
            color: #4CAF50;
        }
        #metric {
            background-color: #2a2a2a;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        #recommendationsWidget {
            background-color: #252525;
            border-radius: 8px;
            padding: 5px;
        }
        QPushButton {
            padding: 8px 15px;
            background-color: #333;
            border: none;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #444;
        }
        QTableWidget {
            background-color: #252525;
            gridline-color: #333;
            color: #e0e0e0;
            border-radius: 5px;
            border: none;
        }
        QTableWidget::item {
            padding: 8px;
        }
        QTableWidget::item:alternate {
            background-color: #2a2a2a;
        }
        QHeaderView::section {
            background-color: #2d2d2d;
            color: #4CAF50;
            padding: 8px;
            border: none;
            font-weight: bold;
        }
        QListWidget {
            background-color: #2a2a2a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #e0e0e0;
            padding: 5px;
        }
        QListWidget::item {
            padding: 5px;
            border-radius: 3px;
        }
        QListWidget::item:selected {
            background-color: #4CAF50;
            color: white;
        }
        QListWidget::item:hover {
            background-color: #333;
        }
        QProgressBar {
            border: 2px solid #333;
            border-radius: 5px;
            text-align: center;
            background-color: #2a2a2a;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QTextEdit {
            background-color: #2a2a2a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #e0e0e0;
            padding: 5px;
        }
        QStatusBar {
            background-color: #252525;
            color: #e0e0e0;
        }
        """
        self.setStyleSheet(style)
