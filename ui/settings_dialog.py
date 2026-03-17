"""
Settings dialog window - Hybrid customizable version
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QSpinBox, QPushButton, QRadioButton,
                             QButtonGroup, QDoubleSpinBox, QSlider, QGroupBox,
                             QCheckBox, QTabWidget, QWidget, QListWidget,
                             QLineEdit, QAbstractItemView)
from PyQt6.QtCore import Qt

class SettingsDialog(QDialog):
    """Settings configuration dialog with tabs"""
    
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings.copy()
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Screener Settings")
        self.setModal(True)
        self.setMinimumWidth(700)
        self.setMinimumHeight(700)
        
        layout = QVBoxLayout(self)
        
        # Create tabbed interface
        tabs = QTabWidget()
        
        # Tab 1: Timeframes & Signal Logic (combined)
        tabs.addTab(self.create_timeframes_and_signals_tab(), "Timeframes & Signals")
        
        # Tab 2: Sentiment
        tabs.addTab(self.create_sentiment_tab(), "Sentiment")
        
        # Tab 3: Lookback Periods
        tabs.addTab(self.create_lookbacks_tab(), "Lookback Periods")
        
        # Tab 4: Recommendations
        tabs.addTab(self.create_recommendations_tab(), "Recommendations")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("saveButton")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
                color: white;
            }
            QTabWidget::pane {
                border: 2px solid #444;
                border-radius: 5px;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #333;
                color: white;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 5px 5px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #4CAF50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #333;
                border: none;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton#saveButton {
                background-color: #4CAF50;
            }
            QPushButton#saveButton:hover {
                background-color: #45a049;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                padding: 5px;
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin-right: 5px;
            }
            QListWidget {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                color: white;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
            }
        """)
    
    def create_timeframes_tab(self):
        """Create timeframes configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title and description
        title = QLabel("Select Timeframes to Analyze")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)
        
        layout.addSpacing(10)
        
        # Timeframe selection list
        group = QGroupBox("Available Timeframes (Select up to 7)")
        group_layout = QVBoxLayout()
        
        self.tf_list = QListWidget()
        self.tf_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        # Add all available timeframes
        available_tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W']
        self.tf_list.addItems(available_tfs)
        
        # Select currently configured timeframes
        current_tfs = self.current_settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W'])
        for i in range(self.tf_list.count()):
            item = self.tf_list.item(i)
            if item.text() in current_tfs:
                item.setSelected(True)
        
        # Selection change handler
        self.tf_list.itemSelectionChanged.connect(self.update_tf_count)
        
        group_layout.addWidget(self.tf_list)
        
        # Selection counter
        self.tf_count_label = QLabel(f"{len(current_tfs)} timeframes selected")
        self.tf_count_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        group_layout.addWidget(self.tf_count_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Quick select buttons
        btn_layout = QHBoxLayout()
        
        intraday_btn = QPushButton("Intraday (5m,15m,1h)")
        intraday_btn.clicked.connect(lambda: self.select_preset_tfs(['5m', '15m', '1h']))
        btn_layout.addWidget(intraday_btn)
        
        swing_btn = QPushButton("Swing (4h,1D,1W)")
        swing_btn.clicked.connect(lambda: self.select_preset_tfs(['4h', '1D', '1W']))
        btn_layout.addWidget(swing_btn)
        
        default_btn = QPushButton("Default (5m,15m,4h,1D,1W)")
        default_btn.clicked.connect(lambda: self.select_preset_tfs(['5m', '15m', '4h', '1D', '1W']))
        btn_layout.addWidget(default_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        
        return widget
    
    def create_timeframes_and_signals_tab(self):
        """Create combined timeframes and signal logic tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # --- TIMEFRAMES SECTION ---
        tf_title = QLabel("Select Timeframes to Analyze")
        tf_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(tf_title)
        
        layout.addSpacing(10)
        
        # Timeframe selection group
        tf_group = QGroupBox("Available Timeframes (Select up to 7)")
        tf_group_layout = QVBoxLayout()
        
        self.tf_list = QListWidget()
        self.tf_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        # Add all available timeframes
        available_tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W']
        self.tf_list.addItems(available_tfs)
        
        # Select currently configured timeframes
        current_tfs = self.current_settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W'])
        for i in range(self.tf_list.count()):
            item = self.tf_list.item(i)
            if item.text() in current_tfs:
                item.setSelected(True)
        
        # Selection change handler
        self.tf_list.itemSelectionChanged.connect(self.update_tf_count)
        
        tf_group_layout.addWidget(self.tf_list)
        
        # Selection counter
        self.tf_count_label = QLabel(f"{len(current_tfs)} timeframes selected")
        self.tf_count_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        tf_group_layout.addWidget(self.tf_count_label)
        
        tf_group.setLayout(tf_group_layout)
        layout.addWidget(tf_group)
        
        # Quick select buttons
        btn_layout = QHBoxLayout()
        
        intraday_btn = QPushButton("Intraday (5m,15m,1h)")
        intraday_btn.clicked.connect(lambda: self.select_preset_tfs(['5m', '15m', '1h']))
        btn_layout.addWidget(intraday_btn)
        
        swing_btn = QPushButton("Swing (4h,1D,1W)")
        swing_btn.clicked.connect(lambda: self.select_preset_tfs(['4h', '1D', '1W']))
        btn_layout.addWidget(swing_btn)
        
        default_btn = QPushButton("Default (5m,15m,4h,1D,1W)")
        default_btn.clicked.connect(lambda: self.select_preset_tfs(['5m', '15m', '4h', '1D', '1W']))
        btn_layout.addWidget(default_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addSpacing(20)
        
        # --- SIGNAL LOGIC SECTION ---
        signal_title = QLabel("Signal Generation Logic")
        signal_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(signal_title)
        
        signal_desc = QLabel("Configure how many timeframes must align to generate LONG or SHORT signals.")
        signal_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        signal_desc.setWordWrap(True)
        layout.addWidget(signal_desc)
        
        layout.addSpacing(10)
        
        # Signal thresholds
        thresholds_group = QGroupBox("Signal Thresholds")
        thresholds_layout = QVBoxLayout()
        
        # Get current thresholds
        current_thresholds = self.current_settings.get('signal_thresholds', {'bullish': 5, 'bearish': 5})
        
        # Bullish threshold
        bull_layout = QHBoxLayout()
        bull_layout.addWidget(QLabel("🟢 Bullish Signal Threshold:"))
        self.bullish_threshold = QSpinBox()
        self.bullish_threshold.setRange(1, 7)
        self.bullish_threshold.setValue(current_thresholds.get('bullish', 5))
        bull_layout.addWidget(self.bullish_threshold)
        bull_layout.addWidget(QLabel("/ selected timeframes must be bullish"))
        bull_layout.addStretch()
        thresholds_layout.addLayout(bull_layout)
        
        # Bearish threshold
        bear_layout = QHBoxLayout()
        bear_layout.addWidget(QLabel("🔴 Bearish Signal Threshold:"))
        self.bearish_threshold = QSpinBox()
        self.bearish_threshold.setRange(1, 7)
        self.bearish_threshold.setValue(current_thresholds.get('bearish', 5))
        bear_layout.addWidget(self.bearish_threshold)
        bear_layout.addWidget(QLabel("/ selected timeframes must be bearish"))
        bear_layout.addStretch()
        thresholds_layout.addLayout(bear_layout)
        
        thresholds_group.setLayout(thresholds_layout)
        layout.addWidget(thresholds_group)
        
        # Example
        example_group = QGroupBox("Example")
        example_layout = QVBoxLayout()
        example_text = QLabel(
            "If you have 5 timeframes selected:\n"
            "• Bullish threshold = 5 means ALL 5 must be bullish for 🟢 LONG (5/5)\n"
            "• Bullish threshold = 4 means 4 out of 5 bullish gives 🟡 Long (4/5)\n\n"
            "Same logic applies to bearish signals.\n"
            "If neither threshold is met, signal shows as ⚪ Mixed"
        )
        example_text.setStyleSheet("color: #888; font-size: 10px; padding: 10px; background-color: #1e1e1e; border-radius: 5px;")
        example_text.setWordWrap(True)
        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)
        layout.addWidget(example_group)
        
        layout.addStretch()
        
        return widget
    
    def create_sentiment_tab(self):
        """Create sentiment configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Sentiment Analysis Configuration")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)
        
        layout.addSpacing(10)
        
        # Enable/disable sentiment
        group = QGroupBox("Sentiment Settings")
        group_layout = QVBoxLayout()
        
        self.sentiment_enabled = QCheckBox("Enable Sentiment Analysis")
        self.sentiment_enabled.setChecked(self.current_settings.get('sentiment_enabled', True))
        group_layout.addWidget(self.sentiment_enabled)
        
        layout.addSpacing(10)
        
        # Timeframe selection
        tf_layout = QHBoxLayout()
        tf_layout.addWidget(QLabel("Sentiment Timeframe:"))
        self.sentiment_tf_combo = QComboBox()
        self.sentiment_tf_combo.addItems(['5m', '15m', '30m', '1h', '4h', '1D', '1W'])
        self.sentiment_tf_combo.setCurrentText(self.current_settings.get('sentiment_timeframe', '1D'))
        tf_layout.addWidget(self.sentiment_tf_combo)
        tf_layout.addStretch()
        group_layout.addLayout(tf_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        
        return widget
    
    def create_lookbacks_tab(self):
        """Create lookback periods configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Lookback Period Analysis")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)
        
        desc = QLabel("Configure up to 3 lookback periods to calculate price changes.\nExample: 30 candles back on 1D = monthly change %")
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(10)
        
        # Get current lookbacks
        current_lookbacks = self.current_settings.get('lookbacks', [
            {'enabled': True, 'timeframe': '1D', 'periods': 30, 'label': 'Monthly'},
            {'enabled': True, 'timeframe': '1D', 'periods': 7, 'label': 'Weekly'},
            {'enabled': False, 'timeframe': '4h', 'periods': 20, 'label': 'Custom'}
        ])
        
        # Ensure we have exactly 3 lookback configs
        while len(current_lookbacks) < 3:
            current_lookbacks.append({'enabled': False, 'timeframe': '1D', 'periods': 10, 'label': f'Lookback {len(current_lookbacks)+1}'})
        
        self.lookback_widgets = []
        
        # Create 3 lookback period configurations
        for i in range(3):
            lb = current_lookbacks[i]
            group = QGroupBox(f"Lookback Period #{i+1}")
            group_layout = QVBoxLayout()
            
            # Enable checkbox
            enabled_cb = QCheckBox(f"Enable Lookback #{i+1}")
            enabled_cb.setChecked(lb.get('enabled', False))
            group_layout.addWidget(enabled_cb)
            
            # Timeframe
            tf_layout = QHBoxLayout()
            tf_layout.addWidget(QLabel("Timeframe:"))
            tf_combo = QComboBox()
            tf_combo.addItems(['5m', '15m', '30m', '1h', '4h', '1D', '1W'])
            tf_combo.setCurrentText(lb.get('timeframe', '1D'))
            tf_layout.addWidget(tf_combo)
            tf_layout.addStretch()
            group_layout.addLayout(tf_layout)
            
            # Periods
            period_layout = QHBoxLayout()
            period_layout.addWidget(QLabel("Candles Back:"))
            period_spin = QSpinBox()
            period_spin.setRange(1, 500)
            period_spin.setValue(lb.get('periods', 30))
            period_layout.addWidget(period_spin)
            period_layout.addWidget(QLabel("candles"))
            period_layout.addStretch()
            group_layout.addLayout(period_layout)
            
            # Label
            label_layout = QHBoxLayout()
            label_layout.addWidget(QLabel("Display Label:"))
            label_edit = QLineEdit(lb.get('label', f'Lookback {i+1}'))
            label_layout.addWidget(label_edit)
            group_layout.addLayout(label_layout)
            
            group.setLayout(group_layout)
            layout.addWidget(group)
            
            # Store widgets for later retrieval
            self.lookback_widgets.append({
                'enabled': enabled_cb,
                'timeframe': tf_combo,
                'periods': period_spin,
                'label': label_edit
            })
        
        layout.addStretch()
        
        return widget
    
    def create_signal_logic_tab(self):
        """Create signal generation logic tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Signal Generation Logic")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)
        
        desc = QLabel("Configure how many timeframes must align to generate LONG or SHORT signals.")
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(10)
        
        # Signal thresholds
        group = QGroupBox("Signal Thresholds")
        group_layout = QVBoxLayout()
        
        # Get current thresholds
        current_thresholds = self.current_settings.get('signal_thresholds', {'bullish': 5, 'bearish': 5})
        
        # Bullish threshold
        bull_layout = QHBoxLayout()
        bull_layout.addWidget(QLabel("🟢 Bullish Signal Threshold:"))
        self.bullish_threshold = QSpinBox()
        self.bullish_threshold.setRange(1, 7)
        self.bullish_threshold.setValue(current_thresholds.get('bullish', 5))
        bull_layout.addWidget(self.bullish_threshold)
        bull_layout.addWidget(QLabel("/ selected timeframes must be bullish (accumulation/re-accumulation)"))
        bull_layout.addStretch()
        group_layout.addLayout(bull_layout)
        
        # Bearish threshold
        bear_layout = QHBoxLayout()
        bear_layout.addWidget(QLabel("🔴 Bearish Signal Threshold:"))
        self.bearish_threshold = QSpinBox()
        self.bearish_threshold.setRange(1, 7)
        self.bearish_threshold.setValue(current_thresholds.get('bearish', 5))
        bear_layout.addWidget(self.bearish_threshold)
        bear_layout.addWidget(QLabel("/ selected timeframes must be bearish (distribution/re-distribution)"))
        bear_layout.addStretch()
        group_layout.addLayout(bear_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Example
        example_group = QGroupBox("Example")
        example_layout = QVBoxLayout()
        example_text = QLabel(
            "If you have 5 timeframes selected:\n"
            "• Bullish threshold = 5 means ALL 5 must be bullish for 🟢 LONG (5/5)\n"
            "• Bullish threshold = 4 means 4 out of 5 bullish gives 🟡 Long (4/5)\n"
            "• Bullish threshold = 3 means 3 out of 5 bullish gives signal\n\n"
            "Same logic applies to bearish signals.\n"
            "If neither threshold is met, signal shows as ⚪ Mixed"
        )
        example_text.setStyleSheet("color: #888; font-size: 10px; padding: 10px; background-color: #1e1e1e; border-radius: 5px;")
        example_text.setWordWrap(True)
        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)
        layout.addWidget(example_group)
        
        layout.addStretch()
        
        return widget
    
    def create_recommendations_tab(self):
        """Create recommendations tab (from original settings)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Recommendation Filter Settings")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)
        
        desc = QLabel("Configure which pairs to highlight as recommended setups.")
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(10)
        
        # Recommendation Settings
        group = QGroupBox("Recommendation Criteria")
        group_layout = QVBoxLayout()
        
        self.recommend_bull = QCheckBox("Full Bull (All timeframes bullish)")
        self.recommend_bear = QCheckBox("Full Bear (All timeframes bearish)")
        self.recommend_percent = QCheckBox("Based on % Change")
        self.recommend_sentiment = QCheckBox("Based on Sentiment Score")
        
        # Set current selections
        recommend_types = self.current_settings.get('recommend_types', ['Full Bull', 'Full Bear'])
        self.recommend_bull.setChecked('Full Bull' in recommend_types)
        self.recommend_bear.setChecked('Full Bear' in recommend_types)
        self.recommend_percent.setChecked('Based on % Gained' in recommend_types or 'Based on % Change' in recommend_types)
        self.recommend_sentiment.setChecked('Based on Sentiment' in recommend_types)
        
        group_layout.addWidget(self.recommend_bull)
        group_layout.addWidget(self.recommend_bear)
        group_layout.addWidget(self.recommend_percent)
        
        # Percent threshold
        percent_layout = QHBoxLayout()
        percent_layout.addSpacing(20)
        percent_layout.addWidget(QLabel("Minimum % Change:"))
        self.percent_spin = QDoubleSpinBox()
        self.percent_spin.setRange(0.1, 100.0)
        self.percent_spin.setValue(self.current_settings.get('recommend_percent', 5.0))
        self.percent_spin.setSingleStep(0.5)
        percent_layout.addWidget(self.percent_spin)
        percent_layout.addStretch()
        group_layout.addLayout(percent_layout)
        
        group_layout.addWidget(self.recommend_sentiment)
        
        # Sentiment slider
        sentiment_layout = QHBoxLayout()
        sentiment_layout.addSpacing(20)
        sentiment_layout.addWidget(QLabel("Min Sentiment Score:"))
        self.sentiment_slider = QSlider(Qt.Orientation.Horizontal)
        self.sentiment_slider.setRange(0, 100)
        self.sentiment_slider.setValue(self.current_settings.get('recommend_sentiment', 70))
        self.sentiment_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sentiment_slider.setTickInterval(10)
        sentiment_layout.addWidget(self.sentiment_slider)
        self.sentiment_label = QLabel(str(self.current_settings.get('recommend_sentiment', 70)))
        self.sentiment_label.setMinimumWidth(30)
        sentiment_layout.addWidget(self.sentiment_label)
        group_layout.addLayout(sentiment_layout)
        
        # Connect slider to label
        self.sentiment_slider.valueChanged.connect(
            lambda v: self.sentiment_label.setText(str(v))
        )
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Sort options
        sort_group = QGroupBox("Sort Results By")
        sort_layout = QVBoxLayout()
        
        self.sort_largest = QCheckBox("Largest Mover")
        self.sort_bullish = QCheckBox("Fully Bullish")
        self.sort_bearish = QCheckBox("Fully Bearish")
        
        # Set current selections
        sort_options = self.current_settings.get('sort_by', ['Largest Mover'])
        self.sort_largest.setChecked('Largest Mover' in sort_options)
        self.sort_bullish.setChecked('Fully Bullish' in sort_options)
        self.sort_bearish.setChecked('Fully Bearish' in sort_options)
        
        sort_layout.addWidget(self.sort_largest)
        sort_layout.addWidget(self.sort_bullish)
        sort_layout.addWidget(self.sort_bearish)
        
        sort_group.setLayout(sort_layout)
        layout.addWidget(sort_group)
        
        layout.addStretch()
        
        return widget
    
    def update_tf_count(self):
        """Update timeframe selection count"""
        selected = len(self.tf_list.selectedItems())
        self.tf_count_label.setText(f"{selected} timeframe{'s' if selected != 1 else ''} selected")
        
        if selected > 7:
            self.tf_count_label.setStyleSheet("color: #f44336; font-weight: bold;")
        else:
            self.tf_count_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def select_preset_tfs(self, tfs):
        """Select a preset group of timeframes"""
        # Clear all selections
        for i in range(self.tf_list.count()):
            self.tf_list.item(i).setSelected(False)
        
        # Select preset timeframes
        for i in range(self.tf_list.count()):
            item = self.tf_list.item(i)
            if item.text() in tfs:
                item.setSelected(True)
    
    def get_settings(self):
        """Get current settings from dialog"""
        # Get selected timeframes
        selected_tfs = [item.text() for item in self.tf_list.selectedItems()]
        
        # Get lookback configurations
        lookbacks = []
        for widgets in self.lookback_widgets:
            lookbacks.append({
                'enabled': widgets['enabled'].isChecked(),
                'timeframe': widgets['timeframe'].currentText(),
                'periods': widgets['periods'].value(),
                'label': widgets['label'].text()
            })
        
        # Build recommend_types list
        recommend_types = []
        if self.recommend_bull.isChecked():
            recommend_types.append('Full Bull')
        if self.recommend_bear.isChecked():
            recommend_types.append('Full Bear')
        if self.recommend_percent.isChecked():
            recommend_types.append('Based on % Change')
        if self.recommend_sentiment.isChecked():
            recommend_types.append('Based on Sentiment')
        
        if not recommend_types:
            recommend_types = ['Full Bull', 'Full Bear']
        
        # Build sort_by list
        sort_by = []
        if self.sort_largest.isChecked():
            sort_by.append('Largest Mover')
        if self.sort_bullish.isChecked():
            sort_by.append('Fully Bullish')
        if self.sort_bearish.isChecked():
            sort_by.append('Fully Bearish')
        
        if not sort_by:
            sort_by = ['Largest Mover']
        
        return {
            # Timeframes
            'timeframes': selected_tfs,
            
            # Sentiment
            'sentiment_enabled': self.sentiment_enabled.isChecked(),
            'sentiment_timeframe': self.sentiment_tf_combo.currentText(),
            
            # Lookbacks
            'lookbacks': lookbacks,
            
            # Signal thresholds
            'signal_thresholds': {
                'bullish': self.bullish_threshold.value(),
                'bearish': self.bearish_threshold.value()
            },
            
            # Recommendations (legacy support)
            'recommend_types': recommend_types,
            'recommend_percent': self.percent_spin.value(),
            'recommend_sentiment': self.sentiment_slider.value(),
            'sort_by': sort_by
        }
