"""
Solar Finance - FX Screener Desktop Application
Main application entry point
"""
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui.main_window import MainWindow

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not bundled, use normal path
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def main():
    """Run the application"""
    # Fix for Windows taskbar icon
    # This tells Windows to treat this as a separate app, not a Python script
    try:
        from ctypes import windll  # Only exists on Windows
        myappid = 'solarfinance.fxscreener.desktop.1.0'  # Arbitrary string
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        pass  # Not on Windows, no problem
    
    app = QApplication(sys.argv)
    app.setApplicationName("Solar Terminal")
    app.setStyle('Fusion')  # Modern look across all platforms
    
    # Set application icon
    # Try to find the icon using resource path
    icon_paths = [
        get_resource_path('assets/solar_icon.ico'),
        get_resource_path('assets/icon.ico'),
        get_resource_path('assets/solar_icon.png'),
        get_resource_path('assets/icon.png'),
        'assets/solar_icon.ico',  # Fallback
        'assets/icon.ico',
    ]
    
    icon_loaded = False
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            print(f"✓ Application icon loaded: {icon_path}")
            icon_loaded = True
            break
    
    if not icon_loaded:
        print(f"⚠ Warning: No icon file found. Searched in:")
        for path in icon_paths:
            print(f"  - {path} (exists: {os.path.exists(path)})")
        print(f"Current directory: {os.getcwd()}")
        print(f"Base path: {get_resource_path('.')}")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
