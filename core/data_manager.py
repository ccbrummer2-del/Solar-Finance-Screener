"""
Data persistence manager for Solar Terminal
Handles saving/loading settings and scan results
"""
import json
import os
from datetime import datetime
from pathlib import Path

class DataManager:
    """Manages persistent storage of settings and scan data"""
    
    def __init__(self):
        # Create data directory in user's home folder
        self.data_dir = Path.home() / '.solar_terminal'
        self.data_dir.mkdir(exist_ok=True)
        
        self.settings_file = self.data_dir / 'settings.json'
        self.scans_dir = self.data_dir / 'scans'
        self.scans_dir.mkdir(exist_ok=True)
    
    def save_settings(self, settings):
        """Save user settings to disk"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def load_settings(self):
        """Load user settings from disk"""
        if not self.settings_file.exists():
            return None
        
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            return None
    
    def save_scan_results(self, results, selected_pairs):
        """Save scan results to disk with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.scans_dir / f'scan_{timestamp}.json'
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'selected_pairs': selected_pairs,
            'results': results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Keep only last 100 scans to avoid filling up disk
            self._cleanup_old_scans(keep=100)
            return True
        except Exception as e:
            print(f"Error saving scan results: {e}")
            return False
    
    def load_latest_scan(self):
        """Load the most recent scan results"""
        scan_files = sorted(self.scans_dir.glob('scan_*.json'), reverse=True)
        
        if not scan_files:
            return None
        
        try:
            with open(scan_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading latest scan: {e}")
            return None
    
    def get_scan_history(self, limit=50):
        """Get list of recent scans"""
        scan_files = sorted(self.scans_dir.glob('scan_*.json'), reverse=True)[:limit]
        
        history = []
        for scan_file in scan_files:
            try:
                with open(scan_file, 'r') as f:
                    data = json.load(f)
                    history.append({
                        'filename': scan_file.name,
                        'timestamp': data.get('timestamp'),
                        'num_results': len(data.get('results', []))
                    })
            except Exception as e:
                print(f"Error reading {scan_file}: {e}")
        
        return history
    
    def load_scan_by_filename(self, filename):
        """Load a specific scan by filename"""
        filepath = self.scans_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading scan {filename}: {e}")
            return None
    
    def _cleanup_old_scans(self, keep=100):
        """Keep only the N most recent scans"""
        scan_files = sorted(self.scans_dir.glob('scan_*.json'), reverse=True)
        
        # Delete old scans
        for old_scan in scan_files[keep:]:
            try:
                old_scan.unlink()
            except Exception as e:
                print(f"Error deleting old scan {old_scan}: {e}")
    
    def export_scan_to_csv(self, results, filepath):
        """Export scan results to CSV"""
        import csv
        
        if not results:
            return False
        
        try:
            with open(filepath, 'w', newline='') as f:
                # Get all possible keys from results
                keys = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def get_data_directory(self):
        """Get the data directory path"""
        return str(self.data_dir)
