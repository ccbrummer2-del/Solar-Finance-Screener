"""
API Client for Solar Terminal Desktop App
Handles communication with server API and API key storage
"""
import requests
import json
from pathlib import Path

class APIClient:
    """Client to communicate with Solar Terminal API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_key = None
        self.config_dir = Path.home() / '.solar_terminal'
        self.auth_file = self.config_dir / 'auth.json'
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load API key if it exists
        self.load_api_key()
    
    def load_api_key(self):
        """Load API key from local config"""
        if not self.auth_file.exists():
            return None
        
        try:
            with open(self.auth_file, 'r') as f:
                data = json.load(f)
                self.api_key = data.get('api_key')
                return self.api_key
        except Exception as e:
            print(f"Error loading API key: {e}")
            return None
    
    def save_api_key(self, key):
        """Save API key to local config"""
        try:
            with open(self.auth_file, 'w') as f:
                json.dump({'api_key': key}, f, indent=2)
            self.api_key = key
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False
    
    def clear_api_key(self):
        """Remove stored API key"""
        try:
            if self.auth_file.exists():
                self.auth_file.unlink()
            self.api_key = None
            return True
        except Exception as e:
            print(f"Error clearing API key: {e}")
            return False
    
    def has_api_key(self):
        """Check if API key is configured"""
        return self.api_key is not None
    
    def get_masked_key(self):
        """Get masked version of API key for display"""
        if not self.api_key:
            return "Not configured"
        
        if len(self.api_key) > 14:
            return self.api_key[:10] + "..." + self.api_key[-4:]
        return self.api_key
    
    def test_connection(self):
        """Test connection to API server (no auth required)"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return True, "Connected to server"
            return False, f"Server returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to server. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, str(e)
    
    def verify_api_key(self, key=None):
        """Verify that an API key is valid"""
        test_key = key if key else self.api_key
        
        if not test_key:
            return False, "No API key provided"
        
        try:
            response = requests.get(
                f"{self.base_url}/ping",
                headers={"X-API-Key": test_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, f"Authenticated as {data['user']}"
            elif response.status_code == 401:
                return False, "Invalid API key"
            elif response.status_code == 403:
                return False, "API key has been deactivated"
            else:
                return False, f"Server error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to server"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, str(e)
    
    def get_latest_results(self):
        """Fetch latest scan results from server"""
        if not self.api_key:
            return None, "No API key configured"
        
        try:
            response = requests.get(
                f"{self.base_url}/latest-scan",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data, None
            elif response.status_code == 401:
                return None, "Invalid API key. Please update your key in settings."
            elif response.status_code == 403:
                return None, "Your API key has been deactivated. Contact administrator."
            elif response.status_code == 503:
                return None, "Scanner service not running yet. Please wait..."
            else:
                return None, f"Server error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to server. Check that server is running."
        except requests.exceptions.Timeout:
            return None, "Connection timeout. Server may be busy."
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_latest_stock_results(self):
        """Fetch latest STOCK scan results from server"""
        if not self.api_key:
            return None, "No API key configured"
        
        try:
            response = requests.get(
                f"{self.base_url}/latest-stock-scan",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data, None
            elif response.status_code == 401:
                return None, "Invalid API key. Please update your key in settings."
            elif response.status_code == 403:
                return None, "Your API key has been deactivated. Contact administrator."
            elif response.status_code == 503:
                return None, "Stock scanner service not running yet. Please wait..."
            else:
                return None, f"Server error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to server. Check that server is running."
        except requests.exceptions.Timeout:
            return None, "Connection timeout. Server may be busy."
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_last_updated(self):
        """Get timestamp of last scan"""
        if not self.api_key:
            return None, "No API key configured"
        
        try:
            response = requests.get(
                f"{self.base_url}/last-updated",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['last_updated'], None
            else:
                return None, f"Error: {response.status_code}"
                
        except Exception as e:
            return None, str(e)
    
    def get_api_key_info(self):
        """Get information about current API key"""
        if not self.api_key:
            return None, "No API key configured"
        
        try:
            response = requests.get(
                f"{self.base_url}/api-keys/info",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"Error: {response.status_code}"
                
        except Exception as e:
            return None, str(e)
