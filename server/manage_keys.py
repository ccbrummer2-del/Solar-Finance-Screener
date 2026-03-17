"""
API Key Management CLI for Solar Terminal
Terminal-based tool to add/remove/manage API keys
"""
import json
import secrets
import argparse
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

API_KEYS_FILE = Path(__file__).parent / 'api_keys.json'

def load_keys():
    """Load API keys from file"""
    if not API_KEYS_FILE.exists():
        return {}
    
    with open(API_KEYS_FILE, 'r') as f:
        return json.load(f)

def save_keys(keys):
    """Save API keys to file"""
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def generate_key(name):
    """Generate a new API key"""
    # Format: sk_firstname_random
    first_name = name.split()[0].lower()
    random_part = secrets.token_hex(6)  # 12 character random string
    return f"sk_{first_name}_{random_part}"

def add_key(name, email):
    """Add a new API key"""
    keys = load_keys()
    
    # Generate unique key
    new_key = generate_key(name)
    
    # Ensure key is unique (very unlikely to collide, but just in case)
    while new_key in keys:
        new_key = generate_key(name)
    
    # Create key entry
    keys[new_key] = {
        "name": name,
        "email": email,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active": True
    }
    
    save_keys(keys)
    
    print("\n✅ API Key Created Successfully!")
    print("=" * 60)
    print(f"Name:  {name}")
    print(f"Email: {email}")
    print(f"Key:   {new_key}")
    print("=" * 60)
    print("\n📧 Send this key to the user:")
    print(f"\n    {new_key}\n")
    print("🔒 They will enter it once in Solar Terminal to authenticate.")
    
    return new_key

def list_keys():
    """List all API keys"""
    keys = load_keys()
    
    if not keys:
        print("\n❌ No API keys found.")
        print("Create one with: python manage_keys.py add \"Name\" \"email@example.com\"")
        return
    
    # Prepare table data
    table_data = []
    for key, info in keys.items():
        status = "✅ Active" if info.get('active', True) else "❌ Inactive"
        # Mask the key for security
        masked_key = key[:10] + "..." + key[-4:]
        table_data.append([
            masked_key,
            info['name'],
            info['email'],
            info.get('created', 'N/A'),
            status
        ])
    
    print("\n" + "=" * 80)
    print("SOLAR TERMINAL API KEYS")
    print("=" * 80)
    print(tabulate(table_data, headers=['Key', 'Name', 'Email', 'Created', 'Status'], tablefmt='grid'))
    print(f"\nTotal: {len(keys)} key(s)")
    print("=" * 80 + "\n")

def deactivate_key(key):
    """Deactivate an API key"""
    keys = load_keys()
    
    if key not in keys:
        print(f"\n❌ Error: Key '{key}' not found")
        return
    
    keys[key]['active'] = False
    save_keys(keys)
    
    print(f"\n✅ Key deactivated: {key}")
    print(f"   User: {keys[key]['name']} ({keys[key]['email']})")
    print("   This user can no longer access the API.")

def activate_key(key):
    """Activate an API key"""
    keys = load_keys()
    
    if key not in keys:
        print(f"\n❌ Error: Key '{key}' not found")
        return
    
    keys[key]['active'] = True
    save_keys(keys)
    
    print(f"\n✅ Key activated: {key}")
    print(f"   User: {keys[key]['name']} ({keys[key]['email']})")
    print("   This user can now access the API.")

def remove_key(key):
    """Completely remove an API key"""
    keys = load_keys()
    
    if key not in keys:
        print(f"\n❌ Error: Key '{key}' not found")
        return
    
    user_info = keys[key]
    del keys[key]
    save_keys(keys)
    
    print(f"\n✅ Key permanently removed: {key}")
    print(f"   User: {user_info['name']} ({user_info['email']})")
    print("   ⚠️  This action cannot be undone.")

def show_key(search_term):
    """Show full key by searching for name or partial key"""
    keys = load_keys()
    
    matches = []
    for key, info in keys.items():
        if (search_term.lower() in info['name'].lower() or 
            search_term.lower() in info['email'].lower() or
            search_term in key):
            matches.append((key, info))
    
    if not matches:
        print(f"\n❌ No keys found matching: {search_term}")
        return
    
    print("\n" + "=" * 80)
    print("MATCHING KEYS")
    print("=" * 80)
    
    for key, info in matches:
        status = "✅ Active" if info.get('active', True) else "❌ Inactive"
        print(f"\nName:    {info['name']}")
        print(f"Email:   {info['email']}")
        print(f"Created: {info.get('created', 'N/A')}")
        print(f"Status:  {status}")
        print(f"Key:     {key}")
        print("-" * 80)
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Manage API keys for Solar Terminal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a new user
  python manage_keys.py add "John Doe" "john@gmail.com"
  
  # List all keys
  python manage_keys.py list
  
  # Show full key (search by name/email/partial key)
  python manage_keys.py show john
  
  # Deactivate a key
  python manage_keys.py deactivate sk_john_abc123
  
  # Reactivate a key
  python manage_keys.py activate sk_john_abc123
  
  # Permanently remove a key
  python manage_keys.py remove sk_john_abc123
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new API key')
    add_parser.add_argument('name', help='User full name')
    add_parser.add_argument('email', help='User email address')
    
    # List command
    subparsers.add_parser('list', help='List all API keys')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show full key (search by name/email/key)')
    show_parser.add_argument('search', help='Search term (name, email, or partial key)')
    
    # Deactivate command
    deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate an API key')
    deactivate_parser.add_argument('key', help='API key to deactivate')
    
    # Activate command
    activate_parser = subparsers.add_parser('activate', help='Activate an API key')
    activate_parser.add_argument('key', help='API key to activate')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Permanently remove an API key')
    remove_parser.add_argument('key', help='API key to remove')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'add':
        add_key(args.name, args.email)
    elif args.command == 'list':
        list_keys()
    elif args.command == 'show':
        show_key(args.search)
    elif args.command == 'deactivate':
        deactivate_key(args.key)
    elif args.command == 'activate':
        activate_key(args.key)
    elif args.command == 'remove':
        remove_key(args.key)

if __name__ == '__main__':
    main()
