import os
import sys
import importlib

print("=== Python Import Debugging ===")
print(f"Current directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print(f"sys.path: {sys.path}")

# Check if data module can be imported directly
print("\nAttempting direct imports...")
try:
    import data
    print("✓ Successfully imported 'data' module")
except Exception as e:
    print(f"✗ Failed to import 'data' module: {e}")

# Try to import volume directly
try:
    from data.volume import Volume
    print("✓ Successfully imported 'Volume' class")
except Exception as e:
    print(f"✗ Failed to import 'Volume' class: {e}")
    print(f"  Import error type: {type(e).__name__}")

# Try to import specific functions
print("\nAttempting to import specific functions...")
try:
    from data.io.paths import list_files, is_aws_ec2_instance
    print("✓ Successfully imported functions from data.io.paths")
except Exception as e:
    print(f"✗ Failed to import from data.io.paths: {e}")

try:
    from setup.accept_terms import get_installation_path
    print("✓ Successfully imported get_installation_path")
    print(f"  Installation path: {get_installation_path()}")
except Exception as e:
    print(f"✗ Failed to import get_installation_path: {e}")

# Try to add the current directory to the path and try again
print("\nAttempting to fix imports by modifying sys.path...")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"Updated sys.path: {sys.path}")

try:
    # Clear cache first to ensure reimport
    if 'data' in sys.modules:
        del sys.modules['data']
    if 'data.volume' in sys.modules:
        del sys.modules['data.volume']
        
    import data
    print("✓ Successfully imported 'data' module after path fix")
except Exception as e:
    print(f"✗ Still failed to import 'data' module: {e}")

try:
    from data.volume import Volume
    print("✓ Successfully imported 'Volume' class after path fix")
except Exception as e:
    print(f"✗ Still failed to import 'Volume' class: {e}")
    
# Try examining errors in more detail
print("\nExamining module structure...")
try:
    spec = importlib.util.find_spec('data')
    if spec:
        print(f"data module spec: {spec}")
        print(f"data module location: {spec.origin}")
    else:
        print("Could not find spec for data module")
except Exception as e:
    print(f"Error examining module: {e}")