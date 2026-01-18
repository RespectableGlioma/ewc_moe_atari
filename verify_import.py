import os
import sys

# Simulate the notebook logic
PROJECT_DIR = os.getcwd()
SRC_DIR = os.path.join(PROJECT_DIR, 'src')

print(f"Current working directory: {PROJECT_DIR}")
print(f"Expected src directory: {SRC_DIR}")

if os.path.exists(SRC_DIR):
    print("src directory exists.")
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
        print("Added src to sys.path")
    
    try:
        from envs import make_atari_env
        print("Successfully imported make_atari_env from envs")
    except ImportError as e:
        print(f"Import failed: {e}")
        # Detailed debugging
        try:
            import envs
            print(f"envs module found at: {envs.__file__}")
        except ImportError:
            print("envs module could not be imported at all.")
            
else:
    print("src directory DOES NOT exist at the expected path.")
    print(f"Contents of {PROJECT_DIR}:")
    print(os.listdir(PROJECT_DIR))
