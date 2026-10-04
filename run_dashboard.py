import os
import sys
import subprocess

if __name__ == "__main__":
    # FIXED: Maps the native argument sequence correctly to find app.py pathing
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    app_script = os.path.join(base_path, "app.py")
    
    # Run the underlying Streamlit server loops seamlessly
    cmd = [sys.executable, "-m", "streamlit", "run", app_script, "--global.developmentMode=false"]
    subprocess.run(cmd)
