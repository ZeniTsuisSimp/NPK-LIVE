"""
Hugging Face Spaces entry point.
Redirects to the main Streamlit app in app/ directory.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/npk_crop_recommendation_app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
    ])
