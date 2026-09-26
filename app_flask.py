"""
Fishery Marine Intelligence Web Application Launcher
Runs the primary application defined in app.py on http://localhost:5000
"""
from app import app

if __name__ == "__main__":
    print("Starting Fishery Marine Web Application on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
