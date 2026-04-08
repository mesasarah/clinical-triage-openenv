"""
OpenEnv server entrypoint.
This exposes a main() function required for multi-mode deployment.
"""

import os
import uvicorn
from server.main import app

def main():
"""Entry point for OpenEnv validator"""
port = int(os.getenv("PORT", "7860"))
uvicorn.run("server.main:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
