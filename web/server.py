
"""
flask で簡易サーバを立てる
./html の下に各種ファイルを設置する
API実装は無いので静的ファイルのみを配信
http://localhost:8000/index.html でアクセス
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS
import argparse
import json
import sys

app = Flask(__name__, static_url_path='', static_folder='html')
CORS(app)  # すべてのドメインからのアクセスを許可
PORT = 8000

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print(f"Starting server at http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT)

# export PYTHONPATH=/workspaces/pj0005_horse_name
# python web/server.py
# http://localhost:8000/index.html でアクセス
# あるいは
# http://host.docker.internal:8000/index.html でアクセス
