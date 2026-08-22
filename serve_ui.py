"""Minimal HTTP server for DetectDump UI."""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")

os.chdir(UI_DIR)

handler = http.server.SimpleHTTPRequestHandler
with http.server.HTTPServer(("127.0.0.1", PORT), handler) as httpd:
    print(f"DetectDump UI running at http://127.0.0.1:{PORT}/detectdump.html")
    httpd.serve_forever()
