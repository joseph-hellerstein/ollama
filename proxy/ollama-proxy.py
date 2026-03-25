from http.server import HTTPServer, BaseHTTPRequestHandler
import json, urllib.request, urllib.error

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        for key in ['think', 'thinking', 'enable_thinking']:
            body.pop(key, None)
        print(f"Forwarding request, remaining keys: {list(body.keys())}")
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            'http://localhost:11434' + self.path,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req) as r:
                resp = r.read()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(resp)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

HTTPServer(('localhost', 11435), Handler).serve_forever()