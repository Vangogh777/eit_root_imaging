import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

class VideoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/video/'):
            filename = self.path[7:]
            video_dir = '/home/ubuntu/synthflix/public/videos'
            filepath = os.path.join(video_dir, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', str(os.path.getsize(filepath)))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<a href="/video/dac20588-cb8b-4b3e-8d61-cf6f1a4e7238.mp4">Test Video</a>')

class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == '__main__':
    server = ThreadedServer(('0.0.0.0', 8081), VideoHandler)
    print('Video test server on :8081')
    server.serve_forever()
