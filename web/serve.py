"""Static server for the GhostShift viewer that disables browser caching, so edits show on plain reload.
    python web/serve.py 8765
"""
import sys, http.server, functools, pathlib

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()
    def log_message(self, fmt, *args):   # one line per request, no noise
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
root = pathlib.Path(__file__).resolve().parent
handler = functools.partial(NoCacheHandler, directory=str(root))
print(f"GhostShift viewer at http://localhost:{port}  (serving {root})", flush=True)
http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
