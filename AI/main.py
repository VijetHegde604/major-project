import http.server
import socketserver

# Define the port number for the server to listen on.
PORT = 8000

# Define a custom request handler by inheriting from SimpleHTTPRequestHandler.
# This handler will serve files from the current directory by default,
# but we'll override its do_GET method to send a custom "Hello, World!" response.
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Set the HTTP response status code to 200 (OK).
        self.send_response(200)

        # Set the Content-type header to indicate that the response is plain text.
        self.send_header("Content-type", "text/plain")

        # End the headers section.
        self.end_headers()

        # Send the "Hello, World!" message as bytes (UTF-8 encoded).
        self.wfile.write(b"Hello, World!\n Working OK!! I hate niggers")

# Create a TCP server that uses our custom handler.
# The server will listen on all available network interfaces (empty string for host)
# and the specified PORT.
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving on port {PORT}")
    print(f"You can access it at http://localhost:{PORT}/")

    # Start the server and keep it running indefinitely until interrupted (e.g., Ctrl+C).
    httpd.serve_forever()