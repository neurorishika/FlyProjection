from flask import Flask, Response, render_template_string
import redis
import cv2
import numpy as np

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)

# HTML template with CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FlyProjection</title>
    <style>
        body { display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #f0f0f0; }
        img { }
        div { box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 10px; background: white; }
        h1 { font-family: Arial, sans-serif; color: #333; text-align: center; }
    </style>
</head>
<body>
    <div>
        <h1>FlyProjection Feed</h1>
        <img src="{{ url_for('stream') }}" alt="Live image stream">
    </div>
    <script>
        setInterval(function() {
            const img = document.querySelector('img');
            img.src = "{{ url_for('stream') }}" + "?t=" + new Date().getTime();
        }, 100);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stream')
def stream():
    img_bytes = r.get('latest_image')
    if img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, img_encoded = cv2.imencode('.png', img)
        return Response(img_encoded.tobytes(), mimetype='image/png')
    else:
        return Response("No image available", mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)