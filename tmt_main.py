import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SPLIT    = '90_10'
PORT     = 8000

# ── Model ─────────────────────────────────────────────────────────────────────

_model        = None
_backbone     = None
_class_names  = None
_use_features = False


def _load_model():
    global _model, _backbone, _class_names, _use_features
    from tensorflow import keras

    model_path = BASE_DIR / f'tomato_disease_{SPLIT}.h5'
    class_file = BASE_DIR / f'features_{SPLIT}' / 'class_names.json'

    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not class_file.exists():
        raise FileNotFoundError(f'class_names.json not found: {class_file}')

    print(f'Loading {model_path.name} ...')
    _model = keras.models.load_model(str(model_path))

    with open(class_file) as f:
        _class_names = json.load(f)

    _use_features = (len(_model.input_shape[1:]) == 1)

    if _use_features:
        _backbone = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg',
            weights='imagenet',
        )
        _backbone.trainable = False

    print(f'Classes : {_class_names}')
    print('Model ready.\n')


def _predict(image_bytes: bytes) -> dict:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    img       = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img       = tf.image.resize(img, (224, 224))
    img_batch = tf.expand_dims(tf.cast(img, tf.float32), 0)

    if _use_features:
        preprocess = keras.applications.mobilenet_v2.preprocess_input
        features   = _backbone(preprocess(img_batch), training=False)
        probs      = _model(features, training=False).numpy()[0]
    else:
        probs = _model(img_batch / 255.0, training=False).numpy()[0]

    top_i = int(np.argmax(probs))
    return {
        'predicted_class': _class_names[top_i],
        'confidence'     : round(float(probs[top_i]) * 100, 2),
        'all_probabilities': {
            name: round(float(prob) * 100, 2)
            for name, prob in zip(_class_names, probs)
        },
    }


# ── Server ────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _CONTENT_TYPES = {
        '.html': 'text/html; charset=utf-8',
        '.css' : 'text/css; charset=utf-8',
        '.js'  : 'application/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.png' : 'image/png',
        '.jpg' : 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif' : 'image/gif',
        '.svg' : 'image/svg+xml',
        '.ico' : 'image/x-icon',
    }

    def _send_file(self, path: Path):
        data = path.read_bytes()
        ctype = self._CONTENT_TYPES.get(path.suffix.lower(), 'application/octet-stream')
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _redirect(self, location: str):
        self.send_response(302)
        self.send_header('Location', location)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # Strip query string / fragment if present.
        path = self.path.split('?', 1)[0].split('#', 1)[0]

        # Home page → redirect to the scan UI at its real path so that the
        # page's relative asset links (../css, ../js, ../images) resolve.
        if path in ('/', '/index.html', '/scan', '/scan.html'):
            self._redirect('/UI/html/scan.html')
            return

        if path == '/favicon.ico':
            icon = BASE_DIR / 'UI' / 'images' / 'logo.png'
            if icon.exists():
                self._send_file(icon)
            else:
                self.send_response(204)
                self.end_headers()
            return

        # Serve any static asset that lives inside the project directory.
        target = (BASE_DIR / path.lstrip('/')).resolve()
        try:
            target.relative_to(BASE_DIR)  # guard against path traversal
        except ValueError:
            self._send_json({'error': 'Forbidden'}, 403)
            return

        if target.is_file():
            self._send_file(target)
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path != '/predict':
            self._send_json({'error': 'Not found'}, 404)
            return
        try:
            length      = int(self.headers.get('Content-Length', 0))
            body        = self.rfile.read(length)
            data        = json.loads(body)
            image_bytes = base64.b64decode(data['image'])
            self._send_json(_predict(image_bytes))
        except Exception as exc:
            self._send_json({'error': str(exc)}, 500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _load_model()
    httpd = HTTPServer(('127.0.0.1', PORT), _Handler)
    print(f'Server running  →  http://127.0.0.1:{PORT}  (scan UI)')
    print('Press Ctrl+C to stop.\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')
