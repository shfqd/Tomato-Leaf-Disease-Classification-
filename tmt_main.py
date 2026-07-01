# ══════════════════════════════════════════════════════════════════════════════
# tmt_main.py — Inference Web Server
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Runs a lightweight HTTP server that exposes the trained tomato disease
#   classifier as a REST API.  A browser-based test page (test.html) is
#   also served so you can upload a leaf image and see a prediction
#   immediately without writing any client code.
#
# HOW IT WORKS:
#   1. On startup, the saved Keras model (tomato_disease_<SPLIT>.h5) is
#      loaded into memory once and kept alive for the lifetime of the server.
#   2. The server listens on localhost:8000 for two routes:
#        GET  /           -> serves test.html (the browser UI)
#        POST /predict    -> accepts a base64-encoded image, runs the model,
#                           and returns a JSON response with the predicted
#                           class and confidence scores.
#
# PREDICTION MODES (selected automatically):
#   Feature mode  -- if the model input is a 1-D vector, the server also
#                    loads the MobileNetV2 backbone and uses it to extract
#                    a 1280-d feature vector before calling the classifier.
#                    This matches the tmt_4 + tmt_5 feature-extraction pipeline.
#   Raw-image mode -- if the model input is a 3-D tensor (H, W, C), the
#                    image is passed directly to the CNN trained in tmt_5.
#
# CONFIGURATION:
#   SPLIT  -- which trained model to load ('90_10', '80_20', or '70_30')
#   PORT   -- TCP port the server binds to (default 8000)
#
# RUN   : python tmt_main.py
# OPEN  : http://127.0.0.1:8000  in your browser
# ══════════════════════════════════════════════════════════════════════════════

import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SPLIT    = '90_10'   # model to load; change to '80_20' or '70_30' to compare splits
PORT     = 8000


# ── Model ─────────────────────────────────────────────────────────────────────
# These module-level variables hold the loaded model and backbone so they are
# initialised once at startup and reused for every prediction request.
# Lazy initialisation via _load_model() avoids importing TensorFlow until
# the server actually starts.

_model        = None   # the trained Dense classifier (or CNN) from tmt_5
_backbone     = None   # MobileNetV2 backbone (only used in feature mode)
_class_names  = None   # list of disease class names, e.g. ['healthy', 'Bacterial_spot', ...]
_use_features = False  # True if the model expects a 1-D feature vector (feature mode)


def _load_model():
    """
    Load the Keras classifier and, if required, the MobileNetV2 backbone.

    Called once at startup before the HTTP server begins accepting requests.
    Determines the prediction mode by inspecting the model's input shape:
      - 1-D input -> feature mode (backbone needed)
      - 3-D input -> raw-image CNN mode (no backbone needed)
    """
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

    # Inspect the input rank to decide which prediction path to use.
    # A Dense-only model has input shape (None, 1280) -> rank 1 after removing batch dim.
    # A CNN model has input shape (None, 224, 224, 3) -> rank 3 after removing batch dim.
    _use_features = (len(_model.input_shape[1:]) == 1)

    if _use_features:
        # Feature mode: load the frozen backbone so incoming images can be
        # converted to 1280-d vectors before being passed to the classifier.
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
    """
    Run a single image through the model and return the prediction result.

    Pipeline:
      1. Decode the raw image bytes to a float32 RGB tensor.
      2. Resize to 224x224 (required by MobileNetV2 and the CNN input layer).
      3. Feature mode: preprocess with MobileNetV2 normalisation, extract
         features via the backbone, then classify.
         Raw-image mode: normalise pixels to [0, 1] and pass directly to CNN.
      4. Return the top predicted class, its confidence %, and the full
         probability distribution across all classes.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG, PNG, etc.)

    Returns:
        dict with keys:
          'predicted_class'    -- name of the most likely disease class
          'confidence'         -- confidence percentage (0.0 -- 100.0)
          'all_probabilities'  -- {class_name: probability %} for all classes
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    # Decode any common image format (JPEG, PNG, GIF, BMP) to a uint8 tensor
    img       = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img       = tf.image.resize(img, (224, 224))
    # Add the batch dimension: (H, W, C) -> (1, H, W, C)
    img_batch = tf.expand_dims(tf.cast(img, tf.float32), 0)

    if _use_features:
        # Feature mode: scale to [-1, 1] (MobileNetV2 convention), extract
        # 1280-d feature vector, then classify
        preprocess = keras.applications.mobilenet_v2.preprocess_input
        features   = _backbone(preprocess(img_batch), training=False)
        probs      = _model(features, training=False).numpy()[0]
    else:
        # Raw-image mode: scale pixels from [0, 255] to [0, 1]
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
    """
    Minimal HTTP request handler built on Python's standard library.

    Routes:
      OPTIONS  *           -- CORS pre-flight response (required by browsers
                              when the UI and API are on the same origin)
      GET  /               -- serve test.html (browser upload UI)
      GET  /test.html      -- same as above
      GET  /favicon.ico    -- silently return 204 (no icon needed)
      POST /predict        -- run inference; body must be JSON with an
                              'image' key containing a base64-encoded image
    """

    def log_message(self, fmt, *args):
        # Suppress the default per-request console logging to keep the
        # terminal output clean; errors are still propagated via exceptions.
        pass

    def _cors(self):
        """Append CORS headers so browser-based clients on any origin can call the API."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, data, status=200):
        """Serialise data to JSON and write the complete HTTP response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path):
        """Read an HTML file from disk and write it as the HTTP response body."""
        data = path.read_bytes()
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests sent by browsers before POST."""
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        """Serve the browser test UI or return 404 for unknown paths."""
        if self.path in ('/', '/test.html'):
            html = BASE_DIR / 'test.html'
            if html.exists():
                self._send_html(html)
            else:
                self._send_json({'error': 'test.html not found'}, 404)
        elif self.path == '/favicon.ico':
            # Browsers automatically request a favicon; return no content
            # to avoid a noisy 404 in the logs.
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        """
        Handle a prediction request.

        Expected request body (JSON):
          { "image": "<base64-encoded image bytes>" }

        Response body (JSON):
          {
            "predicted_class": "Bacterial_spot",
            "confidence": 94.12,
            "all_probabilities": {
              "healthy": 1.23,
              "Bacterial_spot": 94.12,
              "Leaf_Mold": 3.45,
              "Tomato_mosaic_virus": 1.20
            }
          }
        """
        if self.path != '/predict':
            self._send_json({'error': 'Not found'}, 404)
            return
        try:
            length      = int(self.headers.get('Content-Length', 0))
            body        = self.rfile.read(length)
            data        = json.loads(body)
            # Decode the base64 image string back to raw bytes for TensorFlow
            image_bytes = base64.b64decode(data['image'])
            self._send_json(_predict(image_bytes))
        except Exception as exc:
            # Return the exception message so the client can display a
            # meaningful error rather than a silent failure.
            self._send_json({'error': str(exc)}, 500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load the model before binding the socket so any missing-file errors
    # surface immediately rather than only on the first prediction request.
    _load_model()
    httpd = HTTPServer(('127.0.0.1', PORT), _Handler)
    print(f'Server running  ->  http://127.0.0.1:{PORT}')
    print('Press Ctrl+C to stop.\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')
