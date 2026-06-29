const uploadBtn = document.getElementById("upload-btn");
const scanBtn = document.getElementById("scan-btn");
const reuploadBtn = document.getElementById("reupload-btn");
const leafInput = document.getElementById("leaf-input");
const leafPreview = document.getElementById("leaf-preview");
const leafPreviewWrap = document.getElementById("leaf-preview-wrap");
const fileNameBox = document.getElementById("file-name-box");
const fileNameText = document.getElementById("file-name-text");
const formatNote = document.getElementById("format-note");
const resultCard = document.getElementById("result-card");
const resultBody = document.getElementById("result-body");

// When the page is opened directly from disk (file://) we cannot reach the
// model, so fall back to the local server address. When served by tmt_main.py
// the origin already points at the right place.
const API_BASE =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : window.location.origin;

let hasUploaded = false;
let selectedFile = null;

function showScanActions(fileName) {
  hasUploaded = true;
  uploadBtn.hidden = true;
  scanBtn.hidden = false;
  reuploadBtn.hidden = false;

  if (leafPreviewWrap) leafPreviewWrap.classList.add("is-uploaded");
  if (fileNameText) fileNameText.textContent = fileName;
  if (fileNameBox) fileNameBox.hidden = false;
  if (formatNote) formatNote.hidden = true;
}

function previewImage(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    leafPreview.src = event.target.result;
    leafPreview.alt = "Uploaded tomato leaf";
    showScanActions(file.name || "Pasted image");
  };
  reader.readAsDataURL(file);
}

if (uploadBtn && scanBtn && reuploadBtn && leafInput && leafPreview) {
  uploadBtn.addEventListener("click", () => {
    leafInput.click();
  });

  reuploadBtn.addEventListener("click", () => {
    leafInput.click();
  });

  scanBtn.addEventListener("click", () => {
    handleScan();
  });

  leafInput.addEventListener("change", () => {
    const file = leafInput.files?.[0];
    if (!file) return;
    acceptFile(file);
    leafInput.value = "";
  });

  // Paste an image from the clipboard (Ctrl+V / Cmd+V).
  document.addEventListener("paste", (event) => {
    const items = event.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          event.preventDefault();
          acceptFile(file);
        }
        break;
      }
    }
  });
}

function acceptFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please use an image file (JPG, JPEG, or PNG).");
    return;
  }
  selectedFile = file;
  previewImage(file);
  if (resultCard) resultCard.hidden = true;
}

async function handleScan() {
  if (!hasUploaded || !selectedFile) return;

  setScanning(true);
  if (resultCard) resultCard.hidden = true;

  try {
    const base64 = await toBase64(selectedFile);
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Prediction failed.");
    } else {
      showResult(data);
    }
  } catch (err) {
    showError(
      "Could not reach the model server. Make sure tmt_main.py is running."
    );
  } finally {
    setScanning(false);
  }
}

function setScanning(isScanning) {
  if (!scanBtn) return;
  scanBtn.disabled = isScanning;
  scanBtn.textContent = isScanning ? "Scanning…" : "Scan";
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function classIcon(name) {
  const n = name.toLowerCase();
  if (n.includes("healthy")) return "&#129001;"; // green square
  if (n.includes("bacterial")) return "&#129440;"; // microbe
  if (n.includes("mold")) return "&#127811;"; // mushroom
  if (n.includes("mosaic") || n.includes("virus")) return "&#9888;&#65039;";
  return "&#127813;"; // tomato
}

function showResult(data) {
  const probs = data.all_probabilities || {};
  const topName = data.predicted_class;
  const conf = data.confidence;

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  const barsHtml = sorted
    .map(([name, pct]) => {
      const isTop = name === topName;
      return `
        <div class="bar-row">
          <div class="bar-name" title="${name}">${name.replace(/_/g, " ")}</div>
          <div class="bar-track">
            <div class="bar-fill ${isTop ? "top" : "normal"}"
                 data-pct="${pct}" style="width:0%"></div>
          </div>
          <div class="bar-pct">${pct.toFixed(1)}%</div>
        </div>`;
    })
    .join("");

  resultBody.innerHTML = `
    <div class="result-top">
      <div class="result-icon">${classIcon(topName)}</div>
      <div>
        <div class="result-label">Predicted Class</div>
        <div class="result-class">${topName.replace(/_/g, " ")}</div>
        <div class="result-confidence">Confidence: ${conf.toFixed(1)}%</div>
      </div>
    </div>
    <div class="bars-title">All Class Probabilities</div>
    ${barsHtml}`;

  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = el.dataset.pct + "%";
    });
  });
}

function showError(msg) {
  resultBody.innerHTML = `<div class="error-box">&#9888; ${msg}</div>`;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}
