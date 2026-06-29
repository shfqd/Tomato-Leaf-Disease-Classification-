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
const uploadCard = document.getElementById("upload-card");
const scanAgainBtn = document.getElementById("scan-again-btn");
const exportPdfBtn = document.getElementById("export-pdf-btn");

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

if (scanAgainBtn) {
  scanAgainBtn.addEventListener("click", showUploadCard);
}

if (exportPdfBtn) {
  exportPdfBtn.addEventListener("click", exportResultPdf);
}

function exportResultPdf() {
  const classEl = resultCard.querySelector(".result-class");
  const name = classEl
    ? classEl.textContent.trim().replace(/\s+/g, "_")
    : "result";

  // The browser's "Save as PDF" uses the document title as the file name.
  const prevTitle = document.title;
  document.title = `tomaleaf_${name}`;

  const restore = () => {
    document.title = prevTitle;
    window.removeEventListener("afterprint", restore);
  };
  window.addEventListener("afterprint", restore);

  window.print();
}

function showResultCard() {
  if (uploadCard) uploadCard.hidden = true;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showUploadCard() {
  resultCard.hidden = true;
  if (uploadCard) uploadCard.hidden = false;
  if (uploadCard) uploadCard.scrollIntoView({ behavior: "smooth", block: "start" });
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

function classImage(name) {
  const n = name.toLowerCase();
  if (n.includes("bacterial")) return "../images/bacterialLeaf.png";
  if (n.includes("mold")) return "../images/moldLeaf.png";
  if (n.includes("mosaic") || n.includes("virus")) return "../images/mosaicLeaf.png";
  if (n.includes("healthy")) return "../images/leaf.png";
  return "../images/leaf.png";
}

// Per-class knowledge base used to enrich the result card.
const DISEASE_INFO = {
  Bacterial_spot: {
    status: "diseased",
    cause: "Bacterial — Xanthomonas spp.",
    symptoms:
      "Small, dark, water-soaked spots on leaves and fruit that enlarge, dry out and may form yellow halos.",
    treatment: [
      "Remove and destroy infected leaves and fruit.",
      "Apply copper-based bactericides early in the season.",
      "Avoid overhead watering — keep foliage dry.",
      "Rotate crops and plant disease-free certified seed.",
    ],
  },
  Leaf_Mold: {
    status: "diseased",
    cause: "Fungal — Passalora fulva (Fulvia fulva)",
    symptoms:
      "Pale yellow patches on upper leaf surfaces with olive-green to brown velvety mold underneath.",
    treatment: [
      "Improve air circulation and reduce humidity.",
      "Remove affected leaves promptly.",
      "Apply a suitable fungicide when needed.",
      "Space plants well and ventilate greenhouses.",
    ],
  },
  Tomato_mosaic_virus: {
    status: "diseased",
    cause: "Viral — Tomato mosaic virus (ToMV)",
    symptoms:
      "Mottled light and dark green patterns, leaf curling, distortion and stunted growth.",
    treatment: [
      "Remove and destroy infected plants — there is no cure.",
      "Disinfect hands and tools; avoid tobacco near plants.",
      "Control aphids and other insect vectors.",
      "Plant resistant varieties and clean seed.",
    ],
  },
  healthy: {
    status: "healthy",
    cause: "No disease detected",
    symptoms: "The leaf looks clean and vigorous with uniform green colour.",
    treatment: [
      "Keep up consistent watering and balanced nutrition.",
      "Monitor regularly for early signs of disease.",
      "Maintain good airflow and avoid wetting the foliage.",
    ],
  },
};

function getDiseaseInfo(name) {
  if (DISEASE_INFO[name]) return DISEASE_INFO[name];
  const n = name.toLowerCase();
  if (n.includes("bacterial")) return DISEASE_INFO.Bacterial_spot;
  if (n.includes("mold")) return DISEASE_INFO.Leaf_Mold;
  if (n.includes("mosaic") || n.includes("virus"))
    return DISEASE_INFO.Tomato_mosaic_virus;
  if (n.includes("healthy")) return DISEASE_INFO.healthy;
  return null;
}

function confidenceLevel(conf) {
  if (conf >= 80) return { label: "High confidence", cls: "high" };
  if (conf >= 50) return { label: "Medium confidence", cls: "medium" };
  return { label: "Low confidence", cls: "low" };
}

function showResult(data) {
  const probs = data.all_probabilities || {};
  const topName = data.predicted_class;
  const conf = data.confidence;

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  const info = getDiseaseInfo(topName);
  const isHealthy = info && info.status === "healthy";
  const level = confidenceLevel(conf);
  const margin = sorted.length > 1 ? sorted[0][1] - sorted[1][1] : conf;
  const uncertain = conf < 60 || margin < 10;

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

  const statusBanner = info
    ? `<div class="status-banner status-banner--${
        isHealthy ? "healthy" : "diseased"
      }">
         <span class="status-banner__dot"></span>
         ${isHealthy ? "Healthy Leaf" : "Disease Detected"}
       </div>`
    : "";

  const warningBlock = uncertain
    ? `<div class="result-warning">
         &#9888; This prediction is uncertain. Try a clearer, well-lit close-up
         of a single leaf for a more reliable result.
       </div>`
    : "";

  const infoBlock = info
    ? `<div class="info-grid">
         <div class="info-item">
           <div class="info-item__label">${isHealthy ? "Status" : "Cause"}</div>
           <div class="info-item__value">${info.cause}</div>
         </div>
         <div class="info-item">
           <div class="info-item__label">Symptoms</div>
           <div class="info-item__value">${info.symptoms}</div>
         </div>
         <div class="info-item info-item--full">
           <div class="info-item__label">${
             isHealthy ? "Care Tips" : "Treatment & Prevention"
           }</div>
           <ul class="info-list">
             ${info.treatment.map((t) => `<li>${t}</li>`).join("")}
           </ul>
         </div>
       </div>`
    : "";

  const uploadedSrc = leafPreview ? leafPreview.src : "";
  const className = topName.replace(/_/g, " ");

  resultBody.innerHTML = `
    <div class="result-hero">
      ${statusBanner}
      <img class="result-uploaded" src="${uploadedSrc}" alt="Uploaded leaf image" />
      <div class="result-id">
        <img class="result-id__icon" src="${classImage(topName)}" alt="${className} leaf" />
        <div class="result-id__text">
          <div class="result-label">Predicted Class</div>
          <div class="result-class">${className}</div>
          <div class="result-confidence">
            Confidence: ${conf.toFixed(1)}%
            <span class="conf-badge conf-badge--${level.cls}">${level.label}</span>
          </div>
        </div>
      </div>
    </div>
    ${warningBlock}
    ${infoBlock}
    <div class="bars-title">All Class Probabilities</div>
    ${barsHtml}`;

  showResultCard();

  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = el.dataset.pct + "%";
    });
  });
}

function showError(msg) {
  resultBody.innerHTML = `<div class="error-box">&#9888; ${msg}</div>`;
  showResultCard();
}
