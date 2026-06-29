const uploadBtn = document.getElementById("upload-btn");
const scanBtn = document.getElementById("scan-btn");
const reuploadBtn = document.getElementById("reupload-btn");
const leafInput = document.getElementById("leaf-input");
const leafPreview = document.getElementById("leaf-preview");
const leafPreviewWrap = document.getElementById("leaf-preview-wrap");
const fileNameBox = document.getElementById("file-name-box");
const fileNameText = document.getElementById("file-name-text");
const formatNote = document.getElementById("format-note");

let hasUploaded = false;

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
    showScanActions(file.name);
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
    previewImage(file);
    leafInput.value = "";
  });
}

function handleScan() {
  if (!hasUploaded) return;
  // Ready for classification logic / API call
  console.log("Scanning leaf image...");
}
