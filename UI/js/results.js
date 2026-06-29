const filterButtons = document.querySelectorAll(".filter__btn");
const sections = document.querySelectorAll(".results__section");
const emptyMsg = document.getElementById("results-empty");

function applyFilter(filter) {
  sections.forEach((section) => {
    section.hidden = section.dataset.category !== filter;
  });

  if (emptyMsg) {
    const anyVisible = [...sections].some((s) => !s.hidden);
    emptyMsg.hidden = anyVisible;
  }
}

filterButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    filterButtons.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    applyFilter(btn.dataset.filter);
  });
});

// Show only the default (first active) category on load.
const initialBtn =
  document.querySelector(".filter__btn.is-active") || filterButtons[0];
if (initialBtn) applyFilter(initialBtn.dataset.filter);

/* --------------------- Layout mode (tile / spotlight) --------------- */
const TILE_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/></svg>`;
const SPOTLIGHT_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="4" y="5" width="16" height="14" rx="2"/></svg>`;

let layoutMode = "tile";
const sectionRenderers = [];

sections.forEach((section) => {
  const heading = section.querySelector(".results__heading");
  const grid = section.querySelector(".graph-grid");
  if (!heading || !grid) return;

  const cards = [...grid.querySelectorAll(".graph")];

  // Wrap the heading and a layout toggle in a header row.
  const head = document.createElement("div");
  head.className = "results__section-head";
  section.insertBefore(head, heading);
  head.appendChild(heading);

  const toggle = document.createElement("div");
  toggle.className = "layout-toggle";
  toggle.innerHTML = `
    <button type="button" class="layout-btn" data-mode="tile" title="Tile view" aria-label="Tile view">${TILE_ICON}</button>
    <button type="button" class="layout-btn" data-mode="spotlight" title="Spotlight view" aria-label="Spotlight view">${SPOTLIGHT_ICON}</button>`;
  head.appendChild(toggle);

  // Spotlight navigation (prev / counter / next).
  const nav = document.createElement("div");
  nav.className = "spotlight-nav";
  nav.hidden = true;
  nav.innerHTML = `
    <button type="button" class="spotlight-arrow" data-dir="-1" aria-label="Previous graph">&#8249;</button>
    <span class="spotlight-counter"></span>
    <button type="button" class="spotlight-arrow" data-dir="1" aria-label="Next graph">&#8250;</button>`;
  grid.after(nav);

  const counter = nav.querySelector(".spotlight-counter");
  let index = 0;

  function render() {
    const spotlight = layoutMode === "spotlight";
    section.classList.toggle("is-spotlight", spotlight);
    nav.hidden = !spotlight;

    toggle.querySelectorAll(".layout-btn").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.mode === layoutMode);
    });

    if (spotlight) {
      if (index < 0) index = cards.length - 1;
      if (index >= cards.length) index = 0;
      cards.forEach((c, i) => (c.hidden = i !== index));
      counter.textContent = `${index + 1} / ${cards.length}`;
    } else {
      cards.forEach((c) => (c.hidden = false));
    }
  }

  toggle.addEventListener("click", (e) => {
    const btn = e.target.closest(".layout-btn");
    if (btn) setLayoutMode(btn.dataset.mode);
  });

  nav.addEventListener("click", (e) => {
    const arrow = e.target.closest(".spotlight-arrow");
    if (!arrow) return;
    index += parseInt(arrow.dataset.dir, 10);
    render();
  });

  sectionRenderers.push(render);
  render();
});

function setLayoutMode(mode) {
  layoutMode = mode;
  sectionRenderers.forEach((render) => render());
}

/* ----------------------------- Lightbox ----------------------------- */
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCap = document.getElementById("lightbox-cap");
const lightboxClose = document.getElementById("lightbox-close");

function openLightbox(src, caption, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt || "";
  lightboxCap.textContent = caption || "";
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.src = "";
  document.body.style.overflow = "";
}

document.querySelectorAll(".graph").forEach((card) => {
  card.addEventListener("click", () => {
    const img = card.querySelector(".graph__img");
    const cap = card.querySelector(".graph__cap");
    if (img) openLightbox(img.src, cap ? cap.textContent : "", img.alt);
  });
});

if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);

if (lightbox) {
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
});
