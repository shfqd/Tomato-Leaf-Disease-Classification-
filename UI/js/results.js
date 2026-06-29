const filterButtons = document.querySelectorAll(".filter__btn");
const sections = document.querySelectorAll(".results__section");
const emptyMsg = document.getElementById("results-empty");

function applyFilter(filter) {
  sections.forEach((section) => {
    let visibleInSection = 0;

    section.querySelectorAll(".graph").forEach((card) => {
      const split = card.dataset.split;
      // "all"-tagged cards (e.g. comparison) always show; otherwise match split.
      const show = filter === "all" || split === "all" || split === filter;
      card.hidden = !show;
      if (show) visibleInSection += 1;
    });

    section.hidden = visibleInSection === 0;
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
