"use strict";

const resultVideo = document.querySelector("#result-video");
const resultVideoSource = resultVideo?.querySelector("source");
const resultVideoCaption = document.querySelector("#result-video-caption");
const resultTabs = document.querySelectorAll(".result-tab");
const copyButtons = document.querySelectorAll(".copy-button");

function selectResultVideo(button) {
  if (!resultVideo || !resultVideoSource || !button.dataset.video) return;
  resultTabs.forEach((tab) => {
    const selected = tab === button;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
  resultVideoSource.src = button.dataset.video;
  if (resultVideoCaption) resultVideoCaption.textContent = button.dataset.caption || "";
  resultVideo.load();
  const playPromise = resultVideo.play();
  if (playPromise !== undefined) playPromise.catch(() => {});
}

async function copyCitation(button) {
  const target = document.getElementById(button.dataset.copyTarget);
  if (!target) return;
  const originalLabel = button.textContent;
  try {
    await navigator.clipboard.writeText(target.textContent.trim());
  } catch {
    const textArea = document.createElement("textarea");
    textArea.value = target.textContent.trim();
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    textArea.remove();
  }
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = originalLabel; }, 1600);
}

resultTabs.forEach((button) => button.addEventListener("click", () => selectResultVideo(button)));
copyButtons.forEach((button) => button.addEventListener("click", () => copyCitation(button)));

if ("IntersectionObserver" in window) {
  const videoObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting && !entry.target.paused) entry.target.pause();
    });
  }, { threshold: 0.05 });
  document.querySelectorAll("video").forEach((video) => videoObserver.observe(video));
}
