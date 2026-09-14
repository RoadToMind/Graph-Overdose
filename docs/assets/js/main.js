"use strict";

const navigationToggle = document.querySelector(".nav-toggle");
const siteNavigation = document.querySelector(".site-navigation");
const navigationLinks = document.querySelectorAll(".site-navigation a");
const scrollProgress = document.querySelector("#scroll-progress");
const sectionElements = document.querySelectorAll("main section[id]");

function closeNavigation() {
  if (!navigationToggle || !siteNavigation) {
    return;
  }

  navigationToggle.setAttribute("aria-expanded", "false");
  navigationToggle.setAttribute("aria-label", "Open navigation");
  siteNavigation.classList.remove("is-open");
  document.body.classList.remove("menu-open");
}

function toggleNavigation() {
  if (!navigationToggle || !siteNavigation) {
    return;
  }

  const isOpen =
    navigationToggle.getAttribute("aria-expanded") === "true";

  navigationToggle.setAttribute(
    "aria-expanded",
    String(!isOpen)
  );

  navigationToggle.setAttribute(
    "aria-label",
    isOpen ? "Open navigation" : "Close navigation"
  );

  siteNavigation.classList.toggle("is-open", !isOpen);
  document.body.classList.toggle("menu-open", !isOpen);
}

function updateScrollProgress() {
  if (!scrollProgress) {
    return;
  }

  const scrollableHeight =
    document.documentElement.scrollHeight - window.innerHeight;

  const progress =
    scrollableHeight > 0
      ? window.scrollY / scrollableHeight
      : 0;

  scrollProgress.style.transform =
    `scaleX(${Math.min(Math.max(progress, 0), 1)})`;
}

function setActiveNavigation(sectionId) {
  navigationLinks.forEach((link) => {
    const href = link.getAttribute("href") || "";
    const isActive = href.endsWith(`#${sectionId}`);

    link.classList.toggle("is-active", isActive);

    if (isActive) {
      link.setAttribute("aria-current", "location");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

navigationToggle?.addEventListener("click", toggleNavigation);

navigationLinks.forEach((link) => {
  link.addEventListener("click", closeNavigation);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeNavigation();
  }
});

window.addEventListener(
  "scroll",
  updateScrollProgress,
  { passive: true }
);

window.addEventListener("resize", () => {
  if (window.innerWidth > 820) {
    closeNavigation();
  }
});

if ("IntersectionObserver" in window && sectionElements.length > 0) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const visibleSections = entries
        .filter((entry) => entry.isIntersecting)
        .sort(
          (firstEntry, secondEntry) =>
            secondEntry.intersectionRatio -
            firstEntry.intersectionRatio
        );

      if (visibleSections.length > 0) {
        setActiveNavigation(visibleSections[0].target.id);
      }
    },
    {
      rootMargin: "-25% 0px -60% 0px",
      threshold: [0.1, 0.25, 0.5],
    }
  );

  sectionElements.forEach((section) => {
    sectionObserver.observe(section);
  });
}

updateScrollProgress();
