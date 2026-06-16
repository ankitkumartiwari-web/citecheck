const nav = document.querySelector("[data-nav]");
const reveals = document.querySelectorAll(".reveal");
const stage = document.querySelector("[data-depth]");

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("is-visible");
    revealObserver.unobserve(entry.target);
  });
}, { threshold: 0.18 });

reveals.forEach((node, index) => {
  node.style.setProperty("--delay", `${Math.min(index * 45, 220)}ms`);
  revealObserver.observe(node);
});

window.addEventListener("scroll", () => {
  nav.classList.toggle("is-scrolled", window.scrollY > 12);
}, { passive: true });

if (stage && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  stage.addEventListener("pointermove", (event) => {
    const rect = stage.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    const windowCard = stage.querySelector(".window");
    windowCard.style.transform = `rotateX(${4 - y * 5}deg) rotateY(${-7 + x * 6}deg) translateY(-3px)`;
  });

  stage.addEventListener("pointerleave", () => {
    stage.querySelector(".window").style.transform = "";
  });
}
