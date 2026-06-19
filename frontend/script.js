// Point this at wherever the FastAPI backend is running.
const API_BASE_URL = "http://localhost:8000";

const titleInput = document.getElementById("title-input");
const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const verifyBtn = document.getElementById("verify-btn");
const errorMsg = document.getElementById("error-msg");

const idleState = document.getElementById("idle-state");
const loadingState = document.getElementById("loading-state");
const resultState = document.getElementById("result-state");

const stamp = document.getElementById("stamp");
const stampText = document.getElementById("stamp-text");
const confidenceValue = document.getElementById("confidence-value");
const probFake = document.getElementById("prob-fake");
const probReal = document.getElementById("prob-real");
const fakePct = document.getElementById("fake-pct");
const realPct = document.getElementById("real-pct");
const caseNumber = document.getElementById("case-number");

// Cosmetic: a fresh "case number" per visit, like a new file being opened.
caseNumber.textContent = String(Math.floor(100 + Math.random() * 900));

textInput.addEventListener("input", () => {
  const n = textInput.value.length;
  charCount.textContent = `${n.toLocaleString()} character${n === 1 ? "" : "s"}`;
});

function showState(state) {
  idleState.hidden = state !== "idle";
  loadingState.hidden = state !== "loading";
  resultState.hidden = state !== "result";
}

function setError(message) {
  if (!message) {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
    return;
  }
  errorMsg.hidden = false;
  errorMsg.textContent = message;
}

async function runVerification() {
  const text = textInput.value.trim();
  setError(null);

  if (!text) {
    setError("Add the article body before running a verification.");
    textInput.focus();
    return;
  }

  verifyBtn.disabled = true;
  showState("loading");

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: titleInput.value.trim(), text }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Server responded with ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    showState("idle");
    setError(
      err.message === "Failed to fetch"
        ? "Couldn't reach the verification server. Is the backend running at " + API_BASE_URL + "?"
        : err.message
    );
  } finally {
    verifyBtn.disabled = false;
  }
}

function renderResult(data) {
  const isFake = data.is_fake;

  // Restart the stamp-down animation each time.
  stamp.classList.remove("is-fake");
  stamp.style.animation = "none";
  // eslint-disable-next-line no-unused-expressions
  stamp.offsetHeight; // force reflow so the animation restarts
  stamp.style.animation = "";

  stampText.textContent = isFake ? "FAKE" : "REAL";
  stamp.classList.toggle("is-fake", isFake);

  confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;

  const fp = Math.round(data.fake_probability * 100);
  const rp = Math.round(data.real_probability * 100);
  probFake.style.width = `${fp}%`;
  probReal.style.width = `${rp}%`;
  fakePct.textContent = `${fp}%`;
  realPct.textContent = `${rp}%`;

  showState("result");
}

verifyBtn.addEventListener("click", runVerification);

textInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    runVerification();
  }
});
