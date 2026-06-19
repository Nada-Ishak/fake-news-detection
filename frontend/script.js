const API_BASE_URL = "http://localhost:8000";

/* =========================
  Elements
========================= */

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

const confidenceValue =
  document.getElementById("confidence-value");

const probFake =
  document.getElementById("prob-fake");

const probReal =
  document.getElementById("prob-real");

const fakePct =
  document.getElementById("fake-pct");

const realPct =
  document.getElementById("real-pct");

const caseNumber =
  document.getElementById("case-number");

const recommendationText =
  document.querySelector(".tips-card p");

/* =========================
  Init
========================= */

caseNumber.textContent =
  "#" +
  Math.floor(
    1000 + Math.random() * 9000
  );

showState("idle");

/* =========================
  Character Counter
========================= */

textInput.addEventListener(
  "input",
  () => {

    const count =
      textInput.value.length;

    charCount.textContent =
      `${count.toLocaleString()} characters`;

  }
);

/* =========================
  State Handling
========================= */

function showState(state) {

  idleState.hidden =
    state !== "idle";

  loadingState.hidden =
    state !== "loading";

  resultState.hidden =
    state !== "result";
}

function setError(message) {

  if (!message) {

    errorMsg.hidden = true;
    errorMsg.textContent = "";

    return;
  }

  errorMsg.hidden = false;

  errorMsg.textContent =
    message;
}

/* =========================
  Toast Notifications
========================= */

function showToast(
  message,
  type = "success"
) {

  const toast =
    document.createElement("div");

  toast.className =
    `toast ${type}`;

  toast.textContent =
    message;

  document.body.appendChild(
    toast
  );

  requestAnimationFrame(() => {
    toast.classList.add("show");
  });

  setTimeout(() => {

    toast.classList.remove(
      "show"
    );

    setTimeout(() => {

      toast.remove();

    }, 300);

  }, 3000);
}

/* =========================
  Recommendation Text
========================= */

function updateRecommendation(
  isFake,
  confidence
) {

  const percentage =
    Math.round(
      confidence * 100
    );

  if (isFake) {

    recommendationText.textContent =
      `The article shows suspicious patterns. Confidence: ${percentage}%. Verify information through trusted news sources before sharing.`;

  } else {

    recommendationText.textContent =
      `The article appears credible according to the model. Confidence: ${percentage}%. Continue validating information using reliable references.`;
  }
}

/* =========================
  Button Loading
========================= */

function setButtonLoading(
  loading
) {

  if (loading) {

    verifyBtn.disabled = true;

    verifyBtn.dataset.original =
      verifyBtn.textContent;

    verifyBtn.textContent =
      "Analyzing...";

  } else {

    verifyBtn.disabled = false;

    verifyBtn.textContent =
      verifyBtn.dataset.original ||
      "Analyze Article";
  }
}

/* =========================
  Result Rendering
========================= */

function renderResult(data) {

  const isFake =
    data.is_fake;

  const confidence =
    Math.round(
      data.confidence * 100
    );

  const fakeProbability =
    Math.round(
      data.fake_probability * 100
    );

  const realProbability =
    Math.round(
      data.real_probability * 100
    );

  stamp.classList.remove(
    "is-fake"
  );

  void stamp.offsetWidth;

  stampText.textContent =
    isFake
      ? "FAKE"
      : "REAL";

  if (isFake) {

    stamp.classList.add(
      "is-fake"
    );
  }

  confidenceValue.textContent =
    confidence + "%";

  fakePct.textContent =
    fakeProbability + "%";

  realPct.textContent =
    realProbability + "%";

  probFake.style.width =
    fakeProbability + "%";

  probReal.style.width =
    realProbability + "%";

  updateRecommendation(
    isFake,
    data.confidence
  );

  showState("result");

  showToast(
    "Analysis completed successfully",
    "success"
  );
}

/* =========================
   Verification
========================= */

async function runVerification() {

  const text =
    textInput.value.trim();

  setError(null);

  if (!text) {

    setError(
      "Please enter article content before analysis."
    );

    textInput.focus();

    return;
  }

  showState("loading");

  setButtonLoading(true);

  try {

    const response =
      await fetch(
        `${API_BASE_URL}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({

            title:
              titleInput.value.trim(),

            text,
          }),
        }
      );

    if (!response.ok) {

      const detail =
        await response
          .json()
          .catch(() => ({}));

      throw new Error(
        detail.detail ||
        `Server Error (${response.status})`
      );
    }

    const data =
      await response.json();

    renderResult(data);

  } catch (error) {

    console.error(error);

    showState("idle");

    setError(
      error.message ===
      "Failed to fetch"
        ? "Unable to connect to backend server."
        : error.message
    );

    showToast(
      "Analysis failed",
      "error"
    );

  } finally {

    setButtonLoading(false);
  }
}

/* =========================
  Events
========================= */

verifyBtn.addEventListener(
  "click",
  runVerification
);

textInput.addEventListener(
  "keydown",
  (e) => {

    if (
      (e.ctrlKey ||
      e.metaKey) &&
      e.key === "Enter"
    ) {

      runVerification();
    }
  }
);

/* =========================
  Entrance Animation
========================= */

window.addEventListener(
  "load",
  () => {

    document.body.classList.add(
      "loaded"
    );
  }
);