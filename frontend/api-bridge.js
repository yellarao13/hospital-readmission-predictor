(() => {
  const form = document.querySelector("#predictForm");
  if (!form) {
    console.error("HealthAI form #predictForm was not found.");
    return;
  }

  const aliases = {
    age: ["age"],
    gender: ["gender", "sex"],
    admission_type: ["admissiontype", "admission_type"],
    diagnosis: ["diagnosis"],
    time_in_hospital: ["timeinhospital", "time_in_hospital", "hospitaldays"],
    previous_admissions: ["previousadmissions", "previous_admissions"],
    num_lab_procedures: ["numlabprocedures", "num_lab_procedures", "labprocedures"],
    num_medications: ["nummedications", "num_medications", "medications"],
    diabetes: ["diabetes"],
    medication_change: ["medicationchange", "medication_change"],
    discharge_disposition: ["dischargedisposition", "discharge_disposition"]
  };

  const normalize = s => (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");

  function findValue(feature, controls) {
    const candidates = aliases[feature] || [feature];
    for (const el of controls) {
      const identifiers = [
        el.name, el.id, el.getAttribute("aria-label"),
        el.labels ? [...el.labels].map(l => l.textContent).join(" ") : ""
      ].map(normalize);

      if (identifiers.some(id => candidates.includes(id))) {
        return el.value.trim();
      }
    }
    return undefined;
  }

  let resultBox = document.getElementById("apiPredictionResult");
  if (!resultBox) {
    resultBox = document.createElement("section");
    resultBox.id = "apiPredictionResult";
    resultBox.style.cssText =
      "margin:18px 0;padding:18px;border:2px solid #2563eb;border-radius:14px;background:#fff;color:#0f1f3d;display:none";
    form.insertAdjacentElement("afterend", resultBox);
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    event.stopImmediatePropagation();

    const controls = [...form.querySelectorAll("input, select, textarea")]
      .filter(el => !["hidden", "submit", "button", "reset"].includes(el.type)
        && !el.disabled);

    const patient = {};
    const missing = [];

    for (const feature of Object.keys(aliases)) {
      const value = findValue(feature, controls);
      if (value === undefined || value === "") missing.push(feature);
      else patient[feature] = value;
    }

    if (missing.length) {
      resultBox.style.display = "block";
      resultBox.textContent =
        "Cannot map these form fields to the model: " + missing.join(", ") +
        ". Check the form field IDs/names in the HTML.";
      resultBox.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    const button = form.querySelector('button[type="submit"], input[type="submit"]');
    const oldText = button ? button.textContent : "";
    if (button) {
      button.disabled = true;
      if (button.tagName === "BUTTON") button.textContent = "Predicting...";
    }

    resultBox.style.display = "block";
    resultBox.textContent = "Sending patient details to the trained ML model...";
    resultBox.scrollIntoView({ behavior: "smooth", block: "center" });

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(
        data.error + (data.missing ? ": " + data.missing.join(", ") : "")
      );

      resultBox.innerHTML = "";
      const title = document.createElement("h3");
      title.textContent = data.result;
      const score = document.createElement("p");
      score.textContent = "Model-estimated readmission probability: " +
        data.risk_percent + "%";
      const threshold = document.createElement("p");
      threshold.textContent = "Model threshold: " +
        (data.threshold * 100).toFixed(1) + "%";
      const note = document.createElement("small");
      note.textContent = data.note;
      resultBox.append(title, score, threshold, note);
    } catch (error) {
      resultBox.textContent = "Prediction error: " + error.message;
    } finally {
      if (button) {
        button.disabled = false;
        if (button.tagName === "BUTTON") button.textContent = oldText || "Predict";
      }
    }
  }, true);
})();
