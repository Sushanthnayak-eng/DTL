async function predict() {
  const btn = document.getElementById("predictBtn");
  const result = document.getElementById("result");

  const data = {
    Temperature: parseFloat(Temperature.value),
    MC_percent: parseFloat(MC_percent.value),
    pH: parseFloat(pH.value),
    CN_Ratio: parseFloat(CN_Ratio.value),
    Ammonia_mgkg: parseFloat(Ammonia_mgkg.value),
    Nitrate_mgkg: parseFloat(Nitrate_mgkg.value),
    TN_percent: parseFloat(TN_percent.value),
    TOC_percent: parseFloat(TOC_percent.value),
    EC_mscm: parseFloat(EC_mscm.value),
    OM_percent: parseFloat(OM_percent.value),
    T_Value: parseFloat(T_Value.value),
    GI_percent: parseFloat(GI_percent.value)
  };

  btn.classList.add("loading");
  btn.innerText = "Analyzing...";
  result.classList.add("hidden");

  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const r = await res.json();
    const stageClass = r.stage.toLowerCase();

    result.innerHTML = `
      <h2>🌿 Prediction Result</h2>
      <p><strong>Score:</strong> ${r.score}</p>
      <span class="badge ${stageClass}">${r.stage}</span>
      <p><strong>Days to Maturity:</strong> ${r.days_to_maturity}</p>
      <p><strong>95% Confidence Interval:</strong><br>
         ${r.confidence_interval[0]} – ${r.confidence_interval[1]}</p>
    `;

    result.classList.remove("hidden");

  } catch (e) {
    result.innerHTML = "❌ Unable to connect to backend.";
    result.classList.remove("hidden");
  }

  btn.classList.remove("loading");
  btn.innerText = "Predict Quality";
}
