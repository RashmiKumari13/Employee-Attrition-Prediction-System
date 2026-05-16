import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function toLabel(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .trim();
}

function endpoint(path) {
  return `${API_BASE}${path}`;
}

function fallbackSummary(prediction) {
  const factors =
    prediction?.explainability?.top_contributors
      ?.filter((item) => item.direction === "increase_risk")
      ?.slice(0, 2)
      ?.map((item) => toLabel(item.feature)) || [];

  if (prediction?.prediction === 1 && factors.length) {
    return `Employee likely to leave due to ${factors.join(" + ")}.`;
  }

  if (prediction?.prediction === 1) {
    return "Employee likely to leave; attrition risk is elevated.";
  }

  return "Employee currently shows lower attrition risk.";
}

function fallbackRecommendations(prediction) {
  if (Array.isArray(prediction?.recommendations) && prediction.recommendations.length > 0) {
    return prediction.recommendations;
  }

  return [
    "Schedule proactive manager check-ins and monitor employee engagement monthly.",
    "Review workload distribution and work-life balance indicators.",
    "Continue career development conversations to maintain retention."
  ];
}

function FeatureInput({ feature, value, onChange }) {
  const label = toLabel(feature.name);

  if (feature.kind === "numeric") {
    return (
      <label className="field-card">
        <span>{label}</span>
        <input
          type="number"
          value={value ?? ""}
          min={feature.min}
          max={feature.max}
          step={feature.dtype === "integer" ? 1 : 0.01}
          onChange={(event) => onChange(feature.name, event.target.value)}
        />
      </label>
    );
  }

  return (
    <label className="field-card">
      <span>{label}</span>
      <select value={value ?? ""} onChange={(event) => onChange(feature.name, event.target.value)}>
        {feature.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [features, setFeatures] = useState([]);
  const [defaults, setDefaults] = useState({});
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [predictLoading, setPredictLoading] = useState(false);
  const [clientReportLoading, setClientReportLoading] = useState(false);
  const [serverReportLoading, setServerReportLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function bootstrap() {
      setBootstrapLoading(true);
      setError("");

      try {
        const [infoResponse, featureResponse] = await Promise.all([
          fetch(endpoint("/api/model-info")),
          fetch(endpoint("/api/features"))
        ]);

        const infoData = await infoResponse.json().catch(() => ({}));
        const featureData = await featureResponse.json().catch(() => ({}));

        if (!infoResponse.ok || !featureResponse.ok) {
          const message =
            infoData.message || featureData.message || "Backend is up but model artifacts are missing.";
          throw new Error(message);
        }

        const loadedFeatures = featureData.features || [];
        const loadedDefaults = Object.fromEntries(
          loadedFeatures.map((feature) => [feature.name, feature.default])
        );

        setModelInfo(infoData);
        setFeatures(loadedFeatures);
        setDefaults(loadedDefaults);
        setFormData(loadedDefaults);
      } catch (bootstrapError) {
        setError(bootstrapError.message);
      } finally {
        setBootstrapLoading(false);
      }
    }

    bootstrap();
  }, []);

  const maxImpact = useMemo(() => {
    const values = prediction?.explainability?.top_contributors || [];
    if (!values.length) {
      return 1;
    }

    return Math.max(...values.map((item) => Math.abs(item.shap_value)));
  }, [prediction]);

  function updateField(name, value) {
    setFormData((current) => ({ ...current, [name]: value }));
  }

  function resetDefaults() {
    setFormData(defaults);
    setPrediction(null);
    setError("");
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setPredictLoading(true);
    setError("");

    try {
      const response = await fetch(endpoint("/api/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed.");
      }

      setPrediction(payload);
    } catch (predictionError) {
      setError(predictionError.message);
    } finally {
      setPredictLoading(false);
    }
  }

  async function downloadServerReport() {
    if (!prediction) {
      return;
    }

    setServerReportLoading(true);
    setError("");

    try {
      const response = await fetch(endpoint("/api/report"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prediction.input_features || formData)
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "Could not generate server PDF report.");
      }

      const blob = await response.blob();
      const fileUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = fileUrl;
      link.download = `attrition_explainability_report_${Date.now()}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(fileUrl);
    } catch (downloadError) {
      setError(downloadError.message);
    } finally {
      setServerReportLoading(false);
    }
  }

  async function downloadClientReport() {
    if (!prediction) {
      return;
    }

    setClientReportLoading(true);
    setError("");

    try {
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ unit: "pt", format: "a4" });
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 38;
      const contentWidth = pageWidth - margin * 2;
      const reportSummary = prediction.summary || fallbackSummary(prediction);
      const recommendations = fallbackRecommendations(prediction);
      const contributors = prediction?.explainability?.top_contributors || [];
      const maxAbs = Math.max(...contributors.map((item) => Math.abs(item.shap_value)), 1);
      const generatedAt = new Date().toLocaleString();
      let y = 38;

      const ensureSpace = (spaceNeeded) => {
        if (y + spaceNeeded > pageHeight - 40) {
          doc.addPage();
          y = 38;
        }
      };

      const drawSectionTitle = (titleText) => {
        ensureSpace(24);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(12);
        doc.setTextColor(59, 43, 36);
        doc.text(titleText, margin, y);
        y += 18;
      };

      const drawWrappedText = (text, options = {}) => {
        const fontSize = options.fontSize ?? 10;
        const indent = options.indent ?? 0;
        const lineHeight = options.lineHeight ?? 14;
        const color = options.color ?? [76, 62, 54];
        const maxWidth = options.maxWidth ?? contentWidth - indent;
        const lines = doc.splitTextToSize(String(text), maxWidth);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(fontSize);
        doc.setTextColor(color[0], color[1], color[2]);

        lines.forEach((line) => {
          ensureSpace(lineHeight);
          doc.text(line, margin + indent, y);
          y += lineHeight;
        });
      };

      doc.setFillColor(64, 34, 24);
      doc.rect(margin, y, contentWidth, 74, "F");
      doc.setTextColor(255, 248, 240);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.text("Employee Attrition Explainability Report", margin + 14, y + 28);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`Generated: ${generatedAt}`, margin + 14, y + 46);
      doc.text("Source: React jsPDF Export", margin + 14, y + 62);
      y += 92;

      ensureSpace(74);
      doc.setFillColor(255, 243, 232);
      doc.rect(margin, y, contentWidth, 62, "F");
      doc.setTextColor(59, 43, 36);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text("Risk Score", margin + 12, y + 24);
      doc.setFontSize(22);
      doc.text(`${prediction.attrition_percent}%`, margin + 12, y + 50);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.text(`(${prediction.risk_label})`, margin + 120, y + 47);
      y += 80;

      drawSectionTitle("Explainability Summary");
      drawWrappedText(reportSummary, { lineHeight: 14 });
      y += 4;

      drawSectionTitle("Top Contributing Factors");
      if (contributors.length === 0) {
        drawWrappedText("- No SHAP contributors available.", { indent: 4 });
      } else {
        contributors.slice(0, 6).forEach((item) => {
          const direction = item.direction === "increase_risk" ? "increase risk" : "decrease risk";
          const value = Number(item.shap_value);
          const prefix = value > 0 ? "+" : "";
          drawWrappedText(`- ${toLabel(item.feature)}: ${prefix}${value.toFixed(4)} (${direction})`, {
            indent: 4
          });
        });
      }
      y += 6;

      drawSectionTitle("SHAP Contributor Graph");
      if (contributors.length === 0) {
        drawWrappedText("No graph data available.", { indent: 4 });
      } else {
        const rowHeight = 18;
        const chartLeft = margin + 150;
        const chartWidth = contentWidth - 210;
        const centerX = chartLeft + chartWidth / 2;
        ensureSpace(rowHeight * contributors.slice(0, 8).length + 24);

        doc.setDrawColor(200, 183, 174);
        doc.line(centerX, y + 5, centerX, y + rowHeight * contributors.slice(0, 8).length + 1);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(9);
        doc.setTextColor(110, 95, 87);
        doc.text("Risk Down", chartLeft, y - 2);
        doc.text("Risk Up", chartLeft + chartWidth - 36, y - 2);

        contributors.slice(0, 8).forEach((item, index) => {
          const value = Number(item.shap_value);
          const labelY = y + index * rowHeight + 8;
          const barY = labelY - 6;
          const maxHalf = chartWidth / 2 - 6;
          const barWidth = Math.max((Math.abs(value) / maxAbs) * maxHalf, 4);
          const startX = value >= 0 ? centerX : centerX - barWidth;
          const label = toLabel(item.feature).slice(0, 22);

          doc.setTextColor(76, 62, 54);
          doc.text(label, chartLeft - 8, labelY, { align: "right" });
          if (value >= 0) {
            doc.setFillColor(179, 38, 30);
          } else {
            doc.setFillColor(47, 127, 95);
          }
          doc.rect(startX, barY, barWidth, 8, "F");

          doc.setTextColor(76, 62, 54);
          const prefix = value > 0 ? "+" : "";
          doc.text(`${prefix}${value.toFixed(4)}`, chartLeft + chartWidth + 8, labelY);
        });

        y += rowHeight * contributors.slice(0, 8).length + 16;
      }

      drawSectionTitle("Retention Recommendations");
      recommendations.slice(0, 6).forEach((recommendation) => {
        drawWrappedText(`- ${recommendation}`, { indent: 4 });
      });

      doc.save(`attrition_explainability_report_${Date.now()}.pdf`);
    } catch (reportError) {
      setError(reportError.message || "Could not generate jsPDF report.");
    } finally {
      setClientReportLoading(false);
    }
  }

  const testAccuracy = modelInfo?.metrics?.test_accuracy;

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="kicker">Full Stack AI + Explainability</p>
        <h1>Employee Attrition Prediction System</h1>
        <p>
          Flask REST API + React dashboard using a Gradient Boosting model on IBM HR dataset with
          SHAP-based contributors.
        </p>
      </header>

      <section className="stats-grid">
        <article className="stat-card">
          <span>Model</span>
          <strong>{modelInfo?.algorithm || "GradientBoostingClassifier"}</strong>
        </article>
        <article className="stat-card">
          <span>Target Accuracy</span>
          <strong>89%</strong>
        </article>
        <article className="stat-card">
          <span>Current Test Accuracy</span>
          <strong>{testAccuracy ? `${(testAccuracy * 100).toFixed(2)}%` : "--"}</strong>
        </article>
      </section>

      {error && <section className="alert">{error}</section>}

      <main className="layout">
        <section className="panel">
          <div className="panel-header">
            <h2>Employee Profile</h2>
            <p>Adjust values and call the `/api/predict` REST endpoint.</p>
          </div>

          {bootstrapLoading ? (
            <p className="muted">Loading model schema...</p>
          ) : (
            <form onSubmit={submitPrediction}>
              <div className="form-grid">
                {features.map((feature) => (
                  <FeatureInput
                    key={feature.name}
                    feature={feature}
                    value={formData[feature.name]}
                    onChange={updateField}
                  />
                ))}
              </div>

              <div className="button-row">
                <button type="submit" disabled={predictLoading || features.length === 0}>
                  {predictLoading ? "Predicting..." : "Predict Attrition Risk"}
                </button>
                <button type="button" className="secondary" onClick={resetDefaults}>
                  Reset Defaults
                </button>
              </div>
            </form>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Prediction + SHAP Explanation</h2>
            <p>Probability, risk label, and top feature contributors.</p>
          </div>

          {!prediction ? (
            <p className="muted">Submit an employee profile to view risk score and explainability.</p>
          ) : (
            <>
              <div className="risk-card">
                <p className="risk-label">{prediction.risk_label}</p>
                <h3>{prediction.attrition_percent}%</h3>
                <p>Estimated probability of attrition</p>
                <div className="risk-bar">
                  <div style={{ width: `${prediction.attrition_percent}%` }} />
                </div>
              </div>

              <section className="summary-card">
                <h3>Explainability Summary</h3>
                <p>{prediction.summary || fallbackSummary(prediction)}</p>
              </section>

              <div className="impact-list">
                {(prediction.explainability?.top_contributors || []).map((item) => {
                  const width = `${Math.max((Math.abs(item.shap_value) / maxImpact) * 100, 8)}%`;
                  const down = item.direction === "decrease_risk";

                  return (
                    <article key={item.feature} className="impact-item">
                      <div className="impact-head">
                        <span>{toLabel(item.feature)}</span>
                        <strong>
                          {item.shap_value > 0 ? "+" : ""}
                          {item.shap_value}
                        </strong>
                      </div>
                      <div className={`impact-bar ${down ? "down" : "up"}`}>
                        <div style={{ width }} />
                      </div>
                    </article>
                  );
                })}
              </div>

              <section className="recommendation-card">
                <h3>Retention Recommendations</h3>
                <ul>
                  {fallbackRecommendations(prediction).map((recommendation) => (
                    <li key={recommendation}>{recommendation}</li>
                  ))}
                </ul>
              </section>

              <div className="report-row">
                <button type="button" onClick={downloadClientReport} disabled={clientReportLoading}>
                  {clientReportLoading ? "Creating jsPDF..." : "Download PDF (jsPDF)"}
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={downloadServerReport}
                  disabled={serverReportLoading}
                >
                  {serverReportLoading ? "Creating ReportLab PDF..." : "Download PDF (ReportLab)"}
                </button>
              </div>

              {prediction.warnings?.length > 0 && (
                <section className="warnings">
                  {prediction.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </section>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
