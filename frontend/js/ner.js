/**
 * NER Demo JavaScript
 */

document.addEventListener("DOMContentLoaded", function () {
  // Sample texts
  const samples = {
    news: `ឧត្តមសេនីយ៍ឯក ជួន ណារិន្ទ ដឹកនាំ បើក កិច្ចប្រជុំ បូកសរុបលទ្ធផលប្រតិបត្តិការរក្សាសន្តិសុខ សណ្តាប់ធ្នាប់ សាធារណៈ សុវត្ថិភាព សង្គម ឆ្នាំ ២០២៥ និង លើក ទិសដៅ របស់ ស្នងការដ្ឋាននគរបាល រាជធានី ភ្នំពេញ`,

    government: `សម្តេចអគ្គមហាសេនាបតីតេជោ ហ៊ុន សែន នាយករដ្ឋមន្ត្រីនៃព្រះរាជាណាចក្រកម្ពុជា បានអញ្ជើញជាអធិបតីក្នុងពិធីសម្ពោធដាក់ឲ្យប្រើប្រាស់ផ្លូវថ្នល់បំបែកព្រំដែនកម្ពុជា-វៀតណាម ក្នុងខេត្តតាកែវ`,

    tourism: `មណីយដ្ឋានធម្មជាតិ ឆ្នេរខ្សាច់កោះប៉ែន និង ស្ពានប្រដឺសឬស្សី១.០០០ម៉ែត្រ សូមស្វាគមន៍ភ្ញៀវទេសចរណ៍ ទៅកាន់ខេត្តកែប ដើម្បីទស្សនាទេសភាពស្រុកខ្មែរដ៏ស្រស់បំព្រង`,
  };

  // DOM Elements
  const textInput = document.getElementById("khmerTextInput");
  const charCount = document.getElementById("charCount");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const sampleBtns = document.querySelectorAll(".sample-btn");
  const formatBtns = document.querySelectorAll(".format-btn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  // Fallback: prefer an existing `resultsContainer` if present (used by inline page), otherwise create a harmless fallback
  const resultsContainer =
    document.getElementById("resultsContainer") ||
    document.createElement("div");
  const visualResults =
    document.getElementById("visualResults") || resultsContainer;
  const jsonResults =
    document.getElementById("jsonResults") || resultsContainer;
  const textResults =
    document.getElementById("textResults") || resultsContainer;
  const jsonOutput = document.getElementById("jsonOutput") || resultsContainer;
  const textOutput = document.getElementById("textOutput") || resultsContainer;
  const entitySummary =
    document.getElementById("entitySummary") ||
    (function () {
      const el = document.createElement("div");
      el.id = "entitySummary";
      return el;
    })();
  const entityCounts =
    document.getElementById("entityCounts") ||
    (function () {
      const el = document.createElement("div");
      el.id = "entityCounts";
      return el;
    })();
  const performanceMetrics =
    document.getElementById("performanceMetrics") ||
    (function () {
      const el = document.createElement("div");
      el.id = "performanceMetrics";
      return el;
    })();
  const inferenceTime =
    document.getElementById("inferenceTime") ||
    (function () {
      const el = document.createElement("span");
      el.id = "inferenceTime";
      return el;
    })();
  const totalEntities =
    document.getElementById("totalEntities") ||
    (function () {
      const el = document.createElement("span");
      el.id = "totalEntities";
      return el;
    })();
  const apiStatusDot =
    document.getElementById("apiStatusDot") ||
    (function () {
      const el = document.createElement("span");
      el.id = "apiStatusDot";
      return el;
    })();
  const apiStatusText =
    document.getElementById("apiStatusText") ||
    (function () {
      const el = document.createElement("span");
      el.id = "apiStatusText";
      return el;
    })();
  const lastUpdate =
    document.getElementById("lastUpdate") ||
    (function () {
      const el = document.createElement("span");
      el.id = "lastUpdate";
      return el;
    })();

  let currentFormat = "html";
  let lastPrediction = null;

  // Initialize
  updateCharCount();
  checkAPIStatus();

  // Event Listeners
  textInput.addEventListener("input", updateCharCount);

  predictBtn.addEventListener("click", analyzeText);

  clearBtn.addEventListener("click", function () {
    textInput.value = "";
    updateCharCount();
    hideAllResults();
  });

  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      const sampleType = this.dataset.sample;
      textInput.value = samples[sampleType] || "";
      updateCharCount();
    });
  });

  formatBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      formatBtns.forEach((b) => b.classList.remove("active"));
      this.classList.add("active");
      currentFormat = this.dataset.format;
      showResultsInFormat(currentFormat);
    });
  });

  textInput.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key === "Enter") {
      analyzeText();
    }
  });

  // Functions
  function updateCharCount() {
    const count = textInput.value.length;
    charCount.textContent = count;
  }

  function checkAPIStatus() {
    fetch("/api/v1/health")
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "healthy") {
          apiStatusDot.classList.add("active");
          apiStatusText.textContent = "Online";
          apiStatusDot.style.backgroundColor = "#28a745";
        } else {
          apiStatusText.textContent = "Offline";
          apiStatusDot.style.backgroundColor = "#dc3545";
        }
        lastUpdate.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
      })
      .catch((error) => {
        apiStatusText.textContent = "Error";
        apiStatusDot.style.backgroundColor = "#dc3545";
        console.error("API status check failed:", error);
      });
  }

  async function analyzeText() {
    const text = textInput.value.trim();

    if (!text) {
      alert("សូមបញ្ចូលអត្ថបទខ្មែរ!");
      return;
    }

    // Show loading
    loadingSpinner.style.display = "flex";
    predictBtn.disabled = true;
    predictBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> កំពុងវិភាគ...';

    try {
      const response = await fetch("/api/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          format: currentFormat,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      lastPrediction = data;

      // Display results
      displayResults(data);

      // Update performance metrics
      updatePerformanceMetrics(data);

      // Update entity summary
      updateEntitySummary(data.entities);

      // Save anonymous prediction locally if user is not logged in
      try {
        const token = localStorage.getItem("access_token");
        if (!token) {
          const anonItem = {
            text: text,
            created_at: new Date().toISOString(),
            formatted_output: data.formatted_output || null,
            entities: data.entities || {},
            inference_time_ms: data.inference_time_ms || 0,
          };
          if (window.saveAnonPrediction) window.saveAnonPrediction(anonItem);
          else {
            const arr = JSON.parse(
              localStorage.getItem("anon_history") || "[]"
            );
            arr.unshift(anonItem);
            localStorage.setItem(
              "anon_history",
              JSON.stringify(arr.slice(0, 50))
            );
          }
        }
        if (window.updateHistoryBadge) window.updateHistoryBadge();
      } catch (e) {
        /* ignore */
      }
    } catch (error) {
      console.error("Error analyzing text:", error);
      alert("កំហុសក្នុងការវិភាគអត្ថបទ៖ " + error.message);
    } finally {
      // Hide loading
      loadingSpinner.style.display = "none";
      predictBtn.disabled = false;
      predictBtn.innerHTML = '<i class="fas fa-bolt"></i> Analyze Text';
    }
  }

  function displayResults(data) {
    // Hide all result sections first
    hideAllResults();

    // Show results based on current format
    showResultsInFormat(currentFormat, data);

    // Show metrics and summary
    entitySummary.style.display = "block";
    performanceMetrics.style.display = "grid";
  }

  function hideAllResults() {
    visualResults.style.display = "none";
    jsonResults.style.display = "none";
    textResults.style.display = "none";
    entitySummary.style.display = "none";
    performanceMetrics.style.display = "none";
  }

  function showResultsInFormat(format, data = null) {
    if (!data && lastPrediction) {
      data = lastPrediction;
    }

    if (!data) return;

    hideAllResults();

    switch (format) {
      case "html":
        showVisualResults(data.formatted_output || data);
        break;
      case "json":
        showJSONResults(data);
        break;
      case "text":
        showTextResults(data);
        break;
    }
  }

  function showVisualResults(html) {
    if (visualResults && visualResults.style) {
      visualResults.style.display = "block";
    } else {
      console.warn(
        "visualResults element missing; using resultsContainer fallback."
      );
    }

    // If html is a string, use it directly
    if (typeof html === "string") {
      try {
        visualResults.innerHTML = html;
      } catch (e) {
        // Fallback to placing HTML into resultsContainer or document body
        (resultsContainer || document.body).innerHTML = html;
      }
    } else {
      // Otherwise, generate HTML from results
      const noResults =
        visualResults && visualResults.querySelector
          ? visualResults.querySelector(".no-results")
          : null;
      if (noResults && noResults.style) {
        noResults.style.display = "none";
      }

      // Create entity display
      const entityContainer = document.createElement("div");
      entityContainer.className = "khmer-text";
      entityContainer.style.fontSize = "20px";
      entityContainer.style.lineHeight = "2";

      if (Array.isArray(html.results)) {
        html.results.forEach((result) => {
          const span = document.createElement("span");
          span.textContent = result.token + " ";

          if (result.entity_type !== "O") {
            const colors = {
              PER: "#FF9999",
              ORG: "#99CCFF",
              LOC: "#99FF99",
              MISC: "#FFCC99",
            };

            span.style.backgroundColor =
              colors[result.entity_type] || "#CCCCCC";
            span.style.padding = "2px 4px";
            span.style.margin = "0 1px";
            span.style.borderRadius = "3px";
            span.className = "entity";
            span.setAttribute("data-entity", result.entity_type);
          }

          entityContainer.appendChild(span);
        });
      }

      // Clear existing content and add new
      const existingContent =
        visualResults && visualResults.querySelector
          ? visualResults.querySelector(".khmer-text")
          : null;
      if (existingContent) {
        existingContent.remove();
      }
      try {
        (visualResults || resultsContainer || document.body).appendChild(
          entityContainer
        );
      } catch (e) {
        console.warn(
          "Failed to append entity container to visualResults; inserting into resultsContainer instead.",
          e
        );
        (resultsContainer || document.body).appendChild(entityContainer);
      }
    }
  }

  function showJSONResults(data) {
    jsonResults.style.display = "block";
    jsonOutput.textContent = JSON.stringify(data, null, 2);

    // Apply syntax highlighting
    hljs.highlightElement(jsonOutput);
  }

  function showTextResults(data) {
    textResults.style.display = "block";

    if (typeof data.formatted_output === "string") {
      textOutput.textContent = data.formatted_output;
    } else {
      // Generate text format
      let text = "";
      if (Array.isArray(data.results)) {
        data.results.forEach((result) => {
          text += `${result.token}\t${result.label}\n`;
        });
      }
      textOutput.textContent = text;
    }
  }

  function updatePerformanceMetrics(data) {
    inferenceTime.textContent = `${data.inference_time_ms.toFixed(1)}ms`;

    // Calculate total entities
    let total = 0;
    if (data.entities) {
      Object.values(data.entities).forEach((entities) => {
        total += entities.length;
      });
    }
    totalEntities.textContent = total;
  }

  function updateEntitySummary(entities) {
    if (!entities) return;

    entityCounts.innerHTML = "";

    Object.entries(entities).forEach(([type, entityList]) => {
      const count = entityList.length;

      const item = document.createElement("div");
      item.className = "entity-count-item";

      const value = document.createElement("div");
      value.className = "entity-count-value";
      value.textContent = count;

      const label = document.createElement("div");
      label.className = "entity-count-label";
      label.textContent = type;

      item.appendChild(value);
      item.appendChild(label);
      entityCounts.appendChild(item);
    });

    // Add total count
    const totalCount = Object.values(entities).reduce(
      (sum, list) => sum + list.length,
      0
    );

    const totalItem = document.createElement("div");
    totalItem.className = "entity-count-item";

    const totalValue = document.createElement("div");
    totalValue.className = "entity-count-value";
    totalValue.textContent = totalCount;
    totalValue.style.color = "#667eea";

    const totalLabel = document.createElement("div");
    totalLabel.className = "entity-count-label";
    totalLabel.textContent = "Total";

    totalItem.appendChild(totalValue);
    totalItem.appendChild(totalLabel);
    entityCounts.appendChild(totalItem);
  }

  // Load label mapping and attach mapping UI handlers
  async function loadLabelMapping() {
    try {
      const res = await fetch("/api/v1/model/labels");
      const data = await res.json();
      renderLabelMapping(data.idx2label);
      renderLegendFromMapping(data.idx2label);
    } catch (e) {
      console.warn("Could not load label mapping", e);
    }
  }

  function renderLegendFromMapping(idx2label) {
    // Build a legend area within resultsContainer (if present)
    const legend = document.createElement("div");
    legend.className = "entity-legend dynamic-legend";
    legend.innerHTML =
      '<h5><i class="fas fa-tags"></i> Entity Colors</h5><div class="legend-items" id="dynamicLegendItems"></div>';
    const existing = document.querySelector(".dynamic-legend");
    if (existing) existing.remove();
    const container =
      document.getElementById("resultsContainer") || resultsContainer;
    container.appendChild(legend);

    const itemsDiv = document.getElementById("dynamicLegendItems");
    itemsDiv.innerHTML = "";
    const colors = {
      PER: "#FF9999",
      ORG: "#99CCFF",
      LOC: "#99FF99",
      MISC: "#FFCC99",
    };
    Object.entries(idx2label).forEach(([k, v]) => {
      let entityType = v;
      if (v.startsWith("B-") || v.startsWith("I-")) {
        entityType = v.split("-")[1];
      }
      if (entityType === "O") return;
      const span = document.createElement("span");
      span.className = "legend-item";
      span.style.backgroundColor = colors[entityType] || "#CCCCCC";
      span.style.marginRight = "8px";
      span.style.padding = "4px 6px";
      span.textContent = `${entityType}`;
      itemsDiv.appendChild(span);
    });
  }

  function renderLabelMapping(mapping) {
    const list = document.getElementById("labelMappingList");
    list.innerHTML = "";
    Object.keys(mapping)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((k) => {
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.gap = "8px";
        row.style.alignItems = "center";
        row.style.marginBottom = "6px";
        row.innerHTML = `<div style="width:40px;">${k}</div><input data-idx="${k}" value="${mapping[k]}" style="flex:1;padding:6px;" />`;
        list.appendChild(row);
      });
  }

  document
    .getElementById("openMappingBtn")
    .addEventListener("click", function () {
      document.getElementById("labelMappingModal").style.display = "block";
      loadLabelMapping();
    });

  document
    .getElementById("closeMappingBtn")
    .addEventListener("click", function () {
      document.getElementById("labelMappingModal").style.display = "none";
    });

  document
    .getElementById("saveMappingBtn")
    .addEventListener("click", async function () {
      const inputs = Array.from(
        document.querySelectorAll("#labelMappingList input")
      );
      const payload = { idx2label: {} };
      inputs.forEach(
        (i) => (payload.idx2label[i.dataset.idx] = i.value.trim())
      );
      try {
        const res = await fetch("/api/v1/model/labels", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        document.getElementById("labelMappingStatus").innerHTML =
          '<div style="color:green">Saved mapping successfully</div>';
        renderLegendFromMapping(data.idx2label);
        // Also reload model to ensure labels are used everywhere
        await fetch("/api/v1/model/reload", { method: "POST" });
      } catch (e) {
        document.getElementById("labelMappingStatus").innerHTML =
          '<div style="color:red">Error saving mapping: ' +
          e.message +
          "</div>";
      }
    });

  document
    .getElementById("reloadModelBtn")
    .addEventListener("click", async function () {
      try {
        await fetch("/api/v1/model/reload", { method: "POST" });
        document.getElementById("labelMappingStatus").innerHTML =
          '<div style="color:green">Model reloaded</div>';
      } catch (e) {
        document.getElementById("labelMappingStatus").innerHTML =
          '<div style="color:red">Reload failed: ' + e.message + "</div>";
      }
    });

  // Auto-check API status every 30 seconds
  setInterval(checkAPIStatus, 30000);

  // initial load of mapping
  loadLabelMapping();
});
