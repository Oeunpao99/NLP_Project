/**
 * API Utility Functions
 */

class KhmerNERAPI {
  constructor(baseURL = "") {
    this.baseURL = baseURL;
  }

  getAuthHeaders(contentType = "application/json") {
    const headers = {};
    if (contentType) headers["Content-Type"] = contentType;
    const token = localStorage.getItem("access_token");
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return headers;
  }

  async predict(text, format = "json") {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/predict`, {
        method: "POST",
        headers: this.getAuthHeaders("application/json"),
        body: JSON.stringify({
          text: text,
          format: format,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  }

  async batchPredict(texts, format = "json") {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/predict/batch`, {
        method: "POST",
        headers: this.getAuthHeaders("application/json"),
        body: JSON.stringify({
          texts: texts,
          format: format,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Batch prediction error:", error);
      throw error;
    }
  }

  async getHistory(page = 1, pageSize = 10) {
    try {
      const response = await fetch(
        `${this.baseURL}/api/v1/predictions?page=${page}&page_size=${pageSize}`,
        { headers: this.getAuthHeaders(null) }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("History fetch error:", error);
      throw error;
    }
  }

  async getEntityStats(days = 7) {
    try {
      const response = await fetch(
        `${this.baseURL}/api/v1/entities/stats?days=${days}`,
        { headers: this.getAuthHeaders(null) }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Stats fetch error:", error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/health`, {
        headers: this.getAuthHeaders(null),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Health check error:", error);
      throw error;
    }
  }
}

// Create global API instance
window.KhmerNERAPI = KhmerNERAPI;
window.api = new KhmerNERAPI("");
