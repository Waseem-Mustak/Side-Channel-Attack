function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;
      this.showingTraces = true;

      try {
        // Create a worker for sweep counting
        let worker = new Worker("worker.js");

        // Start the sweep and wait for result
        const trace = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Terminate worker
        worker.terminate();

        // Send trace data to backend for heatmap generation
        const response = await fetch("/collect_trace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace })
        });

        if (!response.ok) throw new Error("Failed to send trace to backend");
        const data = await response.json();

        // Add the new heatmap and its stats to the local collection
        if (data.heatmap_url) {
          this.heatmaps.push({
            url: data.heatmap_url,
            min: data.min,
            max: data.max,
            range: data.range,
            samples: data.samples
          });
        }
        this.status = "Trace collected and heatmap generated!";
      } catch (error) {
        console.error("Error collecting trace data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
      try {
        this.status = "Downloading traces...";
        this.statusIsError = false;

        // Fetch the latest data from the backend API
        const response = await fetch("/download_traces");
        if (!response.ok) throw new Error("Failed to fetch traces from server");
        const traces = await response.json();

        // Create a download file with the trace data
        const blob = new Blob([JSON.stringify(traces, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fingerprint_traces_${new Date().toISOString().slice(0,19).replace(/[:]/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        this.status = "Traces downloaded successfully!";
      } catch (error) {
        console.error("Error downloading traces:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      try {
        this.status = "Clearing all results...";
        this.statusIsError = false;

        // Send request to clear all results
        const response = await fetch("/api/clear_results", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Failed to clear results from server");
        const data = await response.json();

        if (data.success) {
          // Clear local copies
          this.heatmaps = [];
          this.traceData = [];
          this.status = "All results cleared successfully!";
        } else {
          throw new Error(data.message || "Failed to clear results");
        }
      } catch (error) {
        console.error("Error clearing results:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },
  };
}
