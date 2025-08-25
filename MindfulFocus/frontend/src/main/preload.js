const { contextBridge, ipcRenderer } = require("electron");

console.log("PRELOAD: starting...");

const api = {
  getBalance: () => ipcRenderer.invoke("get-balance"),
  addTokens: (amt) => ipcRenderer.invoke("add-tokens", amt),
  setApiKey: (key) => ipcRenderer.invoke("set-api-key", key),
  getApiKey: () => ipcRenderer.invoke("get-api-key"),
  clearApiKey: () => ipcRenderer.invoke("clear-api-key"),
};

try {
  contextBridge.exposeInMainWorld("api", api);
  console.log("PRELOAD: exposed api:", Object.keys(api));
} catch (err) {
  console.error("PRELOAD: exposeInMainWorld failed:", err);
}

console.log("PRELOAD: ready");