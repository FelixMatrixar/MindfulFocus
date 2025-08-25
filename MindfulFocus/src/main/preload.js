const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  getBalance: () => ipcRenderer.invoke("get-balance"),
  addTokens: (amt) => ipcRenderer.invoke("add-tokens", amt),
});
