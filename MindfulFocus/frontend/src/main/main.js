const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const Database = require("better-sqlite3");

let mainWindow;
let db;
let ephemeralApiKey = null; // memory-only key when MF_EPHEMERAL_API=1

function initializeDatabase() {
  try {
    const dbPath = path.join(app.getPath("userData"), "tokens.db");
    console.log("Database path:", dbPath);

    db = new Database(dbPath);
    db.prepare(
      "CREATE TABLE IF NOT EXISTS tokens (user_id TEXT PRIMARY KEY, balance INTEGER DEFAULT 0)"
    ).run();
    db.prepare(
      "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)"
    ).run();
    db.prepare("INSERT OR IGNORE INTO tokens(user_id, balance) VALUES(?, ?)").run(
      "default",
      0
    );

    return true;
  } catch (error) {
    console.error("Database init failed:", error);
    return false;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      // ✅ Properly load the preload script and keep isolation on
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../../app/index.html"));
  }
}

app.whenReady().then(() => {
  if (initializeDatabase()) {
    createWindow();
    setupIpcHandlers();
  } else {
    app.quit();
  }
});

function setupIpcHandlers() {
  ipcMain.handle("get-balance", () => {
    try {
      const result = db
        .prepare("SELECT balance FROM tokens WHERE user_id=?")
        .get("default");
      return result ? result.balance : 0;
    } catch (error) {
      console.error("Error getting balance:", error);
      return 0;
    }
  });

  ipcMain.handle("add-tokens", (_, amount) => {
    try {
      db.prepare("UPDATE tokens SET balance = balance + ? WHERE user_id=?").run(
        amount,
        "default"
      );
      const result = db
        .prepare("SELECT balance FROM tokens WHERE user_id=?")
        .get("default");
      return result ? result.balance : 0;
    } catch (error) {
      console.error("Error adding tokens:", error);
      return 0;
    }
  });

  ipcMain.handle("set-api-key", (_, apiKey) => {
    try {
      if (process.env.MF_EPHEMERAL_API === "1") {
        // memory-only
        ephemeralApiKey = apiKey;
        return true;
      }
      db.prepare("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)").run(
        "gemini_api_key",
        apiKey
      );
      return true;
    } catch (error) {
      console.error("Error saving API key:", error);
      return false;
    }
  });

  ipcMain.handle("get-api-key", () => {
    try {
      if (process.env.MF_EPHEMERAL_API === "1") {
        return ephemeralApiKey;
      }
      const result = db
        .prepare("SELECT value FROM settings WHERE key=?")
        .get("gemini_api_key");
      return result ? result.value : null;
    } catch (error) {
      console.error("Error getting API key:", error);
      return null;
    }
  });

  ipcMain.handle("clear-api-key", () => {
    try {
      if (process.env.MF_EPHEMERAL_API === "1") {
        ephemeralApiKey = null;
        return true;
      }
      db.prepare("DELETE FROM settings WHERE key=?").run("gemini_api_key");
      return true;
    } catch (e) {
      console.error("Error clearing API key:", e);
      return false;
    }
  });
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    if (db) db.close();
    app.quit();
  }
});