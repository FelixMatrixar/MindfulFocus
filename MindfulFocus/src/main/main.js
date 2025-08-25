const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const Database = require("better-sqlite3");

// Enable camera/media access
app.commandLine.appendSwitch("enable-features", "MediaStream");

let mainWindow;
const db = new Database(path.join(app.getPath("userData"), "tokens.db"));
db.prepare("CREATE TABLE IF NOT EXISTS tokens (user_id TEXT PRIMARY KEY, balance INTEGER DEFAULT 0)").run();
db.prepare("INSERT OR IGNORE INTO tokens(user_id, balance) VALUES(?, ?)").run("default", 0);

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../../app/index.html"));
  }
}

app.whenReady().then(createWindow);

ipcMain.handle("get-balance", () => {
  return db.prepare("SELECT balance FROM tokens WHERE user_id=?").get("default").balance;
});
ipcMain.handle("add-tokens", (_, amount) => {
  db.prepare("UPDATE tokens SET balance = balance + ? WHERE user_id=?").run(amount, "default");
  return db.prepare("SELECT balance FROM tokens WHERE user_id=?").get("default").balance;
});
