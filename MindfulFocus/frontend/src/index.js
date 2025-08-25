import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  const video = document.getElementById("camera");
  if (video) video.srcObject = stream;
});
