/* ============ 猫狗分类器 前端逻辑 ============ */
(() => {
  "use strict";

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const statusEl = $("model-status");
  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const previewBox = $("preview-box");
  const previewImg = $("preview-img");
  const previewClear = $("preview-clear");
  const btnPredict = $("btn-predict");
  const loading = $("loading");
  const resultEmpty = $("result-empty");
  const resultBox = $("result-box");
  const resultError = $("result-error");
  const resultEmoji = $("result-emoji");
  const resultLabel = $("result-label");
  const resultConf = $("result-conf");
  const historyList = $("history-list");

  const video = $("camera-video");
  const canvas = $("camera-canvas");
  const cameraCover = document.querySelector(".camera-cover");
  const btnCameraStart = $("camera-start");
  const btnCameraCapture = $("camera-capture");
  const btnCameraClose = $("camera-close");

  let currentFile = null;      // 当前待识别图片 (File)
  let cameraStream = null;
  const history = [];

  // ---------- Tab 切换 ----------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // ---------- 模型状态检查 ----------
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.model_trained) {
        statusEl.textContent = `模型就绪 · ${data.model_name}`;
        statusEl.className = "status-badge ready";
      } else {
        statusEl.textContent = "模型未训练";
        statusEl.className = "status-badge error";
        showError("⚠️ 模型尚未训练。\n请先将数据集放入 datasets 目录（文件名形如 cat.1.jpg / dog.1.jpg），再运行 python backend/train.py 训练生成 best_model.pth 后重启服务。");
      }
      return data.model_trained;
    } catch (e) {
      statusEl.textContent = "服务不可达";
      statusEl.className = "status-badge error";
      return false;
    }
  }

  // ---------- 选择图片 ----------
  function setImage(file) {
    if (!file || !file.type.startsWith("image/")) return;
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewBox.hidden = false;
      btnPredict.disabled = false;
      resultError.hidden = true;
    };
    reader.readAsDataURL(file);
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => setImage(e.target.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setImage(file);
  });

  previewClear.addEventListener("click", () => {
    currentFile = null;
    previewBox.hidden = true;
    previewImg.src = "";
    btnPredict.disabled = true;
  });

  // ---------- 摄像头 ----------
  btnCameraStart.addEventListener("click", async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 } },
        audio: false,
      });
      video.srcObject = cameraStream;
      await video.play();
      cameraCover.style.display = "none";
      btnCameraCapture.disabled = false;
      btnCameraClose.hidden = false;
      btnCameraStart.disabled = true;
    } catch (e) {
      showError("无法访问摄像头：请检查浏览器权限设置，或使用上传图片功能。");
    }
  });

  btnCameraCapture.addEventListener("click", () => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `webcam_${Date.now()}.jpg`, { type: "image/jpeg" });
      setImage(file);
    }, "image/jpeg", 0.92);
  });

  btnCameraClose.addEventListener("click", () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    video.srcObject = null;
    cameraCover.style.display = "flex";
    btnCameraCapture.disabled = true;
    btnCameraStart.disabled = false;
    btnCameraClose.hidden = true;
  });

  // ---------- 预测 ----------
  btnPredict.addEventListener("click", async () => {
    if (!currentFile) return;
    loading.hidden = false;
    resultBox.hidden = true;
    resultError.hidden = true;
    btnPredict.disabled = true;

    try {
      const fd = new FormData();
      fd.append("file", currentFile, currentFile.name);
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);

      renderResult(data);
      history.unshift({
        emoji: data.class_index === 0 ? "🐱" : "🐶",
        name: data.label,
        conf: data.confidence,
        src: previewImg.src,
      });
      if (history.length > 6) history.pop();
      renderHistory();
    } catch (e) {
      showError(e.message || "识别失败，请稍后重试。");
    } finally {
      loading.hidden = true;
      btnPredict.disabled = !currentFile;
    }
  });

  function renderResult(data) {
    const isCat = data.class_index === 0;
    resultEmoji.textContent = isCat ? "🐱" : "🐶";
    resultLabel.textContent = data.label;
    resultConf.textContent = (data.confidence * 100).toFixed(2) + "%";
    $("prob-val-cat").textContent = (data.probabilities[0] * 100).toFixed(1) + "%";
    $("prob-val-dog").textContent = (data.probabilities[1] * 100).toFixed(1) + "%";
    $("prob-fill-cat").style.width = (data.probabilities[0] * 100) + "%";
    $("prob-fill-dog").style.width = (data.probabilities[1] * 100) + "%";
    $("result-time").textContent =
      new Date().toLocaleTimeString("zh-CN") +
      (data.source ? ` · ${data.source}` : "");
    resultEmpty.hidden = true;
    resultBox.hidden = false;
  }

  function showError(msg) {
    resultError.textContent = msg;
    resultError.hidden = false;
    resultBox.hidden = true;
    resultEmpty.hidden = false;
  }

  // ---------- 历史 ----------
  function renderHistory() {
    if (!history.length) {
      historyList.innerHTML = '<li class="history-empty">暂无记录</li>';
      return;
    }
    historyList.innerHTML = history
      .map((h, i) => `<li onclick="showHistoryImg(${i})">
          <span class="history-emoji">${h.emoji}</span>
          <span class="history-name">${h.name}</span>
          <span class="history-conf">${(h.conf * 100).toFixed(1)}%</span>
        </li>`)
      .join("");
  }
  window.showHistoryImg = (i) => {
    const h = history[i];
    previewImg.src = h.src;
    previewBox.hidden = false;
    btnPredict.disabled = false;
  };

  // ---------- 初始化 ----------
  renderHistory();
  checkHealth();
})();
