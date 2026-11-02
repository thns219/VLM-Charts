// ==============================================================================
// APP.JS — Giao diện Hội Thoại Hỏi & Đáp Thị Giác VLM chuẩn phong cách Gemini
// 100% Thuần Tiếng Việt · Lưu Lịch Sử Trò Chuyện · Click Xem Lại Ảnh · Tinh Tế
// ==============================================================================

let currentImage = "/api/image?path=charxiv_all_images/validation/chart_00004.png";
let currentImageName = "Biểu đồ Covid (CharXiv)";
let selectedMode = "adaptive";
let modalActiveImage = null;

const MODE_CONFIG = {
  adaptive: {
    name: "Adaptive CoT (Đề xuất)",
    slake: "52.59%",
    charxiv: "37.40%"
  },
  no_cot: {
    name: "No-CoT (Trực tiếp)",
    slake: "50.24%",
    charxiv: "38.80%"
  },
  cot: {
    name: "Standard CoT (Từng bước)",
    slake: "45.90%",
    charxiv: "36.20%"
  }
};

// DOM Elements
const modeNav = document.getElementById("modeNav");
const btnNewChat = document.getElementById("btnNewChat");
const chatMain = document.getElementById("chatMain");
const welcomeContainer = document.getElementById("welcomeContainer");
const suggestionGrid = document.getElementById("suggestionGrid");
const messagesList = document.getElementById("messagesList");

const attachmentPreviewBar = document.getElementById("attachmentPreviewBar");
const attachChip = document.getElementById("attachChip");
const attachThumb = document.getElementById("attachThumb");
const attachName = document.getElementById("attachName");
const btnRemoveAttach = document.getElementById("btnRemoveAttach");

const fileInput = document.getElementById("fileInput");
const btnAttach = document.getElementById("btnAttach");
const promptInput = document.getElementById("promptInput");
const btnSend = document.getElementById("btnSend");

// Lightbox Modal Elements
const imageModal = document.getElementById("imageModal");
const imageModalImg = document.getElementById("imageModalImg");
const imageModalTitle = document.getElementById("imageModalTitle");
const btnModalClose = document.getElementById("btnModalClose");
const btnModalOpenTab = document.getElementById("btnModalOpenTab");
const btnModalUseImage = document.getElementById("btnModalUseImage");

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  updateAttachmentUI();
});

function setupEventListeners() {
  // 1. Chuyển đổi Mode / Hình thức suy luận trên Header
  if (modeNav) {
    modeNav.addEventListener("click", (e) => {
      const pill = e.target.closest(".mode-pill");
      if (!pill) return;
      modeNav.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      selectedMode = pill.dataset.mode || "adaptive";
    });
  }

  // 2. Nút Cuộc trò chuyện mới
  if (btnNewChat) {
    btnNewChat.addEventListener("click", resetToNewChat);
  }

  // 3. Đính kèm ảnh từ máy tính (Sửa lỗi cho phép chọn lại cùng 1 ảnh)
  if (btnAttach && fileInput) {
    btnAttach.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleUploadedFile(e.target.files[0]);
      }
      fileInput.value = ""; // Reset giá trị để sự kiện change luôn kích hoạt
    });
  }

  // 4. Bỏ chọn ảnh đính kèm
  if (btnRemoveAttach) {
    btnRemoveAttach.addEventListener("click", (e) => {
      e.stopPropagation(); // Không kích hoạt click xem ảnh của attachChip
      currentImage = null;
      currentImageName = "";
      if (attachmentPreviewBar) attachmentPreviewBar.style.display = "none";
    });
  }

  // 5. Nhấp vào Chip ảnh đính kèm để xem lại ảnh phóng to
  if (attachChip) {
    attachChip.addEventListener("click", (e) => {
      if (e.target.closest("#btnRemoveAttach")) return;
      if (currentImage) {
        const src = currentImage.startsWith("data:image")
          ? currentImage
          : (currentImage.startsWith("/api/image") ? currentImage : `/api/image?path=${encodeURIComponent(currentImage)}`);
        openImageModal(src, currentImageName || "Hình ảnh đính kèm");
      }
    });
  }

  // 6. Chọn thẻ gợi ý (Suggestion Cards)
  if (suggestionGrid) {
    suggestionGrid.addEventListener("click", (e) => {
      const card = e.target.closest(".suggestion-card");
      if (!card) return;

      const img = card.dataset.img;
      const q = card.dataset.q;
      const cat = card.querySelector(".sugg-category") ? card.querySelector(".sugg-category").textContent : "Ảnh mẫu";

      currentImage = `/api/image?path=${encodeURIComponent(img)}`;
      currentImageName = cat;
      updateAttachmentUI();

      promptInput.value = q;
      autoResizeTextarea();
      submitChat();
    });
  }

  // 7. Gửi câu hỏi
  if (btnSend) {
    btnSend.addEventListener("click", submitChat);
  }

  if (promptInput) {
    promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitChat();
      }
    });
    promptInput.addEventListener("input", autoResizeTextarea);
  }

  // 8. Các sự kiện cho Modal Phóng To Ảnh (Lightbox)
  if (btnModalClose) {
    btnModalClose.addEventListener("click", closeImageModal);
  }

  if (imageModal) {
    imageModal.addEventListener("click", (e) => {
      if (e.target === imageModal) {
        closeImageModal();
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeImageModal();
    }
  });

  if (btnModalOpenTab) {
    btnModalOpenTab.addEventListener("click", () => {
      if (!modalActiveImage || !modalActiveImage.src) return;
      const src = modalActiveImage.src;
      if (src.startsWith("data:image")) {
        const win = window.open();
        if (win) {
          win.document.write(`<img src="${src}" style="max-width:100%;height:auto;display:block;margin:auto;" />`);
          win.document.title = modalActiveImage.title || "Hình ảnh chi tiết";
        }
      } else {
        window.open(src, "_blank");
      }
    });
  }

  if (btnModalUseImage) {
    btnModalUseImage.addEventListener("click", () => {
      if (!modalActiveImage || !modalActiveImage.src) return;
      currentImage = modalActiveImage.src;
      currentImageName = modalActiveImage.title || "Hình ảnh đã chọn";
      updateAttachmentUI();
      closeImageModal();
      if (promptInput) promptInput.focus();
    });
  }
}

// ==============================================================================
// Modal Xem Ảnh (Lightbox)
// ==============================================================================
function openImageModal(imgSrc, title = "Xem hình ảnh chi tiết") {
  if (!imgSrc || !imageModal || !imageModalImg) return;
  modalActiveImage = { src: imgSrc, title: title };
  imageModalImg.src = imgSrc;
  if (imageModalTitle) imageModalTitle.textContent = title;
  imageModal.style.display = "flex";
}

function closeImageModal() {
  if (imageModal) {
    imageModal.style.display = "none";
  }
}

function autoResizeTextarea() {
  if (!promptInput) return;
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + "px";
}

function updateAttachmentUI() {
  if (!currentImage) {
    if (attachmentPreviewBar) attachmentPreviewBar.style.display = "none";
    return;
  }
  if (attachmentPreviewBar) attachmentPreviewBar.style.display = "flex";
  if (attachName) attachName.textContent = currentImageName || "Hình ảnh đính kèm";
  if (attachThumb) {
    if (currentImage.startsWith("data:image") || currentImage.startsWith("/api/image") || currentImage.startsWith("http")) {
      attachThumb.src = currentImage;
    } else {
      attachThumb.src = `/api/image?path=${encodeURIComponent(currentImage)}`;
    }
  }
}

// Xử lý nạp ảnh từ máy tính (Tối ưu canvas để gửi nhanh, không bị đơ)
function handleUploadedFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Vui lòng chọn một tệp hình ảnh hợp lệ (PNG, JPG, WEBP, v.v.).");
    return;
  }

  if (attachmentPreviewBar) attachmentPreviewBar.style.display = "flex";
  if (attachName) attachName.textContent = `Đang xử lý ảnh: ${file.name}...`;

  const reader = new FileReader();
  reader.onload = (e) => {
    const rawDataUrl = e.target.result;
    const img = new Image();
    img.onload = () => {
      // Giới hạn cạnh lớn nhất ở 1600px để gửi nhanh qua mạng mà vẫn giữ trọn vẹn độ nét
      const maxDim = 1600;
      let w = img.width;
      let h = img.height;
      if (w > maxDim || h > maxDim) {
        if (w > h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      currentImage = canvas.toDataURL("image/jpeg", 0.92);
      currentImageName = file.name;
      updateAttachmentUI();
    };
    img.onerror = () => {
      currentImage = rawDataUrl;
      currentImageName = file.name;
      updateAttachmentUI();
    };
    img.src = rawDataUrl;
  };
  reader.readAsDataURL(file);
}

function resetToNewChat() {
  if (messagesList) messagesList.innerHTML = "";
  if (welcomeContainer) welcomeContainer.style.display = "block";
  currentImage = "/api/image?path=charxiv_all_images/validation/chart_00004.png";
  currentImageName = "Biểu đồ Covid (CharXiv)";
  updateAttachmentUI();
  if (promptInput) {
    promptInput.value = "";
    promptInput.style.height = "auto";
    promptInput.focus();
  }
}

// Chuyển đổi Markdown và LaTeX thô thành HTML Tiếng Việt hiển thị đẹp mắt, rõ ràng
function formatResponseToHtml(rawText) {
  if (!rawText) return "---";

  // 1. Loại bỏ các ký tự LaTeX thô
  let text = rawText
    .replace(/\\\(/g, "").replace(/\\\)/g, "")
    .replace(/\$\$/g, "").replace(/\$/g, "")
    .replace(/\\cdot/g, "×")
    .replace(/\\times/g, "×")
    .replace(/\\%/g, "%")
    .replace(/\\le/g, "≤")
    .replace(/\\ge/g, "≥")
    .replace(/\\approx/g, "≈");

  // 2. Chuyển đổi từng dòng
  const lines = text.split("\n");
  const htmlParts = [];

  for (let line of lines) {
    let trimmed = line.trim();
    if (!trimmed) {
      htmlParts.push('<div class="ans-space"></div>');
      continue;
    }

    // Escape HTML cơ bản để an toàn
    let safe = trimmed
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // In đậm **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Code `text`
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Gạch đầu dòng * hoặc -
    if (trimmed.startsWith("* ") || trimmed.startsWith("- ")) {
      const content = safe.replace(/^(\*|-)\s+/, "");
      htmlParts.push(`<div class="ans-bullet"><span class="bullet-dot">•</span><span>${content}</span></div>`);
      continue;
    }

    // Danh sách đánh số 1. 2. 3.
    const numMatch = safe.match(/^(\d+\.)\s+(.*)$/);
    if (numMatch) {
      htmlParts.push(`<div class="ans-bullet"><span class="bullet-num">${numMatch[1]}</span><span>${numMatch[2]}</span></div>`);
      continue;
    }

    htmlParts.push(`<p class="ans-para">${safe}</p>`);
  }

  return htmlParts.join("");
}

// ==============================================================================
// Đưa tin nhắn người dùng vào khung chat (Có thể click vào ảnh để xem lại)
// ==============================================================================
function appendUserTurn(questionText, imageSrc, imageName = "Hình ảnh đính kèm") {
  if (welcomeContainer) welcomeContainer.style.display = "none";

  const turn = document.createElement("div");
  turn.className = "message-turn user-turn";

  const temp = document.createElement("div");
  temp.innerText = questionText;
  const safeText = temp.innerHTML;

  let imgTag = "";
  if (imageSrc) {
    imgTag = `
      <div class="user-img-box" title="Nhấn để xem lại ảnh phóng to">
        <img src="${imageSrc}" class="user-img-preview" alt="Ảnh đính kèm" />
        <span class="img-zoom-tag">Nhấn để phóng to ảnh</span>
      </div>
    `;
  }

  turn.innerHTML = `
    <div class="user-bubble">
      ${imgTag}
      <div class="user-text">${safeText}</div>
    </div>
  `;

  // Thêm sự kiện click để xem lại ảnh phóng to
  const imgBox = turn.querySelector(".user-img-box");
  if (imgBox) {
    imgBox.addEventListener("click", () => {
      openImageModal(imageSrc, imageName);
    });
  }

  messagesList.appendChild(turn);
  scrollToBottom();
}

function appendLoadingTurn(modeName) {
  const loading = document.createElement("div");
  loading.className = "message-turn bot-turn";
  loading.id = "activeLoadingTurn";

  loading.innerHTML = `
    <div class="bot-header-meta">
      <div class="bot-meta-left">
        <span class="meta-mode-badge">${modeName}</span>
      </div>
    </div>
    <div class="loading-bubble">
      <div class="loading-spinner"></div>
      <span>Mô hình đang phân tích điểm ảnh và suy luận trực tiếp...</span>
    </div>
  `;

  messagesList.appendChild(loading);
  scrollToBottom();
}

function appendBotTurn(data, modeCfg) {
  const loading = document.getElementById("activeLoadingTurn");
  if (loading) loading.remove();

  const turn = document.createElement("div");
  turn.className = "message-turn bot-turn";

  const modeName = data.mode_name || modeCfg.name;
  const slake = data.slake_acc || modeCfg.slake;
  const charxiv = data.charxiv_acc || modeCfg.charxiv;
  const latency = data.latency || "---";
  const formattedAnswer = formatResponseToHtml(data.pred || "---");
  const thoughtText = data.thought || "Mô hình đưa ra kết quả trực tiếp dựa trên nhận diện thị giác thời gian thực.";

  turn.innerHTML = `
    <div class="bot-header-meta">
      <div class="bot-meta-left">
        <span class="meta-mode-badge">${modeName}</span>
        <span class="meta-acc-badge">SLAKE: <strong>${slake}</strong> · CharXiv: <strong>${charxiv}</strong></span>
      </div>
      <span class="meta-latency">Thời gian: ${latency}</span>
    </div>
    <div class="bot-bubble">
      <div class="answer-content">
        ${formattedAnswer}
      </div>
      <details class="exp-details">
        <summary class="exp-summary">Giải thích chi tiết từ mô hình</summary>
        <div class="exp-details-body">${thoughtText}</div>
      </details>
    </div>
  `;

  messagesList.appendChild(turn);
  scrollToBottom();
}

function appendBotError(errorMsg) {
  const loading = document.getElementById("activeLoadingTurn");
  if (loading) loading.remove();

  const turn = document.createElement("div");
  turn.className = "message-turn bot-turn";

  turn.innerHTML = `
    <div class="bot-header-meta">
      <span class="meta-mode-badge" style="color:#b91c1c; border-color:#fecaca; background:#fef2f2;">Lỗi xử lý</span>
    </div>
    <div class="error-bubble">
      ${errorMsg}
    </div>
  `;

  messagesList.appendChild(turn);
  scrollToBottom();
}

function scrollToBottom() {
  if (chatMain) {
    chatMain.scrollTop = chatMain.scrollHeight;
  }
}

async function submitChat() {
  let text = promptInput.value.trim();

  if (!currentImage) {
    alert("Vui lòng đính kèm một hình ảnh để mô hình VLM có thể quan sát và trả lời.");
    return;
  }

  // Tự động phân tích nếu người dùng gửi ảnh mà không gõ câu hỏi
  if (!text) {
    text = "Hãy quan sát và phân tích chi tiết nội dung của hình ảnh này.";
  }

  const modeCfg = MODE_CONFIG[selectedMode] || MODE_CONFIG.adaptive;
  const imgSrc = currentImage.startsWith("data:image")
    ? currentImage
    : (currentImage.startsWith("/api/image") ? currentImage : `/api/image?path=${encodeURIComponent(currentImage)}`);

  const imgName = currentImageName || "Hình ảnh đính kèm";

  // 1. Đưa câu hỏi người dùng vào khung chat (cho phép click để xem lại ảnh phóng to)
  appendUserTurn(text, imgSrc, imgName);

  // 2. Hiển thị loading
  appendLoadingTurn(modeCfg.name);

  // Reset input
  promptInput.value = "";
  promptInput.style.height = "auto";
  btnSend.disabled = true;

  try {
    const res = await fetch("/api/vqa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        image: currentImage,
        mode: selectedMode
      })
    });

    if (!res.ok) {
      let errDetail = `Lỗi phản hồi HTTP: ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.error) errDetail = errJson.error;
      } catch (e) {}
      appendBotError(errDetail);
      return;
    }

    const data = await res.json();
    if (data.error) {
      appendBotError(data.error);
      return;
    }

    // 3. Đưa câu trả lời vào khung chat
    appendBotTurn(data, modeCfg);

  } catch (err) {
    console.error("Lỗi VQA:", err);
    appendBotError(`Chi tiết lỗi: ${err.message || "Không thể kết nối đến máy chủ suy luận."}`);
  } finally {
    btnSend.disabled = false;
  }
}
