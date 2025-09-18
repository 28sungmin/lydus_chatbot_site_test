const CHAT_URL = "https://lydus-chatbot.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
  const loginNav = document.getElementById("loginNav");
  const currentId = localStorage.getItem("loginId");
  console.log(currentId);

  if (currentId) {
    // 로그인 상태 → 아이디와 로그아웃 버튼 표시
    loginNav.innerHTML = `
        <span style="color:white; margin-right:10px;">${currentId} 님</span>
        <button id="logoutBtn" class="btn btn-outline-light">Logout</button>
      `;

    // 로그아웃 동작
    document.getElementById("logoutBtn").addEventListener("click", () => {
      localStorage.removeItem("loggedInUsers"); // 로그인 해제
      localStorage.removeItem("loginId"); // 로그인 해제
      window.location.reload(); // 새로고침해서 로그인 버튼으로 복귀
    });
  } else {
    // 로그인 안 된 상태 → 로그인 버튼
    loginNav.innerHTML = `
        <a href="./login.html">
          <button type="button" class="btn btn-outline-light">Login</button>
        </a>
      `;
  }
});

document.getElementById("chatbotImg").addEventListener("click", () => {
  const chatContainer = document.getElementById("chatContainer");
  const frame = document.getElementById("chatbotFrame");

  if (chatContainer.style.display === "block") {
    // 이미 열려 있으면 닫기
    chatContainer.style.display = "none";
  } else {
    // 닫혀 있으면 열기
    const fullUrl = `${CHAT_URL}/?embed=true`;
    frame.src = fullUrl;
    chatContainer.style.display = "block";
  }
});
