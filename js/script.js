const CHAT_URL = "https://lydus-chatbot.onrender.com";

// document.addEventListener("DOMContentLoaded", () => {
//   const loginNav = document.getElementById("loginNav");
//   const currentId = localStorage.getItem("loginId");
//   console.log(currentId);

//   if (currentId) {
//     // 로그인 상태 → 아이디와 로그아웃 버튼 표시
//     loginNav.innerHTML = `
//         <span style="color:white; margin-right:10px;">${currentId} 님</span>
//         <button id="logoutBtn" class="btn btn-outline-light">Logout</button>
//       `;

//     // 로그아웃 동작
//     document.getElementById("logoutBtn").addEventListener("click", () => {
//       localStorage.removeItem("loggedInUsers"); // 로그인 해제
//       localStorage.removeItem("loginId"); // 로그인 해제
//       window.location.reload(); // 새로고침해서 로그인 버튼으로 복귀
//     });
//   } else {
//     // 로그인 안 된 상태 → 로그인 버튼
//     loginNav.innerHTML = `
//         <a href="./login.html">
//           <button type="button" class="btn btn-outline-light">Login</button>
//         </a>
//       `;
//   }
// });

// document.getElementById("chatbotImg").addEventListener("click", () => {
//   const chatContainer = document.getElementById("chatContainer");
//   const frame = document.getElementById("chatbotFrame");

//   if (chatContainer.style.display === "block") {
//     // 이미 열려 있으면 닫기
//     chatContainer.style.display = "none";
//   } else {
//     // 닫혀 있으면 열기
//     const fullUrl = `${CHAT_URL}/?embed=true`;
//     frame.src = fullUrl;
//     chatContainer.style.display = "block";
//   }
// });

const chatbotImg = document.getElementById("chatbotImg");
const chatbotClose = document.getElementById("chatbotClose");
const chatContainer = document.getElementById("chatContainer");
const frame = document.getElementById("chatbotFrame");
const chatBubble = document.getElementById("chatBubble"); // 말풍선

// const CHAT_URL = "https://your-chat-url.com"; // 실제 챗봇 주소로 교체

// 챗봇 열기
chatbotImg.addEventListener("click", () => {
  const fullUrl = `${CHAT_URL}/?embed=true`;
  frame.src = fullUrl;
  chatContainer.style.display = "block";

  chatbotImg.style.display = "none"; // 로봇 아이콘 숨기기
  chatbotClose.style.display = "block"; // X 아이콘 보이기
  if (chatBubble) chatBubble.style.display = "none"; // 말풍선 숨기기
});

// 챗봇 닫기
chatbotClose.addEventListener("click", () => {
  chatContainer.style.display = "none";

  chatbotClose.style.display = "none"; // X 아이콘 숨기기
  chatbotImg.style.display = "block"; // 로봇 아이콘 다시 보이기
  if (chatBubble) chatBubble.style.display = "block"; // 말풍선 다시 보이기
});

// 한영 변환
const langToggle = document.getElementById("langToggle");
let currentLang = "ko"; // 기본 한국어

langToggle.addEventListener("click", () => {
  if (currentLang === "ko") {
    // 영어로 전환
    document.querySelector(".about").textContent = "About";
    document.querySelector(".software").textContent = "Software";
    document.querySelector(".data").textContent = "Data";
    document.querySelector(".share").textContent = "Share";
    document.querySelector(".tutorials").textContent = "Tutorials";
    document.querySelector(".guidelines").textContent = "Guidelines";
    document.querySelector(".text-1").textContent = "Leave Your Data to US";
    document.querySelector(".text-2").textContent = "LYDUS";
    document.querySelector(".text-3").textContent =
      "We curate medical data to drive better healthcare. By enhancing the value of complex medical data, we provide deep insights for researchers, doctors, and patients.";
    document.getElementById("chatBubble").textContent = "If you have any questions, ask me!";
    langToggle.textContent = "KO"; // 버튼 텍스트
    currentLang = "en";
  } else {
    // 한국어로 전환
    document.querySelector(".about").textContent = "회사 소개";
    document.querySelector(".software").textContent = "소프트웨어";
    document.querySelector(".data").textContent = "데이터";
    document.querySelector(".share").textContent = "공유";
    document.querySelector(".tutorials").textContent = "튜토리얼";
    document.querySelector(".guidelines").textContent = "가이드라인";
    document.querySelector(".text-1").textContent = "데이터는 저희에게 맡기세요";
    document.querySelector(".text-2").textContent = "LYDUS";
    document.querySelector(".text-3").textContent =
      "저희는 의료 데이터를 큐레이션하여 더 나은 헬스케어를 만듭니다. 복잡한 의료 데이터의 가치를 높여 연구자, 의사, 환자에게 깊이 있는 인사이트를 제공합니다.";
    document.getElementById("chatBubble").textContent = "궁금하신 점이 있다면 질문해보세요!";
    langToggle.textContent = "EN"; // 버튼 텍스트
    currentLang = "ko";
  }
});
