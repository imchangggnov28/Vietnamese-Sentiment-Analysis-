import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from models import PhoBERT_BILSTM_Attention


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HỆ THỐNG PHÂN LOẠI CẢM XÚC VĂN BẢN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# FUTURISTIC DASHBOARD CSS
# =========================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 50% 42%,
            rgba(0, 190, 255, 0.08),
            transparent 30%
        ),
        linear-gradient(
            rgba(0, 190, 255, 0.045) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 190, 255, 0.045) 1px,
            transparent 1px
        ),
        #06121a;

    background-size:
        auto,
        42px 42px,
        42px 42px,
        auto;

    color: #dcefff;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 25px !important;
    padding-bottom: 60px !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

footer {
    visibility: hidden;
}


/* =========================================================
   HEADER
   ========================================================= */

.ai-header {
    position: relative;
    padding: 25px 36px;
    margin-bottom: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(8, 31, 43, 0.97),
            rgba(4, 18, 27, 0.94)
        );

    border: 1px solid rgba(0, 204, 255, 0.38);

    box-shadow:
        0 0 28px rgba(0, 180, 255, 0.08),
        inset 0 0 35px rgba(0, 160, 255, 0.035);

    clip-path: polygon(
        0 0,
        98% 0,
        100% 22%,
        100% 100%,
        2% 100%,
        0 78%
    );
}

.ai-header::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 5px;
    height: 100%;
    background: #00c8ff;
    box-shadow: 0 0 18px #00c8ff;
}

.ai-header-small {
    color: #00c8ff;
    font-family: monospace;
    font-size: 12px;
    letter-spacing: 3px;
    margin-bottom: 7px;
    text-transform: uppercase;
}

.ai-header-title {
    color: #eaf8ff;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 2px;
    line-height: 1.25;
}

.ai-header-subtitle {
    color: #7898a8;
    font-size: 14px;
    margin-top: 8px;
}


/* =========================================================
   STATUS BAR
   ========================================================= */

.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;

    background: rgba(4, 20, 29, 0.9);
    border: 1px solid rgba(0, 180, 255, 0.18);

    padding: 10px 18px;
    margin-bottom: 25px;

    font-family: monospace;
    font-size: 12px;
    color: #6f9bab;
}

.status-online {
    color: #36f1a4;
}

.status-online::before {
    content: "●";
    margin-right: 7px;
    text-shadow: 0 0 8px #36f1a4;
}


/* =========================================================
   SECTION LABEL
   ========================================================= */

.section-label {
    color: #00c8ff;
    font-family: monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 11px;
}

.section-label::before {
    content: "◆";
    margin-right: 8px;
    color: #00c8ff;
    text-shadow: 0 0 8px #00c8ff;
}


/* =========================================================
   DASHBOARD CARD
   ========================================================= */

.dashboard-card {
    background:
        linear-gradient(
            145deg,
            rgba(10, 34, 47, 0.95),
            rgba(4, 19, 28, 0.95)
        );

    border: 1px solid rgba(0, 196, 255, 0.25);
    padding: 22px;
    margin-bottom: 15px;

    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.25),
        inset 0 0 25px rgba(0, 160, 255, 0.025);

    position: relative;
}

.dashboard-card::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    width: 55px;
    height: 1px;
    background: #00c8ff;
    box-shadow: 0 0 12px #00c8ff;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background: rgba(5, 25, 36, 0.92);
    border: 1px solid rgba(0, 190, 255, 0.24);

    padding: 17px;
    min-height: 78px;

    position: relative;
    transition: all 0.25s ease;
}

.metric-card:hover {
    border-color: rgba(0, 210, 255, 0.65);
    box-shadow: 0 0 18px rgba(0, 190, 255, 0.10);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #e9faff;
    font-family: monospace;
}

.metric-label {
    color: #6c93a4;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 3px;
}


/* =========================================================
   RADIO OPTIONS
   ========================================================= */

div[data-testid="stRadio"] > div {
    gap: 14px;
}

div[data-testid="stRadio"] label {
    background:
        linear-gradient(
            145deg,
            rgba(8, 31, 43, 0.95),
            rgba(4, 18, 27, 0.95)
        );

    border: 1px solid rgba(0, 190, 255, 0.24);

    padding: 18px 22px;
    min-height: 74px;

    transition: all 0.25s ease;
    cursor: pointer;
}

div[data-testid="stRadio"] label:hover {
    border-color: #00c8ff;

    background:
        linear-gradient(
            145deg,
            rgba(0, 83, 110, 0.35),
            rgba(4, 24, 34, 0.95)
        );

    box-shadow:
        0 0 20px rgba(0, 200, 255, 0.12);

    transform: translateY(-2px);
}

div[data-testid="stRadio"] label p {
    color: #cceeff !important;
    font-size: 14px;
    font-weight: 600;
}


/* =========================================================
   INPUT
   ========================================================= */

div[data-testid="stTextInput"] input {
    background: #071b26 !important;
    border: 1px solid rgba(0, 190, 255, 0.30) !important;
    color: #e8faff !important;
    border-radius: 2px !important;
    padding: 14px 16px !important;
    font-family: monospace;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #00c8ff !important;
    box-shadow: 0 0 15px rgba(0, 200, 255, 0.15) !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #557785 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    background:
        linear-gradient(
            90deg,
            #06364a,
            #07536a
        ) !important;

    color: #eaffff !important;

    border: 1px solid #00bde8 !important;
    border-radius: 2px !important;

    padding: 13px 20px !important;

    font-family: monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px;

    transition: all 0.25s ease;
}

.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #08718e,
            #009dcc
        ) !important;

    box-shadow: 0 0 20px rgba(0, 200, 255, 0.25);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: scale(0.99);
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

section[data-testid="stFileUploaderDropzone"] {
    background: #071b26 !important;
    border: 1px dashed rgba(0, 200, 255, 0.40) !important;
    border-radius: 2px !important;
    min-height: 150px;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00c8ff !important;
    box-shadow: inset 0 0 20px rgba(0, 200, 255, 0.04);
}


/* =========================================================
   SELECTBOX
   ========================================================= */

div[data-baseweb="select"] > div {
    background: #071b26 !important;
    border-color: rgba(0, 190, 255, 0.30) !important;
    border-radius: 2px !important;
    color: #dff8ff !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 190, 255, 0.20);
}


/* =========================================================
   PROGRESS
   ========================================================= */

div[data-testid="stProgress"] > div > div {
    background: #00c8ff !important;
    box-shadow: 0 0 10px rgba(0, 200, 255, 0.50);
}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {
    background: rgba(5, 28, 39, 0.95) !important;
    border-radius: 2px !important;
}


/* =========================================================
   DOWNLOAD
   ========================================================= */

.stDownloadButton > button {
    width: 100%;

    background: rgba(5, 40, 54, 0.95) !important;
    border: 1px solid #00c8ff !important;
    color: #c9f7ff !important;

    border-radius: 2px !important;
}

.stDownloadButton > button:hover {
    background: #07536a !important;
    box-shadow: 0 0 18px rgba(0, 200, 255, 0.20);
}


/* =========================================================
   TEXT / HEADINGS
   ========================================================= */

label {
    color: #8caebb !important;
}

.stMarkdown p {
    color: #8caebb;
}

h1, h2, h3 {
    color: #eafaff !important;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #041018;
}

::-webkit-scrollbar-thumb {
    background: #07536a;
}

::-webkit-scrollbar-thumb:hover {
    background: #00aacc;
}


/* =========================================================
   FULL-SCREEN EMOTION EFFECT
   ========================================================= */

.emotion-overlay {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;

    z-index: 999999;
    pointer-events: none;
    overflow: hidden;

    animation: overlayFade 5.2s ease-out forwards;
}

@keyframes overlayFade {
    0%   { opacity: 1; }
    72%  { opacity: 1; }
    100% { opacity: 0; visibility: hidden; }
}


/* Falling emoji */

.emotion-item {
    position: absolute;
    top: -80px;

    font-size: 40px;
    line-height: 1;

    animation:
        fallEmoji 4s linear forwards;

    filter:
        drop-shadow(0 4px 8px rgba(0, 0, 0, 0.25));
}

@keyframes fallEmoji {
    0% {
        transform:
            translateY(-100px)
            translateX(0)
            rotate(0deg)
            scale(0.65);

        opacity: 0;
    }

    10% {
        opacity: 1;
    }

    50% {
        transform:
            translateY(50vh)
            translateX(25px)
            rotate(180deg)
            scale(1.12);

        opacity: 1;
    }

    100% {
        transform:
            translateY(112vh)
            translateX(-25px)
            rotate(360deg)
            scale(0.75);

        opacity: 0;
    }
}


/* Positive flash */

.positive-flash {
    position: absolute;
    inset: 0;

    background:
        radial-gradient(
            circle at center,
            rgba(255, 255, 255, 0.32),
            rgba(255, 215, 0, 0.12),
            transparent 66%
        );

    animation: flashPositive 1.3s ease-out;
}

@keyframes flashPositive {
    0%   { opacity: 0; }
    18%  { opacity: 1; }
    100% { opacity: 0; }
}


/* Negative flash */

.negative-flash {
    position: absolute;
    inset: 0;

    background:
        radial-gradient(
            circle at center,
            rgba(255, 50, 50, 0.20),
            rgba(120, 0, 0, 0.12),
            transparent 70%
        );

    animation: flashNegative 1.2s ease-out;
}

@keyframes flashNegative {
    0%   { opacity: 0; }
    18%  { opacity: 1; }
    100% { opacity: 0; }
}


/* Firework */

.firework {
    position: absolute;

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #ffffff;

    animation:
        explode 1.45s ease-out forwards;

    box-shadow:
        0 0 10px #00eaff,
        0 0 25px #00c8ff,
        0 0 45px rgba(0, 200, 255, 0.8);
}

@keyframes explode {
    0% {
        transform: scale(0);
        opacity: 1;
    }

    20% {
        transform: scale(1.3);
        opacity: 1;
    }

    100% {
        transform: scale(28);
        opacity: 0;
    }
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 15px !important;
        padding-right: 15px !important;
    }

    .ai-header {
        padding: 22px 23px;
    }

    .ai-header-title {
        font-size: 25px;
    }

    .ai-header-subtitle {
        font-size: 12px;
    }

    .status-bar {
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
    }

    div[data-testid="stRadio"] > div {
        flex-direction: column;
    }

    div[data-testid="stRadio"] label {
        width: 100%;
    }

    .emotion-item {
        font-size: 30px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="ai-header">

    <div class="ai-header-small">
        AI • NLP • VIETNAMESE SENTIMENT ANALYSIS
    </div>

    <div class="ai-header-title">
        HỆ THỐNG PHÂN LOẠI CẢM XÚC VĂN BẢN
    </div>

    <div class="ai-header-subtitle">
        PhoBERT + BiLSTM + Attention Neural Network
        &nbsp; | &nbsp;
        Explainable AI
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# STATUS BAR
# =========================================================

st.markdown(
    """
<div class="status-bar">

    <span>SYSTEM / SENTIMENT ENGINE</span>

    <span class="status-online">
        AI MODEL ONLINE
    </span>

    <span>LANGUAGE: VI</span>

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base",
        use_fast=False,
    )

    model = PhoBERT_BILSTM_Attention()

    model.load_state_dict(
        torch.load(
            "best_model.pt",
            map_location=torch.device("cpu"),
        )
    )

    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_resources()

except FileNotFoundError:
    st.error(
        "⚠️ Không tìm thấy file 'best_model.pt'. "
        "Vui lòng đặt file trọng số cùng thư mục với app.py."
    )
    st.stop()

except Exception as e:
    st.error(
        f"⚠️ Không thể tải mô hình. Chi tiết: {str(e)}"
    )
    st.stop()


# =========================================================
# MODE SELECTION
# =========================================================

st.markdown(
    '<div class="section-label">SELECT ANALYSIS MODE</div>',
    unsafe_allow_html=True,
)

selection = st.radio(
    "Chọn chế độ phân tích",
    [
        "◉  PHÂN TÍCH 1 CÂU TRỰC TIẾP",
        "▣  PHÂN TÍCH QUA FILE DỮ LIỆU",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# EMOTION EFFECTS
# =========================================================

def show_positive_effect():
    """Pháo hoa + LOVE + LIKE toàn màn hình."""

    emojis = [
        "❤️", "💖", "💕", "💗", "💓",
        "😍", "🥰", "😊", "👍", "👏",
        "🎉", "✨", "🌟", "💝", "💘",
    ]

    html = """
    <div class="emotion-overlay">
        <div class="positive-flash"></div>
    """

    for i, emoji in enumerate(emojis * 2):
        left = (i * 7 + 3) % 97
        delay = (i % 12) * 0.11
        duration = 3.6 + (i % 5) * 0.30

        html += f"""
        <div 
            class="emotion-item"
            style="
                left:{left}%;
                animation-delay:{delay}s;
                animation-duration:{duration}s;
            "
        >
            {emoji}
        </div>
        """

    fireworks = [
        (14, 22, 0.10),
        (31, 34, 0.45),
        (50, 18, 0.85),
        (69, 30, 0.30),
        (86, 21, 1.00),
        (23, 58, 1.25),
        (62, 56, 0.65),
        (82, 62, 1.15),
    ]

    for left, top, delay in fireworks:
        html += f"""
        <div
            class="firework"
            style="
                left:{left}%;
                top:{top}%;
                animation-delay:{delay}s;
            "
        ></div>
        """

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def show_negative_effect():
    """Phẫn nộ + CRY + broken heart toàn màn hình."""

    emojis = [
        "😡", "🤬", "😤", "💢",
        "😭", "😢", "🥺", "💔",
        "😠", "😡", "💢", "😭",
        "😢", "💔", "🤬",
    ]

    html = """
       <div class="emotion-overlay">
           <div class="negative-flash"></div>
    """

    for i, emoji in enumerate(emojis * 2):
        left = (i * 7 + 5) % 96
        delay = (i % 10) * 0.13
        duration = 3.3 + (i % 5) * 0.32

        html += f"""
        <div
            class="emotion-item"
            style="
                left:{left}%;
                animation-delay:{delay}s;
                animation-duration:{duration}s;
            "
        >
            {emoji}
        </div>
        """

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# OPTION 1 - SINGLE TEXT
# =========================================================

if selection == "◉  PHÂN TÍCH 1 CÂU TRỰC TIẾP":

    st.markdown(
        '<div class="section-label">TEXT ANALYSIS TERMINAL</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="dashboard-card">

            <div style="
                color:#00c8ff;
                font-family:monospace;
                font-size:12px;
                letter-spacing:2px;
                margin-bottom:8px;">
                INPUT TEXT
            </div>

            <div style="
                color:#7898a8;
                font-size:13px;">
                Nhập bình luận tiếng Việt để hệ thống
                phân tích cảm xúc bằng mô hình AI.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    user_input = st.text_input(
        "Nội dung văn bản",
        "Sản phẩm dùng rất tốt, giao hàng siêu nhanh!",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Dashboard metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">01</div>
                <div class="metric-label">Input Text</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">64</div>
                <div class="metric-label">Max Tokens</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">AI</div>
                <div class="metric-label">Prediction Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">XAI</div>
                <div class="metric-label">Attention Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀  BẮT ĐẦU PHÂN TÍCH CẢM XÚC"):

        if user_input.strip() == "":
            st.warning("Vui lòng không để trống ô nhập liệu.")

        else:

            # -------------------------------------------------
            # TOKENIZATION
            # -------------------------------------------------

            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                max_length=64,
                padding="max_length",
                truncation=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            tokens = tokenizer.convert_ids_to_tokens(
                input_ids[0].tolist()
            )

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            with torch.no_grad():
                logits, attn_weights = model(
                    input_ids,
                    attention_mask,
                )

            prediction = logits.argmax(dim=1).item()
            probs = torch.softmax(logits, dim=1)

            confidence = probs[0][prediction].item() * 100

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">PREDICTION RESULT</div>',
                unsafe_allow_html=True,
            )

            if prediction == 1:

                # Positive full-screen effect
                show_positive_effect()

                st.success(
                    f"😊  **TÍCH CỰC**   •   "
                    f"Độ tin cậy: **{confidence:.2f}%**"
                )

            else:

                # Negative full-screen effect
                show_negative_effect()

                st.error(
                    f"😡  **TIÊU CỰC**   •   "
                    f"Độ tin cậy: **{confidence:.2f}%**"
                )

            # -------------------------------------------------
            # CONFIDENCE METRICS
            # -------------------------------------------------

            positive_conf = probs[0][1].item() * 100
            negative_conf = probs[0][0].item() * 100

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">
                            {positive_conf:.2f}%
                        </div>
                        <div class="metric-label">
                            Positive Probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_b:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">
                            {negative_conf:.2f}%
                        </div>
                        <div class="metric-label">
                            Negative Probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # XAI
            # -------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">EXPLAINABLE AI / ATTENTION MAP</div>',
                unsafe_allow_html=True,
            )

            valid_indices = [
                i
                for i, token in enumerate(tokens)
                if token not in ["<s>", "</s>", "<pad>"]
            ]

            clean_tokens = [
                tokens[i].replace("_", " ")
                for i in valid_indices
            ]

            try:
                clean_weights = (
                    attn_weights[
                        0,
                        valid_indices,
                        0
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

            except Exception:
                # Fallback nếu attention tensor có shape khác
                clean_weights = (
                    attn_weights[
                        0,
                        valid_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

            fig_height = max(3.5, len(clean_tokens) * 0.35 + 1)

            fig, ax = plt.subplots(
                figsize=(10, fig_height)
            )

            bar_color = (
                "#ff4b4b"
                if prediction == 0
                else "#00d9ff"
            )

            ax.barh(
                clean_tokens,
                clean_weights,
                color=bar_color,
            )

            ax.set_xlabel(
                "Mức độ tập trung của mô hình (Attention Weight)"
            )

            ax.invert_yaxis()

            ax.set_facecolor("#071b26")
            fig.patch.set_facecolor("#06121a")

            ax.tick_params(
                colors="#9fc2d1",
                labelsize=9,
            )

            ax.xaxis.label.set_color("#8caebb")

            for spine in ax.spines.values():
                spine.set_color("#16485d")

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)


# =========================================================
# OPTION 2 - BATCH FILE
# =========================================================

else:

    st.markdown(
        '<div class="section-label">BATCH DATA ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="dashboard-card">

            <div style="
                color:#00c8ff;
                font-family:monospace;
                font-size:12px;
                letter-spacing:2px;">
                DATASET PROCESSING
            </div>

            <div style="
                color:#7898a8;
                font-size:13px;
                margin-top:8px;">
                Tải lên tập dữ liệu CSV hoặc Excel để hệ thống
                tự động phân loại cảm xúc.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn file dữ liệu của bạn tại đây",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        except Exception as e:

            st.error(
                f"⚠️ Không thể đọc file. Chi tiết: {str(e)}"
            )
            st.stop()

        if df.empty:

            st.warning(
                "⚠️ File không có dữ liệu."
            )
            st.stop()

        # -------------------------------------------------
        # DATA PREVIEW
        # -------------------------------------------------

        st.markdown(
            '<div class="section-label">DATA PREVIEW</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(5),
            use_container_width=True,
        )

        # -------------------------------------------------
        # DATASET METRICS
        # -------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {len(df):,}
                    </div>
                    <div class="metric-label">
                        Total Rows
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {len(df.columns)}
                    </div>
                    <div class="metric-label">
                        Columns
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">CSV</div>
                    <div class="metric-label">
                        Supported Format
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">XLSX</div>
                    <div class="metric-label">
                        Supported Format
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # TEXT COLUMN
        # -------------------------------------------------

        st.markdown(
            '<div class="section-label">SELECT TEXT COLUMN</div>',
            unsafe_allow_html=True,
        )

        text_column = st.selectbox(
            "Chọn cột chứa nội dung bình luận cần AI phân tích:",
            df.columns,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # BATCH ANALYSIS
        # -------------------------------------------------

        if st.button("⚡  CHẠY PHÂN TÍCH HÀNG LOẠT"):

            predictions = []
            confidences = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_rows = len(df)

            for idx, text in enumerate(df[text_column]):

                progress = int(
                    ((idx + 1) / total_rows) * 100
                )

                progress_bar.progress(progress)

                status_text.markdown(
                    f"""
                    <div style="
                        font-family:monospace;
                        color:#00c8ff;
                        padding:8px 0;">
                        PROCESSING DATA:
                        {idx + 1:,} / {total_rows:,}
                        &nbsp;&nbsp; [{progress}%]
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Empty value
                if (
                    pd.isna(text)
                    or str(text).strip() == ""
                ):

                    predictions.append(
                        "Không xác định"
                    )

                    confidences.append("0.00%")

                    continue

                # Tokenize
                inputs = tokenizer(
                    str(text),
                    return_tensors="pt",
                    max_length=64,
                    padding="max_length",
                    truncation=True,
                )

                # Predict
                with torch.no_grad():

                    logits, _ = model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                    )

                pred = logits.argmax(
                    dim=1
                ).item()

                prob = torch.softmax(
                    logits,
                    dim=1,
                )

                predictions.append(
                    "Tích cực"
                    if pred == 1
                    else "Tiêu cực"
                )

                confidences.append(
                    f"{prob[0][pred].item() * 100:.2f}%"
                )

            # -------------------------------------------------
            # ADD RESULT
            # -------------------------------------------------

            df["AI Dự Đoán"] = predictions
            df["Độ Tin Cậy"] = confidences

            status_text.markdown(
                """
                <div style="
                    font-family:monospace;
                    color:#36f1a4;
                    padding:8px 0;">
                    ● ANALYSIS COMPLETE
                </div>
                """,
                unsafe_allow_html=True,
            )

            progress_bar.progress(100)

            st.success(
                "🎉 Đã phân tích xong toàn bộ dữ liệu!"
            )

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">SENTIMENT DISTRIBUTION</div>',
                unsafe_allow_html=True,
            )

            result_counts = df[
                "AI Dự Đoán"
            ].value_counts()

            col1, col2 = st.columns(
                [1.25, 1]
            )

            with col1:

                st.dataframe(
                    df[
                        [
                            text_column,
                            "AI Dự Đoán",
                            "Độ Tin Cậy",
                        ]
                    ].head(10),
                    use_container_width=True,
                )

            with col2:

                if not result_counts.empty:

                    fig, ax = plt.subplots(
                        figsize=(5, 4)
                    )

                    labels = result_counts.index.tolist()
                    values = result_counts.values.tolist()

                    # Giữ màu nhất quán với hệ thống:
                    # tích cực = cyan, tiêu cực = đỏ,
                    # không xác định = xám.
                    colors = []

                    for label in labels:

                        if label == "Tích cực":
                            colors.append("#00d9ff")

                        elif label == "Tiêu cực":
                            colors.append("#ff4b4b")

                        else:
                            colors.append("#71838d")

                    ax.pie(
                        values,
                        labels=labels,
                        autopct="%1.1f%%",
                        colors=colors,
                        startangle=90,
                        textprops={
                            "color": "#dcefff",
                            "fontsize": 9,
                        },
                    )

                    ax.set_facecolor(
                        "#06121a"
                    )

                    fig.patch.set_facecolor(
                        "#06121a"
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(fig)

            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">EXPORT RESULT</div>',
                unsafe_allow_html=True,
            )

            csv_output = (
                df.to_csv(
                    index=False,
                    encoding="utf-8-sig",
                )
                .encode("utf-8-sig")
            )

            st.download_button(
                label="📥  TẢI XUỐNG FILE KẾT QUẢ PHÂN TÍCH (.CSV)",
                data=csv_output,
                file_name="ket_qua_cam_xuc_AI.csv",
                mime="text/csv",
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div style="
    text-align:center;
    margin-top:45px;
    padding-top:15px;
    border-top:1px solid rgba(0,190,255,0.10);
    color:#466b79;
    font-family:monospace;
    font-size:10px;
    letter-spacing:1px;">
    SENTIMENT AI ENGINE
    &nbsp; • &nbsp;
    PHOBERT / BILSTM / ATTENTION
    &nbsp; • &nbsp;
    EXPLAINABLE AI
</div>
""",
    unsafe_allow_html=True,
)
