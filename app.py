import io
import os
from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

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

# Bảng màu dùng chung cho landing page + main page (cũng dùng cho biểu đồ)
C_BG = "#070817"
C_PANEL = "#0f1030"
C_PRIMARY = "#8b5cf6"
C_ACCENT = "#22d3ee"
C_POS = "#22d3ee"
C_NEG = "#fb4b6b"
C_MUTED = "#a9a5c8"


# =========================================================
# GLOBAL CSS  (1 bảng màu: tím = chủ đạo, cyan = điểm nhấn,
#              cyan = tích cực, đỏ hồng = tiêu cực)
# =========================================================

st.markdown(
    """
<style>

:root {
    --bg: #070817;
    --panel: rgba(18, 17, 48, 0.92);
    --panel-solid: #0f1030;
    --primary: #8b5cf6;
    --primary-2: #a855f7;
    --primary-soft: #c4b5fd;
    --accent: #22d3ee;
    --neg: #fb4b6b;
    --text: #ece9ff;
    --muted: #a9a5c8;
    --dim: #6f6b94;
    --line: rgba(139, 92, 246, 0.30);
    --line-strong: rgba(196, 181, 253, 0.65);
    --glow: rgba(139, 92, 246, 0.40);
    --mono: "Fira Code", Consolas, monospace;
}

/* ---------- GLOBAL ---------- */

.stApp {
    background:
        radial-gradient(circle at 12% 6%, rgba(139, 92, 246, 0.17), transparent 32%),
        radial-gradient(circle at 90% 18%, rgba(34, 211, 238, 0.08), transparent 30%),
        linear-gradient(rgba(139, 92, 246, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(139, 92, 246, 0.05) 1px, transparent 1px),
        var(--bg);
    background-size: auto, auto, 44px 44px, 44px 44px, auto;
    color: var(--text);
}

.block-container {
    max-width: 1450px !important;
    padding-top: 25px !important;
    padding-bottom: 60px !important;
}

[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }

label { color: var(--muted) !important; }
.stMarkdown p { color: var(--muted); }
h1, h2, h3 { color: var(--text) !important; }

/* ---------- SECTION LABEL ---------- */

.section-label {
    color: var(--primary-soft);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 11px;
}

.section-label::before {
    content: "◆";
    margin-right: 8px;
    color: var(--accent);
    text-shadow: 0 0 8px var(--accent);
}

/* ---------- MAIN HEADER ---------- */

.ai-header {
    position: relative;
    padding: 25px 36px;
    margin-bottom: 18px;
    background: linear-gradient(145deg, rgba(20, 18, 54, 0.97), rgba(8, 8, 26, 0.95));
    border: 1px solid var(--line);
    box-shadow:
        0 0 30px rgba(139, 92, 246, 0.10),
        inset 0 0 35px rgba(139, 92, 246, 0.04);
    clip-path: polygon(0 0, 98% 0, 100% 22%, 100% 100%, 2% 100%, 0 78%);
}

.ai-header::before {
    content: "";
    position: absolute;
    left: 0; top: 0;
    width: 5px; height: 100%;
    background: linear-gradient(180deg, var(--primary), var(--accent));
    box-shadow: 0 0 18px var(--primary);
}

.ai-header-small {
    color: var(--primary-soft);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 3px;
    margin-bottom: 7px;
    text-transform: uppercase;
}

.ai-header-title {
    color: var(--text);
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 2px;
    line-height: 1.25;
}

.ai-header-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: 8px;
}

/* ---------- STATUS BAR ---------- */

.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
    background: rgba(12, 12, 36, 0.9);
    border: 1px solid rgba(139, 92, 246, 0.20);
    padding: 10px 18px;
    margin-bottom: 25px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
}

.status-online { color: #36f1a4; }
.status-online::before {
    content: "●";
    margin-right: 7px;
    text-shadow: 0 0 8px #36f1a4;
}

/* ---------- CARDS ---------- */

.dashboard-card {
    position: relative;
    background: linear-gradient(145deg, rgba(22, 20, 58, 0.95), rgba(10, 10, 32, 0.95));
    border: 1px solid var(--line);
    padding: 22px;
    margin-bottom: 15px;
    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.25),
        inset 0 0 25px rgba(139, 92, 246, 0.03);
}

.dashboard-card::after {
    content: "";
    position: absolute;
    right: 0; top: 0;
    width: 55px; height: 1px;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 17px;
    min-height: 78px;
    transition: all 0.25s ease;
}

.metric-card:hover {
    border-color: var(--line-strong);
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.18);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: var(--text);
    font-family: var(--mono);
}

.metric-label {
    color: var(--dim);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 3px;
}

.metric-pos .metric-value { color: var(--accent); }
.metric-neg .metric-value { color: var(--neg); }

/* ---------- RADIO (CHỌN CHẾ ĐỘ) ---------- */

div[data-testid="stRadio"] > div { gap: 14px; }

div[data-testid="stRadio"] label {
    background: linear-gradient(145deg, rgba(20, 18, 54, 0.95), rgba(9, 9, 28, 0.95));
    border: 1px solid var(--line);
    padding: 18px 22px;
    min-height: 74px;
    transition: all 0.25s ease;
    cursor: pointer;
}

div[data-testid="stRadio"] label:hover {
    border-color: var(--primary-soft);
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.18);
}

/* Chế độ đang chọn -> đổi màu */
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(110deg, rgba(109, 40, 217, 0.85), rgba(147, 51, 234, 0.70));
    border-color: var(--primary-soft);
    box-shadow: 0 0 22px var(--glow);
}

div[data-testid="stRadio"] label p {
    color: #d9d4f5 !important;
    font-size: 14px;
    font-weight: 600;
}

div[data-testid="stRadio"] label:has(input:checked) p { color: #ffffff !important; }

/* ---------- TEXT INPUT ---------- */

div[data-testid="stTextInput"] { margin: 8px 0 18px; }

div[data-testid="stTextInput"] label { display: none !important; }

div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
}

div[data-testid="stTextInput"] input {
    height: 58px !important;
    box-sizing: border-box !important;
    background: linear-gradient(180deg, rgba(20, 18, 54, 0.96), rgba(8, 8, 26, 0.98)) !important;
    border: 1px solid var(--line) !important;
    border-bottom: 2px solid var(--primary) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    caret-color: var(--accent) !important;
    padding: 0 18px !important;
    font-family: var(--mono) !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    outline: none !important;
    animation: input-glow 2.6s ease-in-out infinite;
    transition: border-color 0.25s ease !important;
}

div[data-testid="stTextInput"] input::placeholder { color: var(--dim) !important; }

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.25) !important;
    animation: none;
}

@keyframes input-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.10); }
    50%      { box-shadow: 0 0 16px rgba(139, 92, 246, 0.38); }
}

/* ---------- BUTTONS (mặc định = nút hành động chính) ---------- */

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #5b21b6, #7c3aed) !important;
    color: #ffffff !important;
    border: 1px solid var(--primary-soft) !important;
    border-radius: 4px !important;
    padding: 13px 20px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #9333ea) !important;
    box-shadow: 0 0 22px var(--glow);
    transform: translateY(-1px);
}

.stButton > button:active { transform: scale(0.99); }

/* ---------- NAV (Trang chủ / Giới thiệu) ---------- */

/* Nút chưa chọn */
.st-key-nav_home_off button,
.st-key-nav_about_off button {
    min-height: 48px;
    background: rgba(18, 17, 48, 0.85) !important;
    border: 1px solid var(--line) !important;
    color: var(--muted) !important;
    box-shadow: none !important;
}

.st-key-nav_home_off button:hover,
.st-key-nav_about_off button:hover {
    border-color: var(--primary-soft) !important;
    color: #ffffff !important;
    background: rgba(139, 92, 246, 0.14) !important;
}

/* Nút đang chọn / đang xem */
.st-key-nav_home_on button,
.st-key-nav_about_on button {
    min-height: 48px;
    background: linear-gradient(110deg, #6d28d9, #9333ea 55%, #0e9fbd) !important;
    border: 1px solid var(--primary-soft) !important;
    color: #ffffff !important;
    box-shadow: 0 0 22px rgba(139, 92, 246, 0.55) !important;
}

/* ---------- NÚT BẮT ĐẦU (landing + about) ---------- */

.st-key-landing_start button,
.st-key-about_start button {
    min-height: 60px;
    font-size: 15px !important;
    background: linear-gradient(110deg, #6d28d9, #9333ea, #a855f7, #6d28d9) !important;
    background-size: 250% 100% !important;
    border: 1px solid #c084fc !important;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.28);
    animation:
        landing-gradient-flow 4s ease infinite,
        landing-purple-glow 2s ease-in-out infinite;
}

.st-key-landing_start button:hover,
.st-key-about_start button:hover {
    border-color: #e9d5ff !important;
    box-shadow: 0 0 30px rgba(168, 85, 247, 0.60);
    transform: translateY(-3px);
}

@keyframes landing-gradient-flow {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

@keyframes landing-purple-glow {
    0%, 100% { box-shadow: 0 0 12px rgba(168, 85, 247, 0.20); }
    50%      { box-shadow: 0 0 26px rgba(168, 85, 247, 0.48); }
}

/* ---------- NÚT QUAY LẠI (cuối trang) ---------- */

.st-key-back_home button {
    min-height: 52px;
    background: transparent !important;
    border: 1px solid var(--primary) !important;
    color: var(--primary-soft) !important;
}

.st-key-back_home button:hover {
    background: rgba(139, 92, 246, 0.18) !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px var(--glow);
}

/* ---------- FILE UPLOADER / SELECT / TABLE / ALERT ---------- */

section[data-testid="stFileUploaderDropzone"] {
    background: rgba(14, 13, 40, 0.95) !important;
    border: 1px dashed rgba(139, 92, 246, 0.50) !important;
    border-radius: 3px !important;
    min-height: 150px;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    box-shadow: inset 0 0 20px rgba(34, 211, 238, 0.05);
}

div[data-baseweb="select"] > div {
    background: rgba(14, 13, 40, 0.95) !important;
    border-color: var(--line) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
}

div[data-testid="stDataFrame"] { border: 1px solid rgba(139, 92, 246, 0.22); }

div[data-testid="stAlert"] {
    background: rgba(14, 13, 40, 0.95) !important;
    border-radius: 3px !important;
}

.stDownloadButton > button {
    width: 100%;
    background: rgba(14, 13, 40, 0.95) !important;
    border: 1px solid var(--accent) !important;
    color: #c9f7ff !important;
    border-radius: 4px !important;
}

.stDownloadButton > button:hover {
    background: rgba(34, 211, 238, 0.14) !important;
    box-shadow: 0 0 18px rgba(34, 211, 238, 0.25);
}

::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: #05050f; }
::-webkit-scrollbar-thumb { background: #4c1d95; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* =========================================================
   LANDING PAGE
   ========================================================= */

.landing-brand {
    color: var(--primary-soft);
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 1px;
}

.landing-brand span { color: var(--accent); }

.landing-bg {
    position: relative;
    min-height: 430px;
    padding: 30px 25px 40px;
    overflow: hidden;
    color: #fff;
}

.landing-hero { text-align: center; padding: 70px 0 40px; }

.landing-title {
    margin: 0;
    color: var(--text);
    font-size: clamp(34px, 5.5vw, 70px);
    line-height: 1.2;
    font-weight: 900;
    letter-spacing: -1.5px;
}

.landing-gradient {
    background: linear-gradient(90deg, #8b5cf6, #b477ff, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.landing-subtitle {
    max-width: 800px;
    margin: 30px auto 0;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.9;
}

.landing-tech { color: #e3d5ff; font-weight: 700; }

.landing-chips {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 26px;
}

.landing-chip {
    padding: 7px 16px;
    border: 1px solid var(--line);
    background: rgba(139, 92, 246, 0.10);
    color: var(--primary-soft);
    font-family: var(--mono);
    font-size: 12px;
}

.landing-chip.alt {
    border-color: rgba(34, 211, 238, 0.35);
    background: rgba(34, 211, 238, 0.08);
    color: var(--accent);
}

.landing-footer {
    margin-top: 30px;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
}

/* About */

.about-panel {
    padding: 42px;
    margin-bottom: 22px;
    color: var(--text);
    background:
        radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.16), transparent 40%),
        rgba(10, 10, 32, 0.95);
    border: 1px solid var(--line);
}

.about-title {
    margin: 0 0 22px;
    color: var(--text);
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    letter-spacing: 1px;
}

.about-content {
    color: var(--muted);
    font-size: 16px;
    line-height: 1.9;
    margin-bottom: 26px;
}

.model-link {
    display: block;
    text-decoration: none !important;
    color: inherit !important;
    margin-bottom: 14px;
}

.model-card {
    padding: 20px 24px;
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid var(--line);
    transition: all 0.25s ease;
}

.model-link:hover .model-card {
    background: rgba(139, 92, 246, 0.18);
    border-color: var(--primary-soft);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.22);
    transform: translateY(-3px);
}

.model-title {
    color: var(--primary-soft);
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.model-desc { color: var(--muted); font-size: 15px; }

/* =========================================================
   LOADER (cốc cà phê)
   ========================================================= */

.loader-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px 0;
}

.coffee-container { width: 120px; height: 120px; position: relative; }

.coffee-cup {
    width: 80px; height: 60px;
    background: var(--panel-solid);
    border: 5px solid var(--primary);
    border-radius: 0 0 40px 40px;
    position: absolute; bottom: 20px; left: 15px;
    box-shadow: 0 0 15px var(--glow);
    animation: squash-stretch 1.8s ease-in-out infinite;
}

.coffee-handle {
    width: 25px; height: 35px;
    border: 5px solid var(--primary);
    border-left: none;
    border-radius: 0 20px 20px 0;
    position: absolute; right: -25px; top: 5px;
}

.smoke-stream {
    position: absolute;
    width: 6px; height: 25px;
    background: rgba(34, 211, 238, 0.55);
    border-radius: 3px;
    animation: float-up 1.4s linear infinite;
}

.smoke-1 { left: 35px; bottom: 85px; animation-delay: 0s; }
.smoke-2 { left: 52px; bottom: 85px; animation-delay: 0.4s; }
.smoke-3 { left: 68px; bottom: 85px; animation-delay: 0.8s; }

@keyframes squash-stretch {
    0%, 100% { transform: scale(1, 1); }
    50%      { transform: scale(1.08, 0.92); }
}

@keyframes float-up {
    0%   { transform: translateY(0) scaleX(1); opacity: 0; }
    20%  { opacity: 0.8; }
    80%  { opacity: 0.4; }
    100% { transform: translateY(-30px) scaleX(1.5); opacity: 0; }
}

.loader-percent {
    color: var(--accent);
    font-family: var(--mono);
    font-size: 16px;
    font-weight: bold;
    margin-top: 15px;
    letter-spacing: 2px;
}

.loader-sub {
    color: var(--dim);
    font-family: var(--mono);
    font-size: 11px;
    margin-top: 4px;
}

/* =========================================================
   FULL-SCREEN EMOTION EFFECT
   ========================================================= */

.emotion-overlay {
    position: fixed;
    inset: 0;
    width: 100vw; height: 100vh;
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

.emotion-item {
    position: absolute;
    top: -80px;
    font-size: 40px;
    line-height: 1;
    pointer-events: auto !important;
    cursor: pointer !important;
    user-select: none;
    animation: fallEmoji 4s linear forwards;
    filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.25));
    transition: transform 0.18s ease, filter 0.18s ease;
}

.emotion-item:hover {
    filter: drop-shadow(0 0 8px var(--primary)) drop-shadow(0 0 18px rgba(139, 92, 246, 0.55));
}

@keyframes fallEmoji {
    0%   { transform: translateY(-100px) translateX(0) rotate(0deg) scale(0.65); opacity: 0; }
    10%  { opacity: 1; }
    50%  { transform: translateY(50vh) translateX(25px) rotate(180deg) scale(1.12); opacity: 1; }
    100% { transform: translateY(112vh) translateX(-25px) rotate(360deg) scale(0.75); opacity: 0; }
}

.emotion-item.is-dissolving {
    pointer-events: none !important;
    animation: emoji-dissolve 0.7s ease-out forwards !important;
}

@keyframes emoji-dissolve {
    0%   { opacity: 1;  transform: scale(1) rotate(0);        filter: blur(0); }
    30%  { opacity: .9; transform: scale(1.35) rotate(10deg); filter: blur(1px); }
    100% { opacity: 0;  transform: scale(2) rotate(-15deg);   filter: blur(10px); }
}

.dissolve-particle {
    position: fixed;
    width: 5px; height: 5px;
    border-radius: 50%;
    pointer-events: none !important;
    z-index: 999999;
    background: #ffffff;
    box-shadow: 0 0 6px var(--primary), 0 0 15px var(--accent);
    animation: particle-fly 0.7s ease-out forwards;
}

@keyframes particle-fly {
    0%   { opacity: 1; transform: translate(0, 0) scale(1); }
    100% { opacity: 0; transform: translate(var(--dx), var(--dy)) rotate(var(--rot)) scale(.1); }
}

.positive-flash {
    position: absolute; inset: 0;
    background: radial-gradient(circle at center,
        rgba(255, 255, 255, 0.30), rgba(34, 211, 238, 0.12), transparent 66%);
    animation: flashPositive 1.3s ease-out;
}

.negative-flash {
    position: absolute; inset: 0;
    background: radial-gradient(circle at center,
        rgba(251, 75, 107, 0.22), rgba(120, 0, 30, 0.12), transparent 70%);
    animation: flashPositive 1.2s ease-out;
}

@keyframes flashPositive {
    0%   { opacity: 0; }
    18%  { opacity: 1; }
    100% { opacity: 0; }
}

.firework {
    position: absolute;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ffffff;
    animation: explode 1.45s ease-out forwards;
    box-shadow:
        0 0 10px #c4b5fd,
        0 0 25px #8b5cf6,
        0 0 45px rgba(34, 211, 238, 0.8);
}

@keyframes explode {
    0%   { transform: scale(0);    opacity: 1; }
    20%  { transform: scale(1.3);  opacity: 1; }
    100% { transform: scale(28);   opacity: 0; }
}

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {
    .block-container { padding-left: 15px !important; padding-right: 15px !important; }
    .ai-header { padding: 22px 23px; }
    .ai-header-title { font-size: 25px; }
    .ai-header-subtitle { font-size: 12px; }
    .status-bar { flex-direction: column; align-items: flex-start; gap: 6px; }
    div[data-testid="stRadio"] > div { flex-direction: column; }
    div[data-testid="stRadio"] label { width: 100%; }
    div[data-testid="stTextInput"] input { height: 54px !important; font-size: 14px !important; }
    .emotion-item { font-size: 30px; }
    .landing-hero { padding: 45px 0 25px; }
    .landing-subtitle { font-size: 15px; }
    .about-panel { padding: 24px; }
}

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "landing_view" not in st.session_state:
    st.session_state.landing_view = "home"


# =========================================================
# HELPERS
# =========================================================

def metric_card(value, label, kind=""):
    return f"""
    <div class="metric-card {kind}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_back_button():
    """Nút quay lại trang chủ - đặt ở cuối trang."""
    st.html("<div style='height:30px'></div>")
    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button(
            "←  QUAY LẠI TRANG CHỦ",
            key="back_home",
            use_container_width=True,
        ):
            st.session_state.show_analysis = False
            st.session_state.landing_view = "home"
            st.rerun()


# =========================================================
# LANDING PAGE
# =========================================================

def render_landing_page():
    view = st.session_state.landing_view

    # ---------- Navigation ----------
    # Key đổi theo trạng thái (_on / _off) để CSS tô màu nút đang chọn
    brand, nav_home, nav_about = st.columns(
        [3.2, 1, 1],
        vertical_alignment="center",
    )

    with brand:
        st.html('<div class="landing-brand">🧠 SENTIMENT<span> ANALYSIS</span></div>')

    with nav_home:
        if st.button(
            "Trang chủ",
            key=f"nav_home_{'on' if view == 'home' else 'off'}",
            use_container_width=True,
        ):
            st.session_state.landing_view = "home"
            st.rerun()

    with nav_about:
        if st.button(
            "Giới thiệu",
            key=f"nav_about_{'on' if view == 'about' else 'off'}",
            use_container_width=True,
        ):
            st.session_state.landing_view = "about"
            st.rerun()

    # ---------- Home ----------
    if view == "home":
        st.html("""
        <section class="landing-bg">
            <div class="landing-hero">
                <h1 class="landing-title">
                    KHÁM PHÁ CẢM XÚC<br>
                    <span class="landing-gradient">BẰNG TRÍ TUỆ NHÂN TẠO</span>
                </h1>

                <div class="landing-subtitle">
                    Phân tích cảm xúc tiếng Việt thông minh với
                    <span class="landing-tech">PhoBERT + BiLSTM + Attention</span>.
                    Khám phá sắc thái cảm xúc ẩn sau từng câu chữ.
                </div>

                <div class="landing-chips">
                    <span class="landing-chip">PhoBERT</span>
                    <span class="landing-chip">BiLSTM</span>
                    <span class="landing-chip alt">Attention</span>
                    <span class="landing-chip alt">Explainable AI</span>
                </div>

                <div class="landing-footer">
                    INTELLIGENT VIETNAMESE SENTIMENT ANALYSIS
                </div>
            </div>
        </section>
        """)

        _, center, _ = st.columns([1, 1.5, 1])
        with center:
            if st.button(
                "🚀  BẮT ĐẦU PHÂN TÍCH CẢM XÚC  →",
                key="landing_start",
                use_container_width=True,
            ):
                st.session_state.show_analysis = True
                st.rerun()

    # ---------- About ----------
    else:
        st.html("""
        <section class="about-panel">
            <h1 class="about-title">GIỚI THIỆU</h1>

            <div class="about-content">
                Hệ thống phân loại cảm xúc văn bản là ứng dụng trí tuệ nhân tạo
                được phát triển nhằm tự động nhận diện và phân loại cảm xúc
                tích cực, tiêu cực trong các văn bản tiếng Việt. Thông qua việc
                kết hợp mô hình PhoBERT với kiến trúc BiLSTM và cơ chế Attention,
                hệ thống có khả năng khai thác ngữ nghĩa, nắm bắt ngữ cảnh hai
                chiều và xác định những từ ngữ quan trọng ảnh hưởng đến kết quả
                dự đoán. Hệ thống hỗ trợ phân tích trực tiếp từng câu văn hoặc
                xử lý dữ liệu hàng loạt từ tệp CSV, Excel, đồng thời cung cấp
                độ tin cậy dự đoán và biểu đồ Attention giúp người dùng hiểu rõ
                hơn về quá trình phân tích.
            </div>

            <a class="model-link" target="_blank" rel="noopener noreferrer"
               href="https://vieclam.nganhang.com/thuat-ngu/phobert-mo-hinh-ngon-ngu-tieng-viet-la-gi">
                <div class="model-card">
                    <div class="model-title">01 / PhoBERT ↗</div>
                    <div class="model-desc">
                        Biểu diễn ngữ nghĩa và ngữ cảnh của văn bản tiếng Việt.
                    </div>
                </div>
            </a>

            <a class="model-link" target="_blank" rel="noopener noreferrer"
               href="https://www.geeksforgeeks.org/nlp/bidirectional-lstm-in-nlp/">
                <div class="model-card">
                    <div class="model-title">02 / BiLSTM ↗</div>
                    <div class="model-desc">
                        Khai thác thông tin ngữ cảnh theo cả hai chiều của chuỗi.
                    </div>
                </div>
            </a>

            <a class="model-link" target="_blank" rel="noopener noreferrer"
               href="https://vietnix.vn/attention-mechanism/">
                <div class="model-card">
                    <div class="model-title">03 / Attention ↗</div>
                    <div class="model-desc">
                        Hỗ trợ quan sát những token được mô hình chú ý trong quá trình dự đoán.
                    </div>
                </div>
            </a>
        </section>
        """)

        _, center, _ = st.columns([1, 1.5, 1])
        with center:
            if st.button(
                "🚀  BẮT ĐẦU PHÂN TÍCH CẢM XÚC  →",
                key="about_start",
                use_container_width=True,
            ):
                st.session_state.show_analysis = True
                st.rerun()


if not st.session_state.show_analysis:
    render_landing_page()
    st.stop()


# =========================================================
# MAIN PAGE - HEADER + STATUS BAR
# =========================================================

st.html(
    """
<div class="ai-header">
    <div class="ai-header-small">AI • NLP • VIETNAMESE SENTIMENT ANALYSIS</div>
    <div class="ai-header-title">HỆ THỐNG PHÂN LOẠI CẢM XÚC VĂN BẢN</div>
    <div class="ai-header-subtitle">
        PhoBERT + BiLSTM + Attention Neural Network &nbsp; | &nbsp; Explainable AI
    </div>
</div>
"""
)

st.html(
    """
<div class="status-bar">
    <span class="status-online">Online</span>
    <span>LANGUAGE: Tiếng Việt</span>
</div>
"""
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_vncorenlp():
    # Tự động lấy đường dẫn chính xác tới thư mục VnCoreNLP nằm cùng cấp với app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(base_dir, "VnCoreNLP", "VnCoreNLP-1.1.1.jar") # Đổi thành 1.1.1 nếu file của bạn là bản 1.1.1
    
    # Chỉ nạp 'wseg' (tách từ) và giới hạn RAM 1GB (-Xmx1g) để chạy mượt trên gói Free của Streamlit
    annotator = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx1g')
    return annotator
@st.cache_resource
def load_model_and_tokenizer():
    # 1. Tải và cache Tokenizer của PhoBERT-base-v2
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base-v2",
        use_fast=False,
    )

    # 2. Kiểm tra và tự động tải file best_model.pt từ Hugging Face nếu chưa có trên server
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_model.pt")
    
    # Thay bằng đường link tải trực tiếp file best_model.pt trên Hugging Face của bạn
    model_url = "https://huggingface.co/immchangnov28/vietnamese-sentiment-model/blob/main/best_model.pt"

    if not os.path.exists(model_path):
        with st.spinner("Đang tải trọng số mô hình lần đầu, vui lòng chờ..."):
            urllib.request.urlretrieve(model_url, model_path)

    # 3. Khởi tạo kiến trúc mô hình và nạp trọng số lên CPU
    device = torch.device("cpu")
    model = PhoBERT_BILSTM_Attention()
    
    # Nếu khi lưu trên Colab bạn dùng torch.save(model.state_dict(), ...) thì giữ nguyên dòng dưới:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    return tokenizer, model
def load_resources():
    tokenizer, model = load_model_and_tokenizer()
    rdrsegmenter = load_vncorenlp()

    return tokenizer, model, rdrsegmenter


def segment_vietnamese_text(text, rdrsegmenter):
    """Tách từ tiếng Việt bằng VnCoreNLP, cùng cách với preprocess.py."""
    text = str(text).strip()
    if not text:
        return ""

    sentences = rdrsegmenter.tokenize(text)
    # VnCoreNLP trả về danh sách các câu, mỗi câu là danh sách token.
    tokens = [token for sentence in sentences for token in sentence]
    return " ".join(tokens)


try:
    tokenizer, model, rdrsegmenter = load_resources()

except FileNotFoundError:
    st.error(
        "⚠️ Không tìm thấy file 'best_model.pt'. "
        "Vui lòng đặt file trọng số cùng thư mục với app.py."
    )
    render_back_button()
    st.stop()

except Exception as e:
    st.error(f"⚠️ Không thể tải mô hình. Chi tiết: {str(e)}")
    render_back_button()
    st.stop()


# =========================================================
# EMOTION EFFECTS
# =========================================================

effect_script = r"""
<script>
(() => {
    const root = document.currentScript?.parentElement;
    if (!root) return;

    const bindEmoji = () => {
        root.querySelectorAll(".emotion-item").forEach((item) => {
            if (item.dataset.dissolveBound === "1") return;
            item.dataset.dissolveBound = "1";

            item.addEventListener("click", (event) => {
                event.stopPropagation();
                if (item.classList.contains("is-dissolving")) return;

                const rect = item.getBoundingClientRect();
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;

                item.classList.add("is-dissolving");

                for (let i = 0; i < 20; i++) {
                    const p = document.createElement("span");
                    p.className = "dissolve-particle";
                    p.style.left = `${cx}px`;
                    p.style.top = `${cy}px`;

                    const angle = (Math.PI * 2 * i / 20) + (Math.random() - 0.5) * 0.6;
                    const dist = 40 + Math.random() * 100;

                    p.style.setProperty("--dx", `${Math.cos(angle) * dist}px`);
                    p.style.setProperty("--dy", `${Math.sin(angle) * dist}px`);
                    p.style.setProperty("--rot", `${Math.random() * 360}deg`);
                    p.style.animationDelay = `${Math.random() * 0.1}s`;

                    document.body.appendChild(p);
                    p.addEventListener("animationend", () => p.remove(), { once: true });
                }

                setTimeout(() => item.remove(), 720);
            });
        });
    };

    bindEmoji();
    requestAnimationFrame(bindEmoji);
})();
</script>
"""


def build_emoji_overlay(emojis, flash_class, left_offset, left_mod, delay_mod, delay_step):
    html = f'<div class="emotion-overlay"><div class="{flash_class}"></div>'

    for i, emoji in enumerate(emojis * 2):
        left = (i * 7 + left_offset) % left_mod
        delay = (i % delay_mod) * delay_step
        duration = 3.6 + (i % 5) * 0.30
        html += f"""
        <div class="emotion-item"
             style="left:{left}%; animation-delay:{delay}s; animation-duration:{duration}s;"
             title="Bấm để hòa tan">{emoji}</div>
        """

    return html


def show_positive_effect():
    emojis = [
        "❤️", "💖", "💕", "💗", "💓",
        "😍", "🥰", "😊", "👍", "👏",
        "🎉", "✨", "🌟", "💝", "💘",
    ]

    html = build_emoji_overlay(emojis, "positive-flash", 3, 97, 12, 0.11)

    fireworks = [
        (14, 22, 0.10), (31, 34, 0.45), (50, 18, 0.85), (69, 30, 0.30),
        (86, 21, 1.00), (23, 58, 1.25), (62, 56, 0.65), (82, 62, 1.15),
    ]
    for left, top, delay in fireworks:
        html += (
            f'<div class="firework" '
            f'style="left:{left}%; top:{top}%; animation-delay:{delay}s;"></div>'
        )

    html += effect_script + "</div>"
    st.html(html, unsafe_allow_javascript=True)


def show_negative_effect():
    emojis = [
        "😡", "🤬", "😤", "💢", "😭", "😢", "🥺", "💔",
        "😠", "😡", "💢", "😭", "😢", "💔", "🤬",
    ]

    html = build_emoji_overlay(emojis, "negative-flash", 5, 96, 10, 0.13)
    html += effect_script + "</div>"
    st.html(html, unsafe_allow_javascript=True)


# =========================================================
# MODE SELECTION
# =========================================================

st.html('<div class="section-label">CHỌN CHẾ ĐỘ PHÂN TÍCH</div>')

selection = st.radio(
    "Chọn chế độ phân tích",
    ["PHÂN TÍCH 1 CÂU TRỰC TIẾP", "PHÂN TÍCH QUA FILE DỮ LIỆU"],
    horizontal=True,
    label_visibility="collapsed",
)

st.html("<br>")


# =========================================================
# OPTION 1 - SINGLE TEXT
# =========================================================

if selection == "PHÂN TÍCH 1 CÂU TRỰC TIẾP":

    st.html(
        '<div class="section-label">Vui lòng nhập câu cần phân loại cảm xúc tại đây</div>'
    )

    user_input = st.text_input(
        "Nhập câu cần phân tích",
        placeholder="Ví dụ: Sản phẩm này dùng rất tốt, tôi rất hài lòng...",
        label_visibility="collapsed",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html(metric_card("01", "Input Text"))
    with c2:
        st.html(metric_card("64", "Max Tokens"))
    with c3:
        st.html(metric_card("AI", "Prediction Engine"))
    with c4:
        st.html(metric_card("XAI", "Attention Analysis"))

    st.html("<br>")

    if st.button("🚀  BẮT ĐẦU PHÂN TÍCH CẢM XÚC"):
        if user_input.strip() == "":
            st.warning("Vui lòng không để trống ô nhập liệu.")

        else:
            # ---------- Word Segmentation + PhoBERT Tokenization ----------
            segmented_input = segment_vietnamese_text(user_input, rdrsegmenter)
            if not segmented_input:
                st.warning("Không có nội dung hợp lệ sau khi tách từ.")
                st.stop()

            st.caption(f"Văn bản sau khi tách từ: {segmented_input}")

            inputs = tokenizer(
                segmented_input,
                return_tensors="pt",
                max_length=64,
                padding="max_length",
                truncation=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

            # ---------- Prediction ----------
            with torch.no_grad():
                logits, attn_weights = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1)
            positive_conf = probs[0][1].item() * 100
            negative_conf = probs[0][0].item() * 100

            if positive_conf > negative_conf:
                prediction, confidence = 1, positive_conf
            else:
                prediction, confidence = 0, negative_conf

            # ---------- Result ----------
            st.html("<br>")
            st.html('<div class="section-label">PREDICTION RESULT</div>')

            if prediction == 1:
                show_positive_effect()
                st.success(f"😊  **TÍCH CỰC**   •   Độ tin cậy: **{confidence:.2f}%**")
            else:
                show_negative_effect()
                st.error(f"😡  **TIÊU CỰC**   •   Độ tin cậy: **{confidence:.2f}%**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.html(metric_card(f"{positive_conf:.2f}%", "Positive Probability", "metric-pos"))
            with col_b:
                st.html(metric_card(f"{negative_conf:.2f}%", "Negative Probability", "metric-neg"))

            # ---------- XAI ----------
            st.html("<br>")
            st.html('<div class="section-label">EXPLAINABLE AI / ATTENTION MAP</div>')

            valid_indices = [
                i for i, token in enumerate(tokens)
                if token not in ["<s>", "</s>", "<pad>"]
            ]
            clean_tokens = [tokens[i].replace("_", " ") for i in valid_indices]

            try:
                clean_weights = attn_weights[0, valid_indices, 0].detach().cpu().numpy()
            except Exception:
                # Fallback nếu attention tensor có shape khác
                clean_weights = (
                    attn_weights[0, valid_indices].detach().cpu().numpy().reshape(-1)
                )

            fig_height = max(3.5, len(clean_tokens) * 0.35 + 1)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            ax.barh(
                clean_tokens,
                clean_weights,
                color=C_POS if prediction == 1 else C_NEG,
            )

            ax.set_xlabel("Mức độ tập trung của mô hình (Attention Weight)")
            ax.invert_yaxis()
            ax.set_facecolor(C_PANEL)
            fig.patch.set_facecolor(C_BG)
            ax.tick_params(colors=C_MUTED, labelsize=9)
            ax.xaxis.label.set_color(C_MUTED)

            for spine in ax.spines.values():
                spine.set_color("#3b2a7a")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# =========================================================
# OPTION 2 - BATCH FILE
# =========================================================

else:
    st.html('<div class="section-label">CHỌN TỆP CẦN PHÂN TÍCH</div>')

    st.html(
        """
        <div class="dashboard-card">
            <div style="color:#a9a5c8; font-size:13px;">
                Tải lên tập dữ liệu CSV hoặc Excel để hệ thống
                tự động phân loại cảm xúc.
            </div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn file dữ liệu của bạn tại đây",
        type=["csv", "xlsx"],
    )

    df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Không thể đọc file. Chi tiết: {str(e)}")
            df = None

        if df is not None and df.empty:
            st.warning("⚠️ File không có dữ liệu.")
            df = None

    if df is not None:

        # ---------- Preview ----------
        st.html('<div class="section-label">BẢN XEM TRƯỚC DỮ LIỆU</div>')
        st.dataframe(df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.html(metric_card(f"{len(df):,}", "Total Rows"))
        with c2:
            st.html(metric_card(len(df.columns), "Columns"))
        with c3:
            st.html(metric_card("CSV / XLSX", "Supported Format"))

        st.html("<br>")

        # ---------- Chọn cột ----------
        st.html('<div class="section-label">CHỌN CỘT DỮ LIỆU</div>')
        text_column = st.selectbox(
            "Chọn cột chứa nội dung bình luận cần phân tích:",
            df.columns,
        )
        st.html("<br>")

        # ---------- Chạy phân tích ----------
        if st.button("⚡  CHẠY PHÂN TÍCH HÀNG LOẠT"):
            predictions = []
            total = len(df)
            loader_placeholder = st.empty()
            device = next(model.parameters()).device

            for idx, text in enumerate(df[text_column]):

                # Cập nhật loader (không cần vẽ lại ở mọi dòng)
                if idx % 3 == 0 or idx == total - 1:
                    percent_done = int((idx + 1) / total * 100)
                    loader_placeholder.html(
                        f"""
                        <div class="loader-wrap">
                            <div class="coffee-container">
                                <div class="smoke-stream smoke-1"></div>
                                <div class="smoke-stream smoke-2"></div>
                                <div class="smoke-stream smoke-3"></div>
                                <div class="coffee-cup"><div class="coffee-handle"></div></div>
                            </div>
                            <div class="loader-percent">SYSTEM PROCESSING: {percent_done}%</div>
                            <div class="loader-sub">Đang xử lý dòng {idx + 1}/{total}...</div>
                        </div>
                        """
                    )

                segmented_text = segment_vietnamese_text(text, rdrsegmenter)
                inputs = tokenizer(
                    segmented_text,
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    logits, _ = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)

                pos_p = probs[0][1].item()
                neg_p = probs[0][0].item()
                predictions.append("Tích cực" if pos_p > neg_p else "Tiêu cực")

            loader_placeholder.empty()

            df["Dự đoán cảm xúc"] = predictions

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)

            # Lưu vào session_state để kết quả không biến mất khi bấm nút tải xuống
            st.session_state.batch_result = {
                "df": df,
                "text_column": text_column,
                "file_name": uploaded_file.name,
                "csv": csv_bytes,
                "xlsx": excel_buffer.getvalue(),
            }
            st.balloons()

        # ---------- Hiển thị kết quả (nằm ngoài khối nút bấm) ----------
        result = st.session_state.get("batch_result")

        if result is not None and result["file_name"] == uploaded_file.name:
            res_df = result["df"]
            res_col = result["text_column"]
            base_name = result["file_name"].rsplit(".", 1)[0] + "_da_gan_nhan"

            st.html("<br>")
            st.html('<div class="section-label">KẾT QUẢ PHÂN TÍCH</div>')

            col_chart, col_preview = st.columns(2)

            with col_chart:
                st.markdown("**PHÂN BỔ TỶ LỆ CẢM XÚC:**")
                counts = res_df["Dự đoán cảm xúc"].value_counts()
                colors_dict = {"Tích cực": C_POS, "Tiêu cực": C_NEG}

                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                fig_pie.patch.set_facecolor(C_BG)
                ax_pie.set_facecolor(C_BG)

                ax_pie.pie(
                    counts.values,
                    labels=counts.index,
                    autopct="%1.1f%%",
                    colors=[colors_dict.get(x, C_PRIMARY) for x in counts.index],
                    textprops={"color": "white", "fontsize": 10},
                    wedgeprops={"edgecolor": C_BG, "linewidth": 2},
                )
                plt.tight_layout()
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)

            with col_preview:
                st.markdown("**BẢN XEM TRƯỚC KẾT QUẢ:**")
                st.dataframe(
                    res_df[[res_col, "Dự đoán cảm xúc"]].head(10),
                    use_container_width=True,
                )

                st.markdown("**TẢI XUỐNG FILE ĐÃ GÁN NHÃN:**")
                dl_csv, dl_xlsx = st.columns(2)

                with dl_csv:
                    st.download_button(
                        label="📥 TẢI FILE .CSV",
                        data=result["csv"],
                        file_name=f"{base_name}.csv",
                        mime="text/csv",
                        key="download_csv",
                    )

                with dl_xlsx:
                    st.download_button(
                        label="📥 TẢI FILE .XLSX",
                        data=result["xlsx"],
                        file_name=f"{base_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_xlsx",
                    )


# =========================================================
# BACK BUTTON (CUỐI TRANG) + FOOTER
# =========================================================

render_back_button()

st.html(
    """
<div style="
    text-align:center;
    margin-top:30px;
    padding-top:15px;
    border-top:1px solid rgba(139,92,246,0.18);
    color:#6f6b94;
    font-family:monospace;
    font-size:10px;
    letter-spacing:1px;">
    SENTIMENT AI ENGINE &nbsp; • &nbsp; PHOBERT / BILSTM / ATTENTION
    &nbsp; • &nbsp; EXPLAINABLE AI
</div>
"""
)
