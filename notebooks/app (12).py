import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import hashlib
import secrets
import re
from datetime import datetime

from streamlit_drawable_canvas import st_canvas

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# Hide sidebar entirely since we use top navigation
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
    --primary-bg: #0F172A;
    --secondary-bg: #1E293B;
    --card-bg: #1E293B;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --accent: #3B82F6;
    --success: #10B981;
    --warning: #F59E0B;
    --border: #334155;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.stApp {
    background-color: var(--primary-bg);
}

.stMarkdown p, .stMarkdown li, .stMarkdown div {
    color: var(--text-primary);
}

/* Card Style */
.metric-card {
    background-color: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    margin-bottom: 15px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    margin-bottom: 8px;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

/* Ad Column Style */
.ad-card {
    background-color: #162032;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
}
.ad-card h4 {
    color: var(--accent);
    font-weight: 800;
    margin-bottom: 10px;
    font-size: 0.95rem;
    letter-spacing: 1px;
}
.ad-card p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    margin: 8px 0;
    line-height: 1.4;
}
.ad-card hr {
    border-color: var(--border);
    margin: 15px 0;
}

/* Steps */
.step-number {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    opacity: 0.8;
}
.step-title {
    font-weight: 700;
    margin-top: 12px;
    font-size: 1.1rem;
    color: var(--text-primary);
}
.step-desc {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 8px;
}

/* Arch Box */
.arch-box {
    background-color: var(--card-bg);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin: 8px auto;
    width: 250px;
    color: var(--text-primary);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.arch-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: var(--accent);
    margin: 4px 0;
}

/* Prediction Card */
.pred-card {
    background-color: var(--card-bg);
    border: 2px solid var(--success);
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    box-shadow: 0 10px 20px rgba(16, 185, 129, 0.1);
}
.pred-value {
    font-size: 7rem;
    font-weight: 800;
    color: var(--success);
    line-height: 1;
    margin: 20px 0;
    text-shadow: 0 4px 10px rgba(16, 185, 129, 0.2);
}
.pred-conf {
    font-size: 1.2rem;
    color: var(--text-secondary);
}

/* Badge */
.badge {
    background-color: rgba(59, 130, 246, 0.15);
    color: var(--accent);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}

/* Profile — same account management behavior as reference */
.profile-card {
    background-color: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,.2);
}
.avatar {
    width:72px; height:72px; border-radius:50%;
    background:rgba(59,130,246,.15); color:var(--accent);
    display:flex; align-items:center; justify-content:center;
    font-size:1.7rem; font-weight:800;
}
.profile-name { font-size:1.35rem; font-weight:800; color:var(--text-primary); }
.profile-muted { color:var(--text-secondary); font-size:.85rem; }

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.6;
}

/* Radio Navigation Styling adjustments */
div.row-widget.stRadio > div {
    flex-direction: row;
    justify-content: center;
    gap: 30px;
}
</style>
<style>
    /* Login/account drawer — matched to the reference app's right-side layout */
    :root {
        --auth-ink: #F8FAFC;
        --auth-muted: #94A3B8;
        --auth-line: #334155;
    }

    div[data-testid="stPopover"] > button {
        min-height: 58px !important;
        width: 100% !important;
        border-radius: 16px !important;
        background: #1E293B !important;
        border: 1px solid var(--auth-line) !important;
        color: var(--auth-ink) !important;
        font-weight: 800 !important;
        box-shadow: 0 5px 18px rgba(15,23,42,.05) !important;
    }
    div[data-testid="stPopover"] > button * {
        color: var(--auth-ink) !important;
        -webkit-text-fill-color: var(--auth-ink) !important;
    }

    /* Full-height right-side drawer, same placement/feel as the reference */
    div[data-testid="stPopoverBody"],
    div[data-baseweb="popover"] {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        left: auto !important;
        bottom: 0 !important;
        width: min(430px, 94vw) !important;
        max-width: min(430px, 94vw) !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        background: #1E293B !important;
        border: 0 !important;
        border-left: 1px solid var(--auth-line) !important;
        border-radius: 0 !important;
        padding: 1.25rem !important;
        z-index: 999999 !important;
        transform: none !important;
        box-shadow: -100vw 0 0 100vw rgba(15,23,42,.42), -14px 0 38px rgba(15,23,42,.20) !important;
    }

    div[data-testid="stPopoverBody"] input,
    div[data-baseweb="popover"] input {
        color: var(--auth-ink) !important;
        -webkit-text-fill-color: var(--auth-ink) !important;
        background: #1E293B !important;
    }
    div[data-testid="stPopoverBody"] label,
    div[data-testid="stPopoverBody"] p,
    div[data-testid="stPopoverBody"] span,
    div[data-testid="stPopoverBody"] h1,
    div[data-testid="stPopoverBody"] h2,
    div[data-testid="stPopoverBody"] h3,
    div[data-baseweb="popover"] label,
    div[data-baseweb="popover"] p,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] h1,
    div[data-baseweb="popover"] h2,
    div[data-baseweb="popover"] h3 {
        color: var(--auth-ink) !important;
    }

    .drawer-head {
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding-bottom:.9rem;
        margin-bottom:1rem;
        border-bottom:1px solid var(--auth-line);
    }
    .drawer-title {
        font-size:1.15rem;
        font-weight:850;
        color:var(--auth-ink);
    }
    .drawer-sub {
        font-size:.78rem;
        color:var(--auth-muted);
        margin-top:.15rem;
    }
    .drawer-badge {
        width:42px;
        height:42px;
        border-radius:13px;
        background:rgba(59,130,246,.15);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.35rem;
    }

    /* Match the reference app's login controls inside the drawer */
    div[data-testid="stPopoverBody"] div[role="radiogroup"] {
        gap: .45rem;
    }
    div[data-testid="stPopoverBody"] div[role="radiogroup"] label {
        background: #1E293B !important;
        border: 1px solid var(--auth-line) !important;
        border-radius: 10px !important;
        padding: .42rem .75rem !important;
        color: var(--auth-ink) !important;
        font-weight: 750 !important;
    }
    div[data-testid="stPopoverBody"] div[role="radiogroup"] label * {
        color: var(--auth-ink) !important;
    }
    div[data-testid="stPopoverBody"] button[kind="primary"] {
        background: #3B82F6 !important;
        color: #fff !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 11px !important;
        min-height: 2.9rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stPopoverBody"] button[kind="primary"] * {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)


# --- LIGHT / REGULAR WEBSITE THEME OVERRIDE ---
# Presentation-only override: keeps all existing functionality and content.
st.markdown("""
<style>
:root {
    --primary-bg: #f7f9fc !important;
    --secondary-bg: #ffffff !important;
    --card-bg: #ffffff !important;
    --text-primary: #111827 !important;
    --text-secondary: #6b7280 !important;
    --accent: #2563eb !important;
    --success: #16a34a !important;
    --warning: #d97706 !important;
    --border: #e5e7eb !important;
    --auth-ink: #111827 !important;
    --auth-muted: #6b7280 !important;
    --auth-line: #e5e7eb !important;
}

.stApp {
    background: #f7f9fc !important;
    color: #111827 !important;
}

html, body, [class*="css"] {
    color: #111827 !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown div,
.stMarkdown span,
.stMarkdown label {
    color: #111827 !important;
}

/* Main project cards */
.metric-card,
.arch-box,
.pred-card,
.profile-card,
.ad-card {
    background: #ffffff !important;
    border-color: #e5e7eb !important;
    box-shadow: 0 4px 16px rgba(15,23,42,.06) !important;
}

.metric-value,
.step-number,
.arch-arrow {
    color: #2563eb !important;
}

.metric-label,
.step-desc,
.pred-conf,
.profile-muted,
.footer,
.ad-card p {
    color: #6b7280 !important;
}

.step-title,
.profile-name,
.arch-box {
    color: #111827 !important;
}

.ad-card {
    background: #ffffff !important;
}

.ad-card h4 {
    color: #2563eb !important;
}

.badge {
    background: #eff6ff !important;
    color: #2563eb !important;
}

/* Main header: make the existing title/logo clearly visible */
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: #6b7280 !important;
}

/* Preserve the blue accent on headings that are intentionally blue */
.metric-value,
.step-number,
.arch-arrow,
.ad-card h4 {
    color: #2563eb !important;
    -webkit-text-fill-color: #2563eb !important;
}

/* Navigation */
div[role="radiogroup"] label,
div.row-widget.stRadio > div label {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #111827 !important;
    box-shadow: 0 2px 7px rgba(15,23,42,.04) !important;
}
div[role="radiogroup"] label *,
div.row-widget.stRadio > div label * {
    color: #111827 !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: #2563eb !important;
    border-color: #2563eb !important;
}
div[role="radiogroup"] label:has(input:checked) * {
    color: #ffffff !important;
}

/* Buttons */
button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}
button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
button[kind="secondary"] {
    background: #ffffff !important;
    border-color: #d1d5db !important;
    color: #111827 !important;
}
button[kind="secondary"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea,
input {
    background: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border-color: #d1d5db !important;
}
div[data-baseweb="input"] *,
div[data-baseweb="select"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* LOGIN / ACCOUNT DRAWER — regular website appearance */
div[data-testid="stPopover"] > button {
    min-height: 58px !important;
    width: 100% !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #111827 !important;
    box-shadow: 0 4px 14px rgba(15,23,42,.06) !important;
}
div[data-testid="stPopover"] > button *,
div[data-testid="stPopover"] > button span {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* White drawer */
div[data-testid="stPopoverBody"],
div[data-baseweb="popover"] {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: #e5e7eb !important;
}

/* Force every drawer heading/subtitle to be visible */
div[data-testid="stPopoverBody"] .drawer-head,
div[data-baseweb="popover"] .drawer-head {
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stPopoverBody"] .drawer-title,
div[data-baseweb="popover"] .drawer-title {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
}

div[data-testid="stPopoverBody"] .drawer-sub,
div[data-baseweb="popover"] .drawer-sub {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stPopoverBody"] .drawer-badge,
div[data-baseweb="popover"] .drawer-badge {
    background: #eff6ff !important;
    color: #2563eb !important;
}

/* Drawer text, labels and captions */
div[data-testid="stPopoverBody"] label,
div[data-testid="stPopoverBody"] p,
div[data-testid="stPopoverBody"] span,
div[data-testid="stPopoverBody"] h1,
div[data-testid="stPopoverBody"] h2,
div[data-testid="stPopoverBody"] h3,
div[data-baseweb="popover"] label,
div[data-baseweb="popover"] p,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] h1,
div[data-baseweb="popover"] h2,
div[data-baseweb="popover"] h3 {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}

/* Drawer input fields */
div[data-testid="stPopoverBody"] input,
div[data-baseweb="popover"] input {
    background: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border-color: #d1d5db !important;
}

/* Drawer Sign In/Create Account selector */
div[data-testid="stPopoverBody"] div[role="radiogroup"] label {
    background: #ffffff !important;
    border: 1px solid #dbe1e8 !important;
    color: #111827 !important;
}
div[data-testid="stPopoverBody"] div[role="radiogroup"] label * {
    color: #111827 !important;
}
div[data-testid="stPopoverBody"] div[role="radiogroup"] label:has(input:checked) {
    background: #eff6ff !important;
    border-color: #2563eb !important;
}
div[data-testid="stPopoverBody"] div[role="radiogroup"] label:has(input:checked) * {
    color: #2563eb !important;
}

/* Drawer primary action */
div[data-testid="stPopoverBody"] button[kind="primary"],
div[data-baseweb="popover"] button[kind="primary"] {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}
div[data-testid="stPopoverBody"] button[kind="primary"] *,
div[data-baseweb="popover"] button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Drawer close/logout buttons */
div[data-testid="stPopoverBody"] button[kind="secondary"],
div[data-baseweb="popover"] button[kind="secondary"] {
    background: #ffffff !important;
    border-color: #e5e7eb !important;
    color: #111827 !important;
}
div[data-testid="stPopoverBody"] button[kind="secondary"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Make inline dark project header text visible on the light page */
div[data-testid="stMarkdownContainer"] h1[style],
div[data-testid="stMarkdownContainer"] h2[style],
div[data-testid="stMarkdownContainer"] h3[style],
div[data-testid="stMarkdownContainer"] p[style] {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Keep intentionally muted inline text muted */
div[data-testid="stMarkdownContainer"] p[style*="94A3B8"] {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
}

/* Horizontal rules */
hr {
    border-color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# --- ML FUNCTIONS (DO NOT MODIFY) ---
@st.cache_resource
def load_model():
    model_path = "models/mnist_cnn.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def preprocess_image(image):
    # Convert to grayscale
    img_gray = image.convert('L')
    
    # Resize to 28x28
    img_resized = img_gray.resize((28, 28))
    
    # Convert to numpy array
    img_arr = np.array(img_resized)
    
    # Invert polarity if background is lighter than the digit
    border_pixels = np.concatenate([
        img_arr[0, :], img_arr[-1, :], img_arr[:, 0], img_arr[:, -1]
    ])
    if np.mean(border_pixels) > 127:
        img_arr = np.invert(img_arr)
    
    # Normalize to 0-1
    img_normalized = img_arr.astype('float32') / 255.0
    
    # Reshape for CNN
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)
    
    return img_reshaped, img_normalized

def get_metrics():
    test_acc = 0.0
    test_loss = 0.0
    if os.path.exists("outputs/metrics.txt"):
        with open("outputs/metrics.txt", "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                test_acc = float(lines[0].strip()) * 100
                test_loss = float(lines[1].strip())
    return test_acc, test_loss


# --- LOCAL ACCOUNT / LOGIN SYSTEM ---
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, expected = stored.split("$", 1)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            120_000,
        ).hex()
        return secrets.compare_digest(actual, expected)
    except ValueError:
        return False


def create_user(username, password, full_name, email, phone):
    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO users(username,password_hash,full_name,email,phone,created_at) VALUES (?,?,?,?,?,?)",
            (
                username,
                hash_password(password),
                full_name,
                email,
                phone,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True, "Account created successfully. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    finally:
        conn.close()


def authenticate(username, password):
    conn = db_connect()
    row = conn.execute(
        "SELECT id, username, password_hash, full_name, email, phone, created_at "
        "FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row and verify_password(password, row[2]):
        return {
            "id": row[0],
            "username": row[1],
            "full_name": row[3],
            "email": row[4],
            "phone": row[5] or "",
            "created_at": row[6],
        }
    return None


def get_user(user_id):
    conn = db_connect()
    row = conn.execute(
        "SELECT id, username, full_name, email, phone, created_at "
        "FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "username": row[1],
        "full_name": row[2],
        "email": row[3],
        "phone": row[4] or "",
        "created_at": row[5],
    }


def update_profile(user_id, full_name, email, phone):
    conn = db_connect()
    try:
        conn.execute(
            "UPDATE users SET full_name=?, email=?, phone=? WHERE id=?",
            (full_name, email, phone, user_id),
        )
        conn.commit()
        return True, "Profile updated successfully."
    except sqlite3.IntegrityError:
        return False, "That email is already used by another account."
    finally:
        conn.close()


def change_password(user_id, current_password, new_password):
    conn = db_connect()
    row = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or not verify_password(current_password, row[0]):
        conn.close()
        return False, "Current password is incorrect."
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


def valid_email(email):
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "") is not None


# Session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"
if "account_created" not in st.session_state:
    st.session_state.account_created = False

user = get_user(st.session_state.user_id) if st.session_state.user_id else None


# Header + right-side account drawer
def render_login_popover():
    """Reference-style account drawer with the first file's dark colors."""
    with st.popover("🔐 Login", help="Open account panel"):
        st.markdown(
            """<div class="drawer-head">
                <div>
                    <div class="drawer-title">Welcome to AI Handwritten Digit Recognition</div>
                    <div class="drawer-sub">Sign in or create your local academic-project account.</div>
                </div>
                <div class="drawer-badge">✍️</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("✕ Close", key="drawer_close_login", width="stretch"):
            st.rerun()
        if st.session_state.pop("account_created", False):
            st.session_state.drawer_auth_choice = "Sign In"

        auth_choice = st.radio(
            "Account", ["Sign In", "Create Account"], horizontal=True,
            label_visibility="collapsed", key="drawer_auth_choice"
        )

        if auth_choice == "Sign In":
            with st.form("drawer_login_form"):
                username = st.text_input("Username", placeholder="Enter your username", key="drawer_login_username")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="drawer_login_password")
                submitted = st.form_submit_button("🔐 Sign In", type="primary", width="stretch")
                if submitted:
                    if not username.strip() or not password:
                        st.error("Please enter both username and password.")
                    else:
                        found = authenticate(username.strip(), password)
                        if found:
                            st.session_state.user_id = found["id"]
                            st.session_state.page_request = "Home"
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
        else:
            with st.form("drawer_signup_form"):
                full_name = st.text_input("Full Name", placeholder="e.g. Riyaz Shaik", key="drawer_full_name")
                email = st.text_input("Email", placeholder="name@example.com", key="drawer_email")
                phone = st.text_input("Phone (optional)", placeholder="Optional", key="drawer_phone")
                username = st.text_input("Username", placeholder="Choose a username", key="drawer_username")
                password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="drawer_password")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="drawer_confirm")
                submitted = st.form_submit_button("✨ Create Account", type="primary", width="stretch")
                if submitted:
                    clean_username = username.strip()
                    clean_name = full_name.strip()
                    clean_email = email.strip().lower()
                    if not clean_name or not clean_username or not clean_email or not password:
                        st.error("Please complete all required fields.")
                    elif not valid_email(clean_email):
                        st.error("Please enter a valid email address.")
                    elif len(clean_username) < 3:
                        st.error("Username must contain at least 3 characters.")
                    elif len(password) < 6:
                        st.error("Password must contain at least 6 characters.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, message = create_user(clean_username, password, clean_name, clean_email, phone.strip())
                        if ok:
                            st.session_state.account_created = True
                            st.rerun()
                        else:
                            st.error(message)

        st.caption("Academic demonstration account. Avoid using passwords that you use for important personal services.")


def render_user_popover(user):
    display_name = user["full_name"] or user["username"]
    with st.popover(f"👤 {display_name}", help="Account and profile"):
        st.markdown(
            f"""<div class="drawer-head">
                <div>
                    <div class="drawer-title">{display_name}</div>
                    <div class="drawer-sub">@{user['username']}</div>
                </div>
                <div class="drawer-badge">👤</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("✕ Close", key="drawer_close_user", width="stretch"):
            st.rerun()
        st.caption(user["email"])
        if user["phone"]:
            st.caption(user["phone"])
        if st.button("👤 My Profile", key="header_profile_button", width="stretch"):
            st.session_state.page_request = "My Profile"
            st.rerun()
        if st.button("🚪 Log Out", key="header_logout_button", width="stretch"):
            st.session_state.user_id = None
            st.session_state.page_request = "Home"
            st.rerun()


# --- APP LAYOUT ---

st.markdown("""
<style>
/* =========================================================
   LEETCODE-INSPIRED PROFESSIONAL VISUAL REFRESH
   Visual-only overrides — application logic is unchanged.
   ========================================================= */

:root {
    --lc-bg: #0f0f0f !important;
    --lc-surface: #171717 !important;
    --lc-surface-2: #1f1f1f !important;
    --lc-text: #f5f5f5 !important;
    --lc-muted: #a3a3a3 !important;
    --lc-accent: #ffa116 !important;
    --lc-accent-soft: rgba(255,161,22,.12) !important;
    --lc-success: #2db55d !important;
    --lc-border: #2b2b2b !important;
}

.stApp {
    background: var(--lc-bg) !important;
    color: var(--lc-text) !important;
}

.main .block-container {
    max-width: 1320px !important;
    padding-top: 0.15rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}

/* Remove Streamlit's top spacer so the header/login row sits at the very top */
header {
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}

div[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

div[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

div[data-testid="stAppViewContainer"] > .main > div,
div[data-testid="stMainBlockContainer"],
section.main > div,
.main .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Keep the header/login row itself tight to the top edge */
.main .block-container > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Global typography */
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown div,
.stMarkdown span,
.stMarkdown label {
    color: var(--lc-text) !important;
}

.stMarkdown p {
    line-height: 1.55 !important;
}

/* Main headings: slightly smaller, denser and more professional */
div[data-testid="stMarkdownContainer"] h1 {
    font-size: 3rem !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em !important;
    color: var(--lc-text) !important;
    -webkit-text-fill-color: var(--lc-text) !important;
}

div[data-testid="stMarkdownContainer"] h2 {
    font-size: 2rem !important;
    line-height: 1.2 !important;
    font-weight: 750 !important;
    color: var(--lc-text) !important;
    -webkit-text-fill-color: var(--lc-text) !important;
}

div[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.25rem !important;
    font-weight: 750 !important;
    color: var(--lc-text) !important;
    -webkit-text-fill-color: var(--lc-text) !important;
}

div[data-testid="stMarkdownContainer"] h4 {
    font-size: .95rem !important;
    font-weight: 750 !important;
}

/* Compact professional cards */
.metric-card,
.arch-box,
.profile-card,
.ad-card {
    background: var(--lc-surface) !important;
    border: 1px solid var(--lc-border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

.metric-card {
    padding: 19px 18px !important;
    min-height: 112px !important;
}

.metric-value {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: var(--lc-accent) !important;
    -webkit-text-fill-color: var(--lc-accent) !important;
    letter-spacing: -.02em !important;
}

.metric-label {
    font-size: .73rem !important;
    letter-spacing: .9px !important;
    color: var(--lc-muted) !important;
    -webkit-text-fill-color: var(--lc-muted) !important;
}

/* Accent / badges */
.badge {
    background: var(--lc-accent-soft) !important;
    color: var(--lc-accent) !important;
    -webkit-text-fill-color: var(--lc-accent) !important;
    border: 1px solid rgba(255,161,22,.25) !important;
    border-radius: 999px !important;
    padding: 5px 11px !important;
    font-size: .72rem !important;
    font-weight: 750 !important;
    letter-spacing: .7px !important;
}

/* Steps */
.step-number {
    font-size: 1.65rem !important;
    color: var(--lc-accent) !important;
    -webkit-text-fill-color: var(--lc-accent) !important;
}

.step-title {
    font-size: .95rem !important;
    color: var(--lc-text) !important;
}

.step-desc {
    font-size: .8rem !important;
    color: var(--lc-muted) !important;
}

/* Side information cards */
.ad-card {
    padding: 16px !important;
}

.ad-card h4 {
    color: var(--lc-accent) !important;
    -webkit-text-fill-color: var(--lc-accent) !important;
    font-size: .82rem !important;
    letter-spacing: .8px !important;
}

.ad-card p {
    color: var(--lc-muted) !important;
    font-size: .76rem !important;
}

/* Architecture */
.arch-box {
    padding: 12px !important;
    width: 225px !important;
    border-color: #3a3a3a !important;
    font-size: .9rem !important;
}

.arch-arrow {
    color: var(--lc-accent) !important;
    -webkit-text-fill-color: var(--lc-accent) !important;
    font-size: 1.2rem !important;
}

/* Prediction result */
.pred-card {
    background: var(--lc-surface) !important;
    border: 1px solid rgba(45,181,93,.55) !important;
    border-radius: 12px !important;
    padding: 28px 18px !important;
    box-shadow: none !important;
}

.pred-value {
    font-size: 6rem !important;
    color: var(--lc-success) !important;
    -webkit-text-fill-color: var(--lc-success) !important;
    margin: 12px 0 !important;
    text-shadow: none !important;
}

.pred-conf {
    font-size: 1rem !important;
    color: var(--lc-muted) !important;
}

/* Top navigation: compact pill-style tabs */
div[role="radiogroup"],
div.row-widget.stRadio > div {
    gap: 8px !important;
    justify-content: center !important;
}

div[role="radiogroup"] label,
div.row-widget.stRadio > div label {
    background: #171717 !important;
    border: 1px solid #2b2b2b !important;
    border-radius: 7px !important;
    color: #a3a3a3 !important;
    padding: 7px 12px !important;
    min-height: 34px !important;
    font-size: .82rem !important;
    font-weight: 650 !important;
    box-shadow: none !important;
    transition: border-color .15s ease, color .15s ease, background .15s ease !important;
}

div[role="radiogroup"] label * ,
div.row-widget.stRadio > div label * {
    color: inherit !important;
    -webkit-text-fill-color: inherit !important;
}

div[role="radiogroup"] label:has(input:checked),
div.row-widget.stRadio > div label:has(input:checked) {
    background: var(--lc-accent-soft) !important;
    border-color: rgba(255,161,22,.45) !important;
    color: var(--lc-accent) !important;
}

/* Buttons */
button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
    background: var(--lc-accent) !important;
    border-color: var(--lc-accent) !important;
    color: #111111 !important;
    border-radius: 7px !important;
    min-height: 2.55rem !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}

button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] button * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

button[kind="secondary"] {
    background: #171717 !important;
    border-color: #363636 !important;
    color: #e5e5e5 !important;
    border-radius: 7px !important;
    min-height: 2.45rem !important;
    box-shadow: none !important;
}

button[kind="secondary"] * {
    color: #e5e5e5 !important;
    -webkit-text-fill-color: #e5e5e5 !important;
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea,
input {
    background: #121212 !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    border-color: #343434 !important;
    border-radius: 7px !important;
}

div[data-baseweb="input"] *,
div[data-baseweb="select"] * {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Account button */
div[data-testid="stPopover"] > button {
    min-height: 46px !important;
    border-radius: 8px !important;
    background: #171717 !important;
    border: 1px solid #2b2b2b !important;
    color: #f5f5f5 !important;
    box-shadow: none !important;
}

div[data-testid="stPopover"] > button * {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Right account drawer */
div[data-testid="stPopoverBody"],
div[data-baseweb="popover"] {
    background: #141414 !important;
    color: #f5f5f5 !important;
    border-left: 1px solid #2b2b2b !important;
}

div[data-testid="stPopoverBody"] input,
div[data-baseweb="popover"] input {
    background: #101010 !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    border-color: #343434 !important;
}

div[data-testid="stPopoverBody"] label,
div[data-testid="stPopoverBody"] p,
div[data-testid="stPopoverBody"] span,
div[data-testid="stPopoverBody"] h1,
div[data-testid="stPopoverBody"] h2,
div[data-testid="stPopoverBody"] h3,
div[data-baseweb="popover"] label,
div[data-baseweb="popover"] p,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] h1,
div[data-baseweb="popover"] h2,
div[data-baseweb="popover"] h3 {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

div[data-testid="stPopoverBody"] div[role="radiogroup"] label {
    background: #191919 !important;
    border-color: #303030 !important;
    color: #d4d4d4 !important;
}

div[data-testid="stPopoverBody"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--lc-accent-soft) !important;
    border-color: rgba(255,161,22,.45) !important;
    color: var(--lc-accent) !important;
}

/* Digit-focused sponsored ads — Home page only */
    .digit-sponsored-ad {
        background: #171717 !important;
        border: 1px solid #2b2b2b !important;
        border-radius: 10px !important;
        padding: 12px !important;
        margin-bottom: 12px !important;
        text-align: left !important;
        box-shadow: none !important;
    }

    .digit-ad-top {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        color: #f5f5f5 !important;
        font-size: .68rem !important;
        font-weight: 750 !important;
        margin-bottom: 9px !important;
    }

    .digit-ad-badge {
        background: #1683ff !important;
        color: #ffffff !important;
        border-radius: 5px !important;
        padding: 2px 6px !important;
        font-size: .58rem !important;
        font-weight: 800 !important;
    }

    .digit-ad-icon {
        height: 58px !important;
        border-radius: 7px !important;
        background: #101010 !important;
        border: 1px solid #303030 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 2rem !important;
        margin-bottom: 9px !important;
    }

    .digit-ad-title {
        color: #f5f5f5 !important;
        font-size: .86rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        margin-bottom: 5px !important;
    }

    .digit-ad-desc {
        color: #a3a3a3 !important;
        font-size: .68rem !important;
        line-height: 1.35 !important;
        min-height: 34px !important;
        margin-bottom: 8px !important;
    }

    .digit-ad-button {
        display: inline-block !important;
        background: #1683ff !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 5px 9px !important;
        font-size: .64rem !important;
        font-weight: 800 !important;
    }

    /* Footer and dividers */
.footer {
    margin-top: 42px !important;
    padding-top: 20px !important;
    border-top: 1px solid var(--lc-border) !important;
    color: var(--lc-muted) !important;
    font-size: .78rem !important;
}

hr {
    border-color: var(--lc-border) !important;
}

/* Make the page feel tighter on smaller screens */
@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    div[data-testid="stMarkdownContainer"] h1 {
        font-size: 2.25rem !important;
    }

    div[role="radiogroup"],
    div.row-widget.stRadio > div {
        gap: 5px !important;
        flex-wrap: wrap !important;
    }
}
</style>
""", unsafe_allow_html=True)


# --- TOP SPACING FIX ---
# Keep the header/title + Login row close to the top of the page.
st.markdown("""
<style>
/* Remove Streamlit's default top whitespace above the first content row */
[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 0 !important;
    margin-top: -73px !important;
}

[data-testid="stAppViewContainer"] section.main > div {
    padding-top: 0 !important;
}

[data-testid="stAppViewContainer"] section.main > div[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}

/* Keep the first header/login row from gaining an extra top margin */
[data-testid="stAppViewContainer"] .main .block-container > div:first-child {
    margin-top: 0 !important;
}

/* Keep the account-panel heading fully visible at the top of the drawer */
div[data-testid="stPopoverBody"],
div[data-baseweb="popover"] {
    padding-top: 1.5rem !important;
}

div[data-testid="stPopoverBody"] .drawer-title,
div[data-baseweb="popover"] .drawer-title {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}

div[data-testid="stPopoverBody"] .drawer-sub,
div[data-baseweb="popover"] .drawer-sub {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

</style>
""", unsafe_allow_html=True)

def main():
    model = load_model()
    test_acc, test_loss = get_metrics()

    # HEADER AREA — reference layout, first file colors/content.
    col_logo, col_space, col_login = st.columns([8.7, 0.2, 1.3], vertical_alignment="center")
    with col_logo:
        st.markdown("""
        <div style="background:#1E293B; border:1px solid #334155; border-radius:16px;
                    padding:.75rem 1rem; box-shadow:0 5px 18px rgba(0,0,0,.2);
                    min-height:58px; display:flex; align-items:center;">
            <div style="display:flex; align-items:center; gap:.65rem;">
                <div style="width:38px; height:38px; border-radius:11px; background:rgba(59,130,246,.15);
                            display:flex; align-items:center; justify-content:center; font-size:1.25rem;">✍️</div>
                <div>
                    <div style="font-weight:800; color:#F8FAFC; font-size:1rem; line-height:1.1;">AI HANDWRITTEN DIGIT</div>
                    <div style="color:#94A3B8; font-size:.72rem;">Recognition System</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_login:
        if user:
            render_user_popover(user)
        else:
            render_login_popover()

    st.markdown("<hr style='border-color: #334155; margin: 10px 0;'>", unsafe_allow_html=True)
    if 'nav_selection' not in st.session_state:
        st.session_state.nav_selection = "Home"

    nav_options = ["Home", "Try Your Own Digit", "Model Performance", "About Project"]
    if user:
        nav_options.append("My Profile")
    # Apply programmatic navigation requests BEFORE the navigation widget is created.
    # This avoids StreamlitWidgetAlreadyInstantiatedError when login/logout/profile
    # actions request a page change.
    if "page_request" in st.session_state:
        requested_page = st.session_state.pop("page_request")
        if requested_page in nav_options:
            st.session_state.nav_selection = requested_page

    if st.session_state.nav_selection not in nav_options:
        st.session_state.nav_selection = "Home"

    page = st.radio(
        "", nav_options, horizontal=True, label_visibility="collapsed", key="nav_selection"
    )
    st.markdown("<hr style='border-color: #334155; margin: 10px 0 30px 0;'>", unsafe_allow_html=True)

    def go_to_prediction():
        st.session_state.page_request = "Try Your Own Digit"

    if page == "Home":
        # Home page: keep the existing project content, with digit-focused
        # sponsored ads on both outer sides.
        left_col, main_col, right_col = st.columns([1.5, 6, 1.5], gap="large")

        def sponsored_ad(label, title, desc, icon, cta):
            return f"""
            <div class="digit-sponsored-ad">
                <div class="digit-ad-top">
                    <span>Sponsored</span>
                    <span class="digit-ad-badge">AD</span>
                </div>
                <div class="digit-ad-icon">{icon}</div>
                <div class="digit-ad-title">{title}</div>
                <div class="digit-ad-desc">{desc}</div>
                <div class="digit-ad-button">{cta} →</div>
            </div>
            """

        with left_col:
            st.markdown(
                sponsored_ad("Sponsored", "MNIST Dataset Guide",
                             "Explore handwritten digit datasets and computer-vision learning resources.",
                             "🔢", "Learn More")
                + sponsored_ad("Sponsored", "Digit Writing Practice",
                               "Practice writing 0–9 with simple handwriting exercises.",
                               "✍️", "Try Now")
                + sponsored_ad("Sponsored", "CNN Digit Recognition",
                               "Learn how neural networks recognize handwritten digits.",
                               "🧠", "Explore")
                + sponsored_ad("Sponsored", "Handwriting Samples",
                               "Browse digit examples for AI and image-classification projects.",
                               "🔟", "View Samples"),
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown(
                sponsored_ad("Sponsored", "Handwritten Digit Tools",
                             "Tools and resources for creating clean digit images for ML projects.",
                             "🖊️", "Learn More")
                + sponsored_ad("Sponsored", "MNIST Learning Kit",
                               "Study the classic 0–9 handwritten digit benchmark.",
                               "📚", "Explore")
                + sponsored_ad("Sponsored", "AI Vision Course",
                               "Build practical computer-vision skills with digit recognition.",
                               "🤖", "View Course")
                + sponsored_ad("Sponsored", "Digit Image Library",
                               "Explore handwritten digit examples for educational projects.",
                               "🖼️", "View Library"),
                unsafe_allow_html=True,
            )

        with main_col:
            # HERO
            st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
            st.markdown('<div class="badge">AI • COMPUTER VISION</div>', unsafe_allow_html=True)
            st.markdown('<h1 style="margin-top: 0; font-size: 3.5rem; font-weight: 800; color: #F8FAFC; line-height: 1.2;">AI-Based Handwritten<br>Digit Recognition</h1>', unsafe_allow_html=True)
            st.markdown('<p style="color: #94A3B8; font-size: 1.2rem; max-width: 650px; margin: 20px auto; line-height: 1.6;">A CNN-powered computer vision system that recognizes handwritten digits using the MNIST dataset.</p>', unsafe_allow_html=True)

            st.button("TRY DIGIT RECOGNITION →", type="primary", use_container_width=False, on_click=go_to_prediction)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<hr style="border-color: #334155; margin: 50px 0;">', unsafe_allow_html=True)

            # METRICS
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown('<div class="metric-card"><div class="metric-value">60,000</div><div class="metric-label">Training Images</div></div>', unsafe_allow_html=True)
            mc2.markdown('<div class="metric-card"><div class="metric-value">10,000</div><div class="metric-label">Test Images</div></div>', unsafe_allow_html=True)
            mc3.markdown('<div class="metric-card"><div class="metric-value">10</div><div class="metric-label">Digit Classes</div></div>', unsafe_allow_html=True)
            mc4.markdown('<div class="metric-card"><div class="metric-value">CNN</div><div class="metric-label">Deep Learning Model</div></div>', unsafe_allow_html=True)

            # Existing AI / ML and MACHINE LEARNING information cards,
            # moved into the main content area so the outer columns are ads.
            st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
            info1, info2 = st.columns(2, gap="medium")
            with info1:
                st.markdown("""
                <div class="metric-card" style="height:100%; text-align:left; padding:22px;">
                    <div style="text-align:center; font-size:1.05rem; font-weight:800; color:#F5F5F5;">AI / ML</div>
                    <div style="text-align:center; color:#A3A3A3; font-size:.78rem; margin-top:8px;">COMPUTER VISION</div>
                    <hr style="border-color:#2B2B2B; margin:14px 0;">
                    <div style="font-weight:800; color:#F5F5F5;">CNN</div>
                    <div style="font-weight:800; color:#F5F5F5; margin-top:7px;">MNIST</div>
                    <hr style="border-color:#2B2B2B; margin:14px 0;">
                    <div style="color:#A3A3A3; font-size:.78rem; line-height:1.9;">
                        60K Training Images<br>
                        10K Test Images<br>
                        Digit Classes 0 – 9
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with info2:
                st.markdown("""
                <div class="metric-card" style="height:100%; text-align:center; padding:22px;">
                    <div style="font-size:1.05rem; font-weight:800; color:#F5F5F5;">MACHINE LEARNING</div>
                    <div style="color:#A3A3A3; font-size:.78rem; margin-top:8px;">HOW IT WORKS</div>
                    <hr style="border-color:#2B2B2B; margin:14px 0;">
                    <div style="color:#A3A3A3; font-size:.78rem; line-height:1.75;">
                        IMAGE<br><span style="color:#FFA116;">↓</span><br>
                        PREPROCESSING<br><span style="color:#FFA116;">↓</span><br>
                        CNN<br><span style="color:#FFA116;">↓</span><br>
                        PREDICTION<br><span style="color:#FFA116;">↓</span><br>
                        CONFIDENCE SCORE
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # HOW IT WORKS
            st.markdown('<br><h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 24px; text-align: center;">HOW IT WORKS</h3>', unsafe_allow_html=True)
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.markdown('<div class="metric-card" style="height:100%;"><div class="step-number">01</div><div class="step-title">INPUT</div><div class="step-desc">Upload handwritten digit</div></div>', unsafe_allow_html=True)
            hc2.markdown('<div class="metric-card" style="height:100%;"><div class="step-number">02</div><div class="step-title">PREPROCESSING</div><div class="step-desc">Resize + normalize</div></div>', unsafe_allow_html=True)
            hc3.markdown('<div class="metric-card" style="height:100%;"><div class="step-number">03</div><div class="step-title">CNN</div><div class="step-desc">Feature extraction</div></div>', unsafe_allow_html=True)
            hc4.markdown('<div class="metric-card" style="height:100%;"><div class="step-number">04</div><div class="step-title">PREDICTION</div><div class="step-desc">Digit + confidence</div></div>', unsafe_allow_html=True)

            # TECHNOLOGIES
            st.markdown('<br><h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 24px; text-align: center;">TECHNOLOGIES</h3>', unsafe_allow_html=True)
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.markdown('<div class="metric-card" style="padding:15px;"><div style="color:#F8FAFC; font-weight:bold; font-size:1.1rem;">TensorFlow</div><div class="step-desc" style="margin-top:5px;">Deep Learning</div></div>', unsafe_allow_html=True)
            tc2.markdown('<div class="metric-card" style="padding:15px;"><div style="color:#F8FAFC; font-weight:bold; font-size:1.1rem;">Keras</div><div class="step-desc" style="margin-top:5px;">Neural Network API</div></div>', unsafe_allow_html=True)
            tc3.markdown('<div class="metric-card" style="padding:15px;"><div style="color:#F8FAFC; font-weight:bold; font-size:1.1rem;">CNN</div><div class="step-desc" style="margin-top:5px;">Image Classification</div></div>', unsafe_allow_html=True)
            tc4.markdown('<div class="metric-card" style="padding:15px;"><div style="color:#F8FAFC; font-weight:bold; font-size:1.1rem;">MNIST</div><div class="step-desc" style="margin-top:5px;">Handwritten Digits</div></div>', unsafe_allow_html=True)

    elif page == "Model Performance":
        st.markdown('<div style="max-width: 1000px; margin: 0 auto;">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #F8FAFC; font-weight: 800; font-size: 2.5rem; text-align: center;">MODEL PERFORMANCE</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94A3B8; font-size: 1.1rem; text-align: center;">Evaluation results of the trained CNN on unseen MNIST test data.</p>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color: #334155; margin: 30px 0;">', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{test_acc:.2f}%</div><div class="metric-label">TEST ACCURACY</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{test_loss:.4f}</div><div class="metric-label">TEST LOSS</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">60,000</div><div class="metric-label">TRAINING IMAGES</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value">10,000</div><div class="metric-label">TEST IMAGES</div></div>', unsafe_allow_html=True)
        
        st.markdown('<br><h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 24px; text-align: center;">TRAINING PERFORMANCE</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card" style="padding:15px;">', unsafe_allow_html=True)
            if os.path.exists("outputs/accuracy_curve.png"):
                st.image("outputs/accuracy_curve.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card" style="padding:15px;">', unsafe_allow_html=True)
            if os.path.exists("outputs/loss_curve.png"):
                st.image("outputs/loss_curve.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 10px; text-align: center;">CONFUSION MATRIX</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94A3B8; margin-bottom: 24px; text-align: center;">Actual vs predicted digit classifications across the MNIST test dataset. The diagonal represents correctly classified digits, while off-diagonal values represent classification errors.</p>', unsafe_allow_html=True)
        
        col_cm1, col_cm2, col_cm3 = st.columns([1, 4, 1])
        with col_cm2:
            st.markdown('<div class="metric-card" style="padding:15px;">', unsafe_allow_html=True)
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<br><h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 24px; text-align: center;">MODEL ARCHITECTURE</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-box"><strong>INPUT</strong><br><span style="color:#94A3B8; font-size:0.9rem;">28 × 28 × 1</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>CONVOLUTION</strong><br><span style="color:#94A3B8; font-size:0.9rem;">32 Filters</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>MAX POOLING</strong></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>CONVOLUTION</strong><br><span style="color:#94A3B8; font-size:0.9rem;">64 Filters</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>MAX POOLING</strong></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>FLATTEN</strong></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>DENSE</strong><br><span style="color:#94A3B8; font-size:0.9rem;">128 Neurons</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>DROPOUT</strong><br><span style="color:#94A3B8; font-size:0.9rem;">0.5</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box"><strong>OUTPUT</strong><br><span style="color:#94A3B8; font-size:0.9rem;">10 Classes</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Try Your Own Digit":
        # Keep the page heading close to the upper divider without changing the rest of the layout.
        st.markdown('<div style="max-width: 1100px; margin: -45px auto 0 auto;">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #F8FAFC; font-weight: 800; font-size: 2.5rem; text-align: center; margin-top: 0;">Recognize Your Digit</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94A3B8; font-size: 1.1rem; text-align: center; margin-top: 8px;">Draw a digit with your mouse or upload an image, then let the CNN identify it.</p>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color: #334155; margin: 18px 0 30px 0;">', unsafe_allow_html=True)

        if model is None:
            st.error("Model not found. Please run `python train_model.py` first.")
            return

        # Keep the drawing canvas resettable without changing the ML pipeline.
        if "canvas_version" not in st.session_state:
            st.session_state.canvas_version = 0

        st.markdown("""
        <div style="background:#1E293B; border:1px solid #334155; border-radius:18px; padding:22px; margin-bottom:18px;">
            <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">DRAW OR UPLOAD</div>
            <div style="font-size:.88rem; color:#94A3B8; margin-top:5px;">Write one handwritten digit inside the canvas, or choose a PNG/JPG/JPEG image.</div>
        </div>
        """, unsafe_allow_html=True)

        draw_col, upload_col = st.columns([2.5, 1], gap="large")

        with draw_col:
            # The drawable-canvas toolbar lives inside its component iframe.
            # Give that iframe a light background so its black toolbar icons
            # remain visible on the dark application theme.
            st.markdown("""
            <style>
            iframe[title*="streamlit_drawable_canvas"],
            iframe[title*="st_canvas"] {
                background: #F8FAFC !important;
                border-radius: 0 0 10px 10px !important;
            }
            div[data-testid="stCustomComponentV1"] {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                border-radius: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<div style="color:#94A3B8; font-size:.78rem; font-weight:800; letter-spacing:1px; margin-bottom:8px;">DRAWING CANVAS</div>', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=18,
                # Dark drawing canvas with a bright stroke for clear handwriting.
                # This also matches the application's dark visual theme.
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=420,
                width=620,
                drawing_mode="freedraw",
                key=f"digit_canvas_{st.session_state.canvas_version}",
            )
            st.caption("Use your mouse to write a single digit. For best results, write it large and centered.")

        with upload_col:
            st.markdown('<div class="metric-card" style="padding:24px; height:100%; box-sizing:border-box;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label" style="margin-bottom:12px;">UPLOAD IMAGE</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#94A3B8; font-size:.85rem; line-height:1.5;">Prefer an existing handwritten digit? Upload it here.</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "",
                type=["png", "jpg", "jpeg"],
                key="digit_upload",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                st.success("Image ready")
            st.markdown('</div>', unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
        with action_col1:
            if st.button("CLEAR CANVAS", use_container_width=True, key="clear_digit_canvas"):
                st.session_state.canvas_version += 1
                st.session_state.pop("live_canvas_image", None)
                st.session_state.pop("prediction_source_image", None)
                st.session_state.pop("prediction_source_label", None)
                st.rerun()
        with action_col2:
            recognize_clicked = st.button("RECOGNIZE DIGIT", type="primary", use_container_width=True, key="recognize_digit")
        with action_col3:
            if uploaded_file is not None:
                st.info("Upload selected — the uploaded image will be used for prediction.")
            else:
                st.caption("No upload selected — the drawing canvas will be used.")

        image_to_predict = None
        source_label = "DRAWN INPUT"

        # Capture the current canvas as soon as it changes. This keeps the
        # drawing available in session state when the separate
        # "RECOGNIZE DIGIT" button causes a Streamlit rerun.
        if canvas_result.image_data is not None:
            canvas_array = canvas_result.image_data[:, :, :3].astype("uint8")
            # The canvas is light (#F8FAFC) with dark strokes, so a dark
            # pixel is a reliable indication that the user actually drew.
            if np.min(canvas_array) < 220:
                live_canvas_image = Image.fromarray(canvas_array, mode="RGB")
                st.session_state.live_canvas_image = live_canvas_image

        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file).convert("RGB")
            source_label = "UPLOADED INPUT"
        elif "live_canvas_image" in st.session_state:
            image_to_predict = st.session_state.live_canvas_image
            source_label = "DRAWN INPUT"

        if recognize_clicked:
            if image_to_predict is None:
                st.warning("Please draw a digit on the canvas or upload an image first.")
            else:
                # Persist the exact image used for this recognition run so
                # both MODEL INPUT cards stay filled after the rerun.
                st.session_state.prediction_source_image = image_to_predict
                st.session_state.prediction_source_label = source_label

        # Always keep the result area useful — show informative placeholders
        # before recognition, then replace them with the real model output.
        if "prediction_source_image" in st.session_state:
            image_to_predict = st.session_state.prediction_source_image
            source_label = st.session_state.get("prediction_source_label", "INPUT")

            # Keep MODEL INPUT close to the divider above it.
            st.markdown('<hr style="border-color:#334155; margin:8px 0 10px 0;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color:#F8FAFC; font-weight:700; text-align:center; margin:0 0 10px 0;">MODEL INPUT</h3>', unsafe_allow_html=True)

            processed_input, display_img = preprocess_image(image_to_predict)

            # Keep both MODEL INPUT previews exactly the same visual size.
            # The CNN still receives the original 28 × 28 tensor; these are
            # display-only previews inside equal 620 × 420 frames.
            preview_size = (620, 420)
            source_preview = Image.new("RGB", preview_size, "black")
            source_fit = ImageOps.contain(image_to_predict.convert("RGB"), preview_size)
            source_preview.paste(
                source_fit,
                ((preview_size[0] - source_fit.width) // 2,
                 (preview_size[1] - source_fit.height) // 2),
            )

            model_preview_raw = Image.fromarray(
                np.clip(display_img * 255, 0, 255).astype("uint8"), mode="L"
            ).convert("RGB")
            # Display the 28 × 28 model input across the exact same
            # visual area as the drawn input. This is display-only; the
            # CNN still receives the original 28 × 28 tensor above.
            model_preview = model_preview_raw.resize(
                preview_size, Image.Resampling.NEAREST
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="metric-card" style="height:100%; min-height:250px;">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label" style="margin-bottom:15px;">{source_label}</div>', unsafe_allow_html=True)
                st.image(source_preview, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="metric-card" style="height:100%; min-height:250px;">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="margin-bottom:15px;">28 × 28 MODEL INPUT</div>', unsafe_allow_html=True)
                st.image(model_preview, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner("Analyzing digit..."):
                prediction = model.predict(processed_input, verbose=0)
                predicted_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction) * 100)

            st.markdown('<br>', unsafe_allow_html=True)
            c_res1, c_res2 = st.columns([1, 1.5])
            with c_res1:
                st.markdown(f"""
                <div class="pred-card">
                    <div class="metric-label">PREDICTED DIGIT</div>
                    <div class="pred-value">{predicted_digit}</div>
                    <div class="pred-conf">Confidence<br><strong style="color:#F8FAFC; font-size:1.5rem;">{confidence:.2f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with c_res2:
                prob_dict = {i: float(prediction[0][i]) * 100 for i in range(10)}
                probability_rows = ''.join(
                    f'''<div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                        <span style="width:18px; color:#A3A3A3; font-size:.78rem; font-weight:700;">{i}</span>
                        <div style="flex:1; height:9px; background:#2A2A2A; border-radius:99px; overflow:hidden;">
                            <div style="width:{min(prob, 100):.2f}%; height:100%; background:{'#FFA116' if i == predicted_digit else '#6B7280'}; border-radius:99px;"></div>
                        </div>
                        <span style="width:54px; text-align:right; color:#D4D4D4; font-size:.72rem; font-weight:700;">{prob:.2f}%</span>
                    </div>'''
                    for i, prob in prob_dict.items()
                )
                st.markdown(f'''
                <div class="metric-card" style="text-align:left; padding:24px 30px; min-height:100%;">
                    <h4 style="margin:0 0 14px 0; color:#F5F5F5; text-align:center;">PROBABILITY DISTRIBUTION</h4>
                    {probability_rows}
                </div>
                ''', unsafe_allow_html=True)

            st.markdown(
                '<div style="background:#162032; border:1px solid #334155; border-radius:14px; padding:16px 20px; margin-top:18px; color:#94A3B8; font-size:.88rem; text-align:center;">The CNN analyzes the input after converting it to the 28 × 28 grayscale format used during MNIST training.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Helpful empty-state content instead of blank cards.
            st.markdown('<hr style="border-color:#334155; margin:0 0 8px 0;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color:#F8FAFC; font-weight:700; text-align:center; margin:0 0 10px 0;">MODEL INPUT</h3>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('''
                <div class="metric-card" style="min-height:250px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:30px;">
                    <div style="font-size:2.4rem; margin-bottom:12px;">✍️</div>
                    <div style="color:#F5F5F5; font-size:1rem; font-weight:800; letter-spacing:.5px;">DRAWN / UPLOADED INPUT</div>
                    <div style="color:#A3A3A3; font-size:.85rem; line-height:1.5; margin-top:9px; max-width:300px;">Your selected handwritten digit will appear here after you draw or upload an image.</div>
                    <div style="color:#FFA116; font-size:.75rem; font-weight:700; margin-top:16px;">STEP 1 • PROVIDE A DIGIT</div>
                </div>
                ''', unsafe_allow_html=True)

            with c2:
                st.markdown('''
                <div class="metric-card" style="min-height:250px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:30px;">
                    <div style="font-size:2.4rem; margin-bottom:12px;">▦</div>
                    <div style="color:#F5F5F5; font-size:1rem; font-weight:800; letter-spacing:.5px;">28 × 28 MODEL INPUT</div>
                    <div style="color:#A3A3A3; font-size:.85rem; line-height:1.5; margin-top:9px; max-width:300px;">The CNN will convert your digit to grayscale, resize it to 28 × 28, and normalize the pixels.</div>
                    <div style="color:#FFA116; font-size:.75rem; font-weight:700; margin-top:16px;">STEP 2 • PREPROCESS IMAGE</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            c_res1, c_res2 = st.columns([1, 1.5])
            with c_res1:
                st.markdown('''
                <div class="pred-card" style="min-height:210px; display:flex; flex-direction:column; justify-content:center;">
                    <div class="metric-label">PREDICTED DIGIT</div>
                    <div class="pred-value" style="color:#555 !important; -webkit-text-fill-color:#555 !important;">—</div>
                    <div class="pred-conf">Waiting for a digit<br><strong style="color:#A3A3A3; font-size:1rem;">Click “RECOGNIZE DIGIT” after drawing or uploading.</strong></div>
                </div>
                ''', unsafe_allow_html=True)

            with c_res2:
                probability_rows = ''.join(
                    f'''<div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                        <span style="width:18px; color:#A3A3A3; font-size:.78rem; font-weight:700;">{i}</span>
                        <div style="flex:1; height:8px; background:#2A2A2A; border-radius:99px; overflow:hidden;">
                            <div style="width:4%; height:100%; background:#555; border-radius:99px;"></div>
                        </div>
                        <span style="width:44px; text-align:right; color:#777; font-size:.72rem;">—</span>
                    </div>'''
                    for i in range(10)
                )
                st.markdown(f'''
                <div class="metric-card" style="text-align:left; padding:24px 30px; min-height:210px;">
                    <h4 style="margin:0 0 5px 0; color:#F5F5F5; text-align:center;">PROBABILITY DISTRIBUTION</h4>
                    <div style="color:#777; font-size:.75rem; text-align:center; margin-bottom:10px;">Class probabilities will appear after recognition</div>
                    {probability_rows}
                </div>
                ''', unsafe_allow_html=True)

            st.markdown(
                '<div style="background:#171717; border:1px solid #2B2B2B; border-radius:12px; padding:16px 20px; margin-top:18px; color:#A3A3A3; font-size:.88rem; text-align:center;"><strong style="color:#F5F5F5;">Ready for input.</strong> Draw one digit on the canvas or upload a handwritten image, then press <strong style="color:#FFA116;">RECOGNIZE DIGIT</strong> to run the CNN.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "My Profile":
        profile_user = get_user(st.session_state.user_id)
        if not profile_user:
            st.session_state.user_id = None
            st.session_state.page_request = "Home"
            st.rerun()

        st.markdown('<div style="max-width: 1000px; margin: 0 auto;">', unsafe_allow_html=True)
        st.markdown('<h1 style="color:#F8FAFC; font-weight:800; font-size:2.5rem; text-align:center;">MY PROFILE</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94A3B8; font-size:1.05rem; text-align:center;">Manage your local account details and password.</p>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#334155; margin:30px 0;">', unsafe_allow_html=True)

        initials = "".join(part[0].upper() for part in profile_user["full_name"].split()[:2]) or profile_user["username"][0].upper()
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(
                f'''<div class="profile-card">
                    <div class="avatar">{initials}</div>
                    <div style="height:.8rem"></div>
                    <div class="profile-name">{profile_user["full_name"]}</div>
                    <div class="profile-muted">@{profile_user["username"]}</div>
                    <div class="profile-muted" style="margin-top:.55rem;">Member since {profile_user["created_at"][:10]}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        with c2:
            with st.form("profile_form"):
                st.markdown("#### Account Details")
                full_name = st.text_input("Full Name", value=profile_user["full_name"])
                email = st.text_input("Email", value=profile_user["email"])
                phone = st.text_input("Phone", value=profile_user["phone"])
                st.text_input("Username", value=profile_user["username"], disabled=True)
                saved = st.form_submit_button("💾 Save Profile", type="primary", width="stretch")
                if saved:
                    clean_email = email.strip().lower()
                    if not full_name.strip() or not valid_email(clean_email):
                        st.error("Please provide a valid name and email.")
                    else:
                        ok, msg = update_profile(profile_user["id"], full_name.strip(), clean_email, phone.strip())
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        st.markdown("### 🔐 Change Password")
        with st.form("password_form"):
            old_pw = st.text_input("Current Password", type="password")
            new_pw = st.text_input("New Password", type="password")
            confirm_pw = st.text_input("Confirm New Password", type="password")
            changed = st.form_submit_button("🔑 Change Password", type="primary", width="stretch")
            if changed:
                if len(new_pw) < 6:
                    st.error("New password must contain at least 6 characters.")
                elif new_pw != confirm_pw:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = change_password(profile_user["id"], old_pw, new_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown("### Session")
        if st.button("🚪 Log Out", type="secondary"):
            st.session_state.user_id = None
            st.session_state.page_request = "Home"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "About Project":
        # Professional academic project overview. MNIST facts are based on the
        # original MNIST documentation; evaluation values are read from this project.
        st.markdown('<div style="max-width:1180px; margin:0 auto;">', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; padding:18px 0 30px;">
            <div style="display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border:1px solid #3a3a3a; border-radius:999px; background:#181818; color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px;">
                <span style="width:7px; height:7px; border-radius:50%; background:#2db55d; display:inline-block;"></span>
                ACADEMIC COMPUTER VISION PROJECT
            </div>
            <h1 style="color:#F8FAFC; font-size:2.85rem; font-weight:850; margin:16px 0 10px; letter-spacing:-.8px;">About the Project</h1>
            <p style="color:#a3a3a3; font-size:1.02rem; max-width:820px; margin:0 auto; line-height:1.7;">An end-to-end handwritten digit recognition system that combines a convolutional neural network with an interactive Streamlit application for image classification and model evaluation.</p>
        </div>

        <div style="display:grid; grid-template-columns:1.25fr .75fr; gap:18px; margin-bottom:18px;">
            <div class="metric-card" style="padding:28px; border:1px solid #2b2b2b;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:11px;">PROJECT OVERVIEW</div>
                <h2 style="color:#F8FAFC; font-size:1.45rem; margin:0 0 12px;">From handwritten input to an AI prediction</h2>
                <p style="color:#b8b8b8; font-size:.96rem; line-height:1.75; margin:0;">The application lets a user draw or upload a handwritten digit. The image is converted to grayscale, resized to the model's 28 × 28 input format, normalized, and passed to the trained CNN. The system then returns the most likely digit together with the class-probability distribution.</p>
            </div>
            <div class="metric-card" style="padding:28px; border:1px solid #2b2b2b;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:11px;">PROJECT STATUS</div>
                <div style="color:#2db55d; font-size:1.65rem; font-weight:850; margin-bottom:6px;">Model Evaluated</div>
                <p style="color:#a3a3a3; font-size:.88rem; line-height:1.6; margin:0;">The values below are connected to the saved evaluation metrics used by this application.</p>
                <div style="display:flex; gap:10px; margin-top:18px;">
                    <div style="flex:1; background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><div style="color:#F8FAFC; font-size:1.2rem; font-weight:800;">{test_acc:.2f}%</div><div style="color:#737373; font-size:.68rem; margin-top:4px;">TEST ACCURACY</div></div>
                    <div style="flex:1; background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><div style="color:#F8FAFC; font-size:1.2rem; font-weight:800;">{test_loss:.4f}</div><div style="color:#737373; font-size:.68rem; margin-top:4px;">TEST LOSS</div></div>
                </div>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:18px;">
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:12px;">OBJECTIVE</div>
                <h3 style="color:#F8FAFC; margin:0 0 10px; font-size:1.18rem;">Reliable 10-class digit classification</h3>
                <p style="color:#a3a3a3; line-height:1.7; margin:0;">Train a CNN to learn visual patterns in handwritten digits and classify an input image into one of the ten classes: <strong style="color:#f5f5f5;">0–9</strong>.</p>
            </div>
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:12px;">APPLICATION FLOW</div>
                <h3 style="color:#F8FAFC; margin:0 0 10px; font-size:1.18rem;">Input → Preprocessing → CNN → Result</h3>
                <p style="color:#a3a3a3; line-height:1.7; margin:0;">The same preprocessing path is applied before inference so the user's image is presented to the network in the expected numerical format.</p>
            </div>
        </div>

        <div class="metric-card" style="padding:28px; margin-bottom:18px;">
            <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:8px;">MNIST DATASET</div>
            <h2 style="color:#F8FAFC; font-size:1.38rem; margin:0 0 8px;">A standard benchmark for handwritten digit recognition</h2>
            <p style="color:#a3a3a3; font-size:.9rem; line-height:1.65; margin:0 0 20px;">The original MNIST database contains 70,000 handwritten digit images: 60,000 for training and 10,000 for testing. Each sample is a centered, size-normalized 28 × 28 grayscale image and belongs to one of ten classes (0 through 9).</p>
            <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:11px;">
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="color:#ffa116; font-size:1.35rem; font-weight:850;">70,000</div><div style="color:#737373; font-size:.68rem; margin-top:5px;">TOTAL IMAGES</div></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="color:#F8FAFC; font-size:1.35rem; font-weight:850;">60,000</div><div style="color:#737373; font-size:.68rem; margin-top:5px;">TRAINING</div></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="color:#F8FAFC; font-size:1.35rem; font-weight:850;">10,000</div><div style="color:#737373; font-size:.68rem; margin-top:5px;">TESTING</div></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="color:#F8FAFC; font-size:1.35rem; font-weight:850;">28 × 28</div><div style="color:#737373; font-size:.68rem; margin-top:5px;">PIXELS / IMAGE</div></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="color:#F8FAFC; font-size:1.35rem; font-weight:850;">10</div><div style="color:#737373; font-size:.68rem; margin-top:5px;">CLASSES</div></div>
            </div>
            <div style="margin-top:17px; padding:13px 15px; background:#181818; border-left:3px solid #ffa116; border-radius:7px; color:#a3a3a3; font-size:.82rem; line-height:1.6;">Source: MNIST database documentation by Yann LeCun, Corinna Cortes and Christopher J. C. Burges. The dataset was constructed from NIST handwritten-digit data.</div>
        </div>

        <div class="metric-card" style="padding:28px; margin-bottom:18px;">
            <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:17px;">MODEL PIPELINE</div>
            <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:10px; align-items:stretch;">
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="font-size:1.15rem; margin-bottom:7px;">01</div><strong style="color:#F8FAFC; font-size:.86rem;">INPUT</strong><br><span style="color:#888; font-size:.75rem;">Draw / Upload</span></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="font-size:1.15rem; margin-bottom:7px;">02</div><strong style="color:#F8FAFC; font-size:.86rem;">PREPROCESS</strong><br><span style="color:#888; font-size:.75rem;">28 × 28 • Normalize</span></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="font-size:1.15rem; margin-bottom:7px;">03</div><strong style="color:#F8FAFC; font-size:.86rem;">FEATURE LEARNING</strong><br><span style="color:#888; font-size:.75rem;">Convolution + Pooling</span></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="font-size:1.15rem; margin-bottom:7px;">04</div><strong style="color:#F8FAFC; font-size:.86rem;">CLASSIFY</strong><br><span style="color:#888; font-size:.75rem;">10 output classes</span></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:17px 10px; text-align:center;"><div style="font-size:1.15rem; margin-bottom:7px;">05</div><strong style="color:#F8FAFC; font-size:.86rem;">RESULT</strong><br><span style="color:#888; font-size:.75rem;">Prediction + probabilities</span></div>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:1.05fr .95fr; gap:18px; margin-bottom:18px;">
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:14px;">CNN DESIGN</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    <div style="background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><strong style="color:#F8FAFC;">Input</strong><br><span style="color:#888; font-size:.78rem;">28 × 28 × 1</span></div>
                    <div style="background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><strong style="color:#F8FAFC;">Convolution</strong><br><span style="color:#888; font-size:.78rem;">32 filters</span></div>
                    <div style="background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><strong style="color:#F8FAFC;">Convolution</strong><br><span style="color:#888; font-size:.78rem;">64 filters</span></div>
                    <div style="background:#111; border:1px solid #2b2b2b; border-radius:10px; padding:13px;"><strong style="color:#F8FAFC;">Dense</strong><br><span style="color:#888; font-size:.78rem;">128 neurons</span></div>
                </div>
                <p style="color:#a3a3a3; font-size:.84rem; line-height:1.6; margin:14px 0 0;">The application is built around the trained CNN stored as <code style="color:#ffa116;">models/mnist_cnn.keras</code>. The model uses convolutional feature extraction followed by dense classification for the ten digit classes.</p>
            </div>
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:14px;">EVALUATION</div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:18px; margin-bottom:11px;"><div style="color:#2db55d; font-size:1.7rem; font-weight:850;">{test_acc:.2f}%</div><div style="color:#888; font-size:.7rem; letter-spacing:.7px; margin-top:4px;">CNN TEST ACCURACY</div></div>
                <div style="background:#111; border:1px solid #2b2b2b; border-radius:11px; padding:18px;"><div style="color:#F8FAFC; font-size:1.7rem; font-weight:850;">{test_loss:.4f}</div><div style="color:#888; font-size:.7rem; letter-spacing:.7px; margin-top:4px;">CNN TEST LOSS</div></div>
                <p style="color:#94A3B8; font-size:.78rem; line-height:1.55; margin:13px 0 0;">These values are read from the project's saved evaluation file rather than being hard-coded into this page.</p>
            </div>
        </div>

        <div class="metric-card" style="padding:26px; margin-bottom:18px;">
            <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:14px;">TECHNOLOGY STACK</div>
            <div style="display:flex; flex-wrap:wrap; gap:9px;">
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">Python</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">TensorFlow</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">Keras</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">Streamlit</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">NumPy</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">Pillow</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">Matplotlib</span>
                <span style="border:1px solid #343434; background:#111; color:#F8FAFC; padding:8px 12px; border-radius:8px; font-size:.84rem;">SQLite</span>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:20px;">
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:12px;">ACADEMIC VALUE</div>
                <p style="color:#b8b8b8; line-height:1.7; margin:0;">The project demonstrates the complete machine-learning lifecycle: dataset understanding, image preprocessing, CNN training, quantitative evaluation, model inference, and deployment through an interactive web interface.</p>
            </div>
            <div class="metric-card" style="padding:26px;">
                <div style="color:#ffa116; font-size:.76rem; font-weight:800; letter-spacing:1.2px; margin-bottom:12px;">SCOPE & LIMITATIONS</div>
                <p style="color:#b8b8b8; line-height:1.7; margin:0;">MNIST is a controlled benchmark. A user's handwriting can differ in scale, stroke thickness, positioning, or style, so an excellent MNIST test score does not guarantee perfect performance on every real-world handwritten image.</p>
            </div>
        </div>

        <div style="text-align:center; border-top:1px solid #242424; padding:18px 0 8px;">
            <p style="color:#666; font-size:.74rem; line-height:1.6; margin:0;">Dataset facts: MNIST database documentation • Model metrics: this project's saved evaluation results • Purpose: academic demonstration</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        <strong style="color: #F8FAFC;">AI-Based Handwritten Digit Recognition System</strong><br>
        CNN • MNIST • TensorFlow/Keras<br>
        <span style="opacity: 0.7;">Academic Machine Learning Project</span>
    </div>
    """, unsafe_allow_html=True)


# --- FINAL VISIBILITY FIX ---
# Presentation-only override for the dark theme. Keeps application/ML logic unchanged.
st.markdown("""
<style>
/* Force visible text where the older light-theme selectors were more specific. */
div[data-testid="stMarkdownContainer"] h1[style],
div[data-testid="stMarkdownContainer"] h2[style],
div[data-testid="stMarkdownContainer"] h3[style],
div[data-testid="stMarkdownContainer"] h4[style] {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Home hero */
div[data-testid="stMarkdownContainer"] h1[style*="F8FAFC"] {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
div[data-testid="stMarkdownContainer"] p[style*="94A3B8"] {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Side information cards */
.ad-card h4 {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}
.ad-card p {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}
.ad-card p[style*="F8FAFC"] {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.ad-card p span[style*="3B82F6"] {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}

/* Cards and project content */
.metric-card,
.metric-card div,
.metric-card p,
.metric-card span,
.metric-card strong {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.metric-card .metric-label,
.metric-card .step-desc {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}
.metric-card .metric-value,
.metric-card .step-number {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}

/* Preserve muted inline text */
div[data-testid="stMarkdownContainer"] p[style*="94A3B8"],
div[data-testid="stMarkdownContainer"] span[style*="94A3B8"] {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Keep the main dark page background and surfaces consistent */
.stApp {
    background: #0f0f0f !important;
}
</style>
""", unsafe_allow_html=True)


# --- FINAL DARK-THEME TEXT VISIBILITY OVERRIDE ---
# Presentation-only: restores readable contrast across all pages without
# changing navigation, authentication, canvas, preprocessing, or ML logic.
st.markdown("""
<style>
:root {
    --primary-bg: #0f0f0f !important;
    --card-bg: #171717 !important;
    --text-primary: #f5f5f5 !important;
    --text-secondary: #a3a3a3 !important;
    --accent: #ffa116 !important;
    --success: #2db55d !important;
    --border: #2b2b2b !important;
}

/* Base Streamlit markdown text */
.stMarkdown p,
.stMarkdown li,
.stMarkdown div,
.stMarkdown span {
    color: #f5f5f5;
}

/* Headings */
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* All page descriptions / explanatory text */
div[data-testid="stMarkdownContainer"] p[style],
div[data-testid="stMarkdownContainer"] div[style] {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Muted text used by the application */
div[data-testid="stMarkdownContainer"] p[style*="94A3B8"],
div[data-testid="stMarkdownContainer"] span[style*="94A3B8"] {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Blue inline headings from the old theme -> visible orange accent */
div[data-testid="stMarkdownContainer"] h3[style*="3B82F6"],
div[data-testid="stMarkdownContainer"] h4[style*="3B82F6"] {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}

/* Home side cards */
.ad-card,
.ad-card * {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.ad-card h4 {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}
.ad-card p {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}
.ad-card p[style*="F8FAFC"] {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Metric / information cards */
.metric-card,
.metric-card div,
.metric-card p,
.metric-card span,
.metric-card strong {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.metric-card .metric-value,
.metric-card .step-number {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}
.metric-card .metric-label,
.metric-card .step-desc {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Architecture cards */
.arch-box,
.arch-box strong {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.arch-box span {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}
.arch-arrow {
    color: #ffa116 !important;
    -webkit-text-fill-color: #ffa116 !important;
}

/* Try Your Own Digit */
.pred-card,
.pred-card * {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
.pred-card .pred-value {
    color: #2db55d !important;
    -webkit-text-fill-color: #2db55d !important;
}
.pred-card .pred-conf {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Canvas/upload captions and labels */
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] * {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}
section[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] * {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}
section[data-testid="stFileUploaderDropzone"] small {
    color: #a3a3a3 !important;
    -webkit-text-fill-color: #a3a3a3 !important;
}

/* Model Performance / About Project text */
div[data-testid="stMarkdownContainer"] {
    color: #f5f5f5;
}
div[data-testid="stMarkdownContainer"] > p {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Keep dividers visible */
hr {
    border-color: #2b2b2b !important;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
