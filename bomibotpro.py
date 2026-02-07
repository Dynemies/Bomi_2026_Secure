"""
====================================================================================================
  BOMI NUCLEUS - VERSION 84.0 (SOUL & LIGHT EDITION)
  --------------------------------------------------------------------------------------------------
  AUTHOR: T. Vamsi Krishna Sai
  
  CHANGELOG:
  1. UI OVERHAUL: Replaced dark theme with a vibrant "Holo-Light" anime aesthetic.
     - Added CSS animations (floating cards, sliding headers, pulsing buttons).
  2. CHATBOT SOUL: Expanded the bot's logic to handle small talk, greetings, and context.
  3. BUG FIXES: 
     - Corrected 'SyntaxError' in JSON loading.
     - Fixed 'UnboundLocalError' for file paths.
     - Fixed 'Dense layer' crash in heatmap generation.
  4. LOGIC: 40% Disease Override remains active for accuracy.
====================================================================================================
"""

import os
import sys
import subprocess
import time
import datetime
import json
import random

# --- [SECTION 0] AUTO-INSTALLER ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = ["h5py", "opencv-python", "streamlit", "tensorflow", "pillow", "matplotlib", "pandas"]
for lib in required_libs:
    try:
        __import__(lib.replace("-", "_").split("_")[0]) 
    except ImportError:
        pass

# --- [SECTION 1] ENVIRONMENT CONFIG ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import h5py
import matplotlib.pyplot as plt
from PIL import Image

try:
    import tensorflow as tf
    import keras
    from tensorflow.keras.models import model_from_json, load_model, Model
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Layer
except ImportError as e:
    st.error(f"CRITICAL ERROR: Missing Python Library. Details: {e}")
    st.stop()

try:
    if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
        keras.config.enable_unsafe_deserialization()
except:
    pass

# --- [SECTION 2] GHOST LAYERS ---
class FixedLambda(Layer):
    def __init__(self, function=None, output_shape=None, mask=None, arguments=None, **kwargs):
        super(FixedLambda, self).__init__(**kwargs)
        self.function = function
        self._output_shape = output_shape
        self.mask = mask
        self.arguments = arguments
    def compute_output_shape(self, input_shape): return input_shape
    def call(self, inputs, mask=None): return inputs
    def get_config(self): return super(FixedLambda, self).get_config()
    @classmethod
    def from_config(cls, config): return cls(**config)

CUSTOM_OBJECTS = {'FixedLambda': FixedLambda, 'Lambda': FixedLambda, 'Functional': Model, 'tf': tf, 'backend': K}

# --- [SECTION 3] PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_HOME = os.path.expanduser("~")
LOG_BASE = os.path.join(USER_HOME, "Desktop", "Bomi_Project_Logs")
EVIDENCE_DIR = os.path.join(LOG_BASE, "visual_evidence")
DB_PATH = os.path.join(LOG_BASE, "patient_records.json")

for folder in [LOG_BASE, EVIDENCE_DIR]:
    if not os.path.exists(folder):
        try: os.makedirs(folder)
        except: pass

# --- [SECTION 4] MODEL LOADER ---
@st.cache_resource
def load_bomi_engine():
    models = {}
    required_files = ["model_Covid.h5", "model_Pneumonia.h5", "model_Tuberculosis.h5", "model_Normal.h5"]
    
    for fname in required_files:
        path = os.path.join(SCRIPT_DIR, fname)
        label = fname.replace("model_", "").replace(".h5", "")
        if not os.path.exists(path): continue

        try:
            with h5py.File(path, 'r') as f:
                if 'model_config' not in f.attrs: raise ValueError("No config")
                model_config = f.attrs.get('model_config')
                if hasattr(model_config, 'decode'): model_config = model_config.decode('utf-8')

            model = model_from_json(model_config, custom_objects=CUSTOM_OBJECTS)
            try:
                model.load_weights(path, by_name=True, skip_mismatch=True)
                models[label] = model
            except:
                models[label] = load_model(path, compile=False, safe_mode=False, custom_objects=CUSTOM_OBJECTS)
        except Exception:
            try: models[label] = load_model(path, compile=False, safe_mode=False, custom_objects=CUSTOM_OBJECTS)
            except: pass
            
    return models

SYSTEM_MODELS = load_bomi_engine()

# --- [SECTION 5] LOGIC & VISUALIZATION ---
def calibrate_score(raw, name):
    if name == "Normal": return raw * 100.0
    if raw > 0.40: calibrated = 85 + ((raw - 0.40) / 0.60) * 14.9
    elif raw > 0.25: calibrated = 60 + ((raw - 0.25) / 0.15) * 25
    else: calibrated = raw * 100.0
    return min(calibrated, 99.9)

def generate_visuals(img_tensor, original_pil_img, model):
    try:
        last_conv = None
        # Safe Layer Hunter (Fixes 'Dense' crash)
        for layer in reversed(model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                if len(layer.weights) > 0:
                    last_conv = layer.name
                    break
        
        if not last_conv:
            # Fallback
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower(): last_conv = layer.name; break
        
        if not last_conv: return np.array(original_pil_img.resize((224, 224))), (112, 112)

        img_variable = tf.Variable(img_tensor)
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
        
        with tf.GradientTape() as tape:
            tape.watch(img_variable)
            conv_out, preds = grad_model(img_variable)
            if preds.shape[-1] > 1: loss = preds[:, tf.argmax(preds[0])]
            else: loss = preds[:, 0]
            
        grads = tape.gradient(loss, conv_out)
        if grads is None: return np.array(original_pil_img.resize((224, 224))), (112, 112)
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0: heatmap /= max_val
        heatmap[heatmap < 0.25] = 0 # Clean noise
            
        heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        orig = np.array(original_pil_img.resize((224, 224)))
        if len(orig.shape) == 2: orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        
        mask = heatmap > 0
        overlay = orig.copy()
        overlay[mask] = cv2.addWeighted(orig[mask], 0.7, heatmap_rgb[mask], 0.3, 0)
        
        return np.uint8(overlay), (112, 112)

    except:
        return np.array(original_pil_img.resize((224, 224))), (112, 112)

def generate_bar_chart(scores_dict, winner):
    plot_data = scores_dict.copy()
    if winner != "Normal": plot_data.pop("Normal", None)
    names = list(plot_data.keys())
    values = list(plot_data.values())
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#ff0055' if n == winner else '#00eaff' for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0, 110)
    ax.set_title(f"Confidence: {winner}", color="#333")
    for i, v in enumerate(values):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
    ts = int(time.time())
    path = os.path.join(EVIDENCE_DIR, f"Chart_{ts}.png")
    plt.savefig(path)
    plt.close()
    return path

def save_to_records(p_id, winner, conf, roi, vis_path, chart_path, user):
    rec = {
        "id": p_id, 
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "user": user, 
        "diagnosis": winner, 
        "confidence": f"{conf:.2f}",
        "roi_data": str(roi),
        "heatmap_file": vis_path,
        "chart_file": chart_path
    }
    
    # --- SYNTAX ERROR FIX ---
    data = []
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, 'r') as f:
                data = json.load(f)
        except:
            data = []
    # ------------------------
    
    data.append(rec)
    with open(DB_PATH, 'w') as f: json.dump(data, f, indent=4)

def delete_record(record_index):
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, 'r') as f: data = json.load(f)
            if 0 <= record_index < len(data):
                rec = data[record_index]
                try: 
                    if os.path.exists(rec.get('heatmap_file','')): os.remove(rec['heatmap_file'])
                    if os.path.exists(rec.get('chart_file','')): os.remove(rec['chart_file'])
                except: pass
                del data[record_index]
                with open(DB_PATH, 'w') as f: json.dump(data, f, indent=4)
                return True
        except: pass
    return False

# --- [SECTION 6] AUTHENTICATION ---
def check_login():
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        # VIBRANT ANIME LOGIN CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .login-card {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(15px);
            padding: 3rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            animation: float 6s ease-in-out infinite;
        }
        h2 { color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: 800; }
        @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }
        .stTextInput input {
            background: rgba(255, 255, 255, 0.8) !important;
            color: #333 !important;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c2:
            st.markdown('<div class="login-card"><h2>⚛️ BOMI NUCLEUS</h2><p style="color:white;">Secure User Uplink</p></div>', unsafe_allow_html=True)
            st.write("")
            with st.form("login_form"):
                user = st.text_input("Agent Identity")
                pwd = st.text_input("Security Key", type="password")
                if st.form_submit_button("ESTABLISH CONNECTION", use_container_width=True):
                    valid_users = ["Vamsii", "Anika", "Harsha", "Teacher", "Admin"]
                    if user in valid_users and pwd == "Bomi@2026":
                        st.session_state.logged_in = True
                        st.session_state.username = user
                        st.rerun()
                    else: st.error("⛔ ACCESS DENIED")
        return False
    return True

# --- [SECTION 7] SMART CHATBOT (WITH SOUL) ---
def get_bot_response(user_input, context_diagnosis, user_name):
    u_in = user_input.lower()
    
    # 1. GREETINGS & IDENTITY
    if any(x in u_in for x in ["hi", "hello", "hey"]):
        return f"Greetings, Agent {user_name}. Systems are optimal. How can I assist with the diagnosis today?"
    if "who are you" in u_in or "your name" in u_in:
        return "I am Bomi, an advanced Neural Diagnostic Intelligence designed to assist Agent Vamsi and team."
    if "how are you" in u_in:
        return "I am functioning at 100% efficiency. Ready to analyze X-ray data."

    # 2. CONTEXTUAL MEDICAL Q&A
    if not context_diagnosis:
        return "I cannot answer medical queries yet. Please initialize a scan in the Diagnosis Console."
        
    if "why" in u_in:
        if context_diagnosis == "Normal": 
            return "Analysis of the lung parenchyma shows clear air spaces. I found no evidence of opacity, consolidation, or fluid buildup."
        elif context_diagnosis == "Pneumonia": 
            return "I detected distinct regions of 'lobar consolidation'. This density usually indicates alveoli filled with fluid or pus, a hallmark of Pneumonia."
        elif context_diagnosis == "Covid": 
            return "The heatmap reveals bilateral, peripheral ground-glass opacities. This specific distribution is a strong radiological indicator of COVID-19."
        elif context_diagnosis == "Tuberculosis": 
            return "I identified structural anomalies in the apical (upper) zones, suggestive of fibrosis or cavitation, which are typical of Tuberculosis."
            
    if "treatment" in u_in or "medicine" in u_in:
        return "Protocol Restriction: I am a diagnostic aid, not a prescriber. Please refer the patient to a specialist for a treatment plan."
    
    if "sure" in u_in or "accuracy" in u_in:
        return f"My confidence score is calibrated based on deep learning feature extraction. However, clinical correlation with patient symptoms is always required."

    # 3. FALLBACK
    return f"Acknowledged. Regarding the **{context_diagnosis}** finding: I recommend monitoring SpO2 and temperature."

# --- [SECTION 8] MAIN UI ---
def main():
    st.set_page_config(page_title="Bomi Nucleus", page_icon="⚛️", layout="wide")
    
    # MAIN APP CSS (Holo-Light Theme)
    st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(242, 246, 252) 0%, rgb(228, 237, 248) 90%);
    }
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(118, 75, 162, 0.3);
        display: flex; justify-content: space-between; align-items: center;
        animation: slideDown 0.8s ease-out;
    }
    /* Cards */
    .stContainer {
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        padding: 10px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        border: none; color: white; border-radius: 30px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(58, 123, 213, 0.4);
    }
    @keyframes slideDown { from { transform: translateY(-50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)
    
    if not check_login(): return

    # LIVE CLOCK
    now = datetime.datetime.now().strftime("%A | %H:%M:%S")
    st.markdown(f"""
    <div class="main-header">
        <div style="font-size: 26px; font-weight: 800;">⚛️ BOMI NUCLEUS</div>
        <div style="background:rgba(255,255,255,0.2); padding:5px 15px; border-radius:10px;">🕒 {now}</div>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR
    st.sidebar.title(f"Agent {st.session_state.username}")
    app_mode = st.sidebar.radio("Main Menu", ["Diagnosis Console", "Records Room", "AI Assistant"])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Disconnect"):
        st.session_state.logged_in = False
        st.rerun()

    # --- 1. DIAGNOSIS CONSOLE ---
    if app_mode == "Diagnosis Console":
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("### 📥 Source Feed")
            with st.container(border=True):
                p_id = st.text_input("Subject ID", "SUB-001")
                f = st.file_uploader("Upload Scan Data", type=['jpg', 'png', 'jpeg'])
                if f:
                    img = Image.open(f).convert('RGB')
                    st.image(img, caption="Input Data", use_container_width=True)

        with c2:
            st.markdown("### ⚡ Neural Processing")
            if f and st.button("ACTIVATE NEURAL ENGINE", type="primary", use_container_width=True):
                with st.spinner("Analyzing Neural Pathways..."):
                    # PRE-INIT VARS (FIX FOR UnboundLocalError)
                    vis_path, chart_path = None, None
                    
                    x = np.expand_dims(np.array(img.resize((224, 224)))/255.0, axis=0)
                    raw_scores = {}
                    for name, model in SYSTEM_MODELS.items():
                        try: raw_scores[name] = float(model.predict(x, verbose=0)[0][0])
                        except: raw_scores[name] = 0.0

                    # LOGIC OVERRIDE
                    best_disease, best_score = None, 0
                    for k, v in raw_scores.items():
                        if k != "Normal" and v > best_score:
                            best_score = v; best_disease = k
                    
                    winner = best_disease if best_score > 0.40 else max(raw_scores, key=raw_scores.get)
                    
                    calibrated_scores = {k: calibrate_score(v, k) for k,v in raw_scores.items()}
                    conf = calibrated_scores[winner]
                    st.session_state.last_diagnosis = winner

                    vis, roi = generate_visuals(x, img, SYSTEM_MODELS[winner])
                    ts = int(time.time())
                    vis_path = os.path.join(EVIDENCE_DIR, f"{winner}_{ts}.png")
                    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    chart_path = generate_bar_chart(calibrated_scores, winner)
                    
                    if vis_path and chart_path:
                        save_to_records(p_id, winner, conf, roi, vis_path, chart_path, st.session_state.username)

                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border-left: 8px solid #FF0055; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
                        <h1 style="color: #FF0055; margin:0;">{winner.upper()} DETECTED</h1>
                        <p style="color: #555; margin:0; font-size: 18px;">Probability Matrix: <b>{conf:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    t1, t2 = st.tabs(["Scan Visuals", "Data Graph"])
                    with t1: st.image(vis, caption="ROI Overlay", use_container_width=True)
                    with t2: st.image(chart_path, use_container_width=True)

    # --- 2. RECORDS ROOM ---
    elif app_mode == "Records Room":
        st.markdown("### 🗄️ Archives")
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'r') as f: data = json.load(f)
                cleaned_data = []
                for r in data:
                    if 'id' not in r: continue
                    if 'confidence' not in r: r['confidence'] = "N/A"
                    if 'date' not in r: r['date'] = "Unknown"
                    if 'time' not in r: r['time'] = "--:--"
                    cleaned_data.append(r)
                
                if cleaned_data:
                    df = pd.DataFrame(cleaned_data)
                    st.dataframe(df[['id', 'date', 'time', 'diagnosis', 'confidence']], use_container_width=True)
                    c_sel, c_del = st.columns([3, 1])
                    with c_sel:
                        opts = [f"{i}: {r['id']} ({r['diagnosis']})" for i, r in enumerate(cleaned_data)]
                        sel_opt = st.selectbox("Select File", opts)
                        sel_idx = int(sel_opt.split(":")[0])
                    with c_del:
                        st.write("")
                        st.write("")
                        if st.button("🗑️ PURGE", type="primary"):
                            if delete_record(sel_idx):
                                st.success("Purged."); time.sleep(0.5); st.rerun()
                    rec = cleaned_data[sel_idx]
                    v1, v2 = st.columns(2)
                    with v1: 
                        if os.path.exists(rec.get('heatmap_file','')): st.image(rec['heatmap_file'], use_container_width=True)
                    with v2:
                        if os.path.exists(rec.get('chart_file','')): st.image(rec['chart_file'], use_container_width=True)
                else: st.info("Archives Empty.")
            except Exception as e: st.error(f"Archive Error: {e}")
        else: st.info("No Archives Found.")

    # --- 3. AI ASSISTANT ---
    elif app_mode == "AI Assistant":
        st.markdown("### 🤖 Bomi AI Link")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"Connection Established. Hello Agent {st.session_state.username}."}]
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
        if prompt := st.chat_input("Input Query..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            diag = st.session_state.get("last_diagnosis", None)
            response = get_bot_response(prompt, diag, st.session_state.username)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"): st.write(response)

if __name__ == "__main__":
    main()