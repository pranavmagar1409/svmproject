import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (svm.pkl must sit next to this file)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

# ---------------------------------------------------------------------------
# Label encodings used when the model was trained.
# The model only stores numbers, so EDIT THE LABELS below if your training
# encoding differs (keep the numeric codes, change the text).
# ---------------------------------------------------------------------------
GENDER = {0: "Female", 1: "Male"}
REGION = {0: "Rural", 1: "Urban"}
OCCUPATION = {0: "Business", 1: "Employed", 2: "Student", 3: "Self-employed"}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVM Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',sans-serif;min-height:100vh;display:flex;
    align-items:center;justify-content:center;padding:24px;color:#fff;
    background:linear-gradient(-45deg,#1e1b4b,#4c1d95,#0e7490,#be185d);
    background-size:400% 400%;animation:bg 16s ease infinite;overflow-x:hidden;
  }
  @keyframes bg{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
  .blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.45;z-index:0;animation:float 12s ease-in-out infinite}
  .b1{width:320px;height:320px;background:#22d3ee;top:-80px;left:-80px}
  .b2{width:380px;height:380px;background:#f472b6;bottom:-120px;right:-100px;animation-delay:-5s}
  .b3{width:220px;height:220px;background:#a78bfa;top:50%;left:60%;animation-delay:-8s}
  @keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,-50px) scale(1.15)}}

  .card{
    position:relative;z-index:1;width:100%;max-width:480px;padding:38px 34px;
    background:rgba(255,255,255,.12);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
    border:1px solid rgba(255,255,255,.28);border-radius:26px;
    box-shadow:0 25px 60px rgba(0,0,0,.35);animation:rise .9s cubic-bezier(.2,.8,.2,1);
  }
  @keyframes rise{from{opacity:0;transform:translateY(40px) scale(.96)}to{opacity:1;transform:none}}
  .icon{font-size:42px;text-align:center;animation:bob 3s ease-in-out infinite}
  @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  h1{text-align:center;font-weight:700;font-size:26px;margin:6px 0 4px;
     background:linear-gradient(90deg,#67e8f9,#f9a8d4);-webkit-background-clip:text;background-clip:text;color:transparent}
  .sub{text-align:center;font-weight:300;font-size:13px;opacity:.85;margin-bottom:26px}

  .field{margin-bottom:16px}
  label{display:block;font-size:12px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;opacity:.9}
  input,select{
    width:100%;padding:13px 15px;border-radius:13px;border:1px solid rgba(255,255,255,.3);
    background:rgba(255,255,255,.14);color:#fff;font:inherit;font-size:15px;outline:none;transition:.25s;
  }
  input::placeholder{color:rgba(255,255,255,.55)}
  select option{color:#111}
  input:focus,select:focus{border-color:#67e8f9;background:rgba(255,255,255,.22);box-shadow:0 0 0 4px rgba(103,232,249,.25);transform:translateY(-1px)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}

  button{
    width:100%;margin-top:8px;padding:15px;border:none;border-radius:14px;cursor:pointer;
    font:inherit;font-weight:600;font-size:16px;color:#fff;position:relative;overflow:hidden;
    background:linear-gradient(90deg,#06b6d4,#8b5cf6,#ec4899);background-size:200% 100%;
    transition:.35s;box-shadow:0 10px 25px rgba(139,92,246,.45);
  }
  button:hover{background-position:100% 0;transform:translateY(-3px);box-shadow:0 16px 32px rgba(236,72,153,.5)}
  button:active{transform:scale(.98)}
  button.loading::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent);animation:shine 1s infinite}
  @keyframes shine{from{transform:translateX(-100%)}to{transform:translateX(100%)}}

  #result{margin-top:22px;text-align:center;padding:20px;border-radius:18px;display:none;font-weight:600;font-size:20px}
  #result.show{display:block;animation:pop .6s cubic-bezier(.2,1.4,.4,1)}
  #result.yes{background:rgba(16,185,129,.28);border:1px solid #34d399;box-shadow:0 0 30px rgba(52,211,153,.5)}
  #result.no{background:rgba(244,63,94,.26);border:1px solid #fb7185;box-shadow:0 0 30px rgba(251,113,133,.5)}
  #result.err{background:rgba(234,179,8,.25);border:1px solid #facc15}
  #result small{display:block;font-weight:300;font-size:13px;margin-top:4px;opacity:.9}
  @keyframes pop{from{opacity:0;transform:scale(.6)}to{opacity:1;transform:scale(1)}}
  footer{text-align:center;margin-top:18px;font-size:11px;opacity:.6}
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

<div class="card">
  <div class="icon">🤖</div>
  <h1>SVM Predictor</h1>
  <p class="sub">Fill in the details and let the model decide</p>

  <form id="form">
    <div class="row">
      <div class="field"><label>Age</label>
        <input type="number" name="Age" min="1" max="120" placeholder="e.g. 28" required></div>
      <div class="field"><label>Income</label>
        <input type="number" name="Income" min="0" step="500" placeholder="e.g. 20000" required></div>
    </div>
    <div class="row">
      <div class="field"><label>Gender</label>
        <select name="Gender" required>
          {% for k, v in gender.items() %}<option value="{{k}}">{{v}}</option>{% endfor %}
        </select></div>
      <div class="field"><label>Region</label>
        <select name="Region" required>
          {% for k, v in region.items() %}<option value="{{k}}">{{v}}</option>{% endfor %}
        </select></div>
    </div>
    <div class="field"><label>Occupation</label>
      <select name="Occupation" required>
        {% for k, v in occupation.items() %}<option value="{{k}}">{{v}}</option>{% endfor %}
      </select></div>
    <button type="submit" id="btn">✨ Predict</button>
  </form>

  <div id="result"></div>
  <footer>Powered by scikit-learn SVM · Flask · Render</footer>
</div>

<script>
const form = document.getElementById('form');
const btn = document.getElementById('btn');
const box = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.classList.add('loading'); box.className = '';
  try {
    const res = await fetch('/predict', {method:'POST', body:new FormData(form)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong');
    const yes = data.prediction === 'yes';
    box.className = 'show ' + (yes ? 'yes' : 'no');
    box.innerHTML = (yes ? '✅ Prediction: YES' : '❌ Prediction: NO') +
                    '<small>' + (yes ? 'The model says this is a positive case.' : 'The model says this is a negative case.') + '</small>';
  } catch (err) {
    box.className = 'show err'; box.textContent = '⚠️ ' + err.message;
  } finally { btn.classList.remove('loading'); }
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        PAGE, gender=GENDER, region=REGION, occupation=OCCUPATION
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {name: float(request.form[name]) for name in FEATURES}
        df = pd.DataFrame([row], columns=FEATURES)
        pred = model.predict(df)[0]
        return jsonify({"prediction": str(pred)})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError:
        return jsonify({"error": "Please enter valid numbers."}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
