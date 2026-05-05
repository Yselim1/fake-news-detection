# ============================================================
# CELL 1 — Install sentencepiece (required by xlm-roberta tokenizer)
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"])

# ============================================================
# CELL 2 — Load fine-tuned model from Drive
# ============================================================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "/content/drive/MyDrive/LLM-CV/results/finetuned_model"
# xlm-roberta tokenizer files may not be saved in the checkpoint;
# fall back to the base model name which always has them.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).cuda()
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    fake_p, real_p = probs[0].item(), probs[1].item()
    label = "REAL" if real_p > fake_p else "FAKE"
    return label, fake_p, real_p


# ============================================================
# CELL 2 — Headlines for demonstration
# ============================================================

# Format: (text, expected_label, language)
HEADLINES = [
    # --- Turkish REAL (credible, neutral tone) ---
    ("Merkez Bankası politika faizini yüzde 45'te sabit tuttu",             "REAL", "TR"),
    ("İstanbul'da metro hattı genişletme projesi ihaleye çıktı",            "REAL", "TR"),
    ("Türkiye-AB gümrük birliği müzakerelerinde yeni tur başladı",          "REAL", "TR"),
    ("Sağlık Bakanlığı grip aşısı kampanyasını başlattı",                   "REAL", "TR"),
    ("Borsa İstanbul haftayı yüzde 2,3 yükselişle kapattı",                "REAL", "TR"),

    # --- Turkish FAKE / Satirical ---
    ("Bilim insanları çayın aslında bir tür yakıt olduğunu kanıtladı",      "FAKE", "TR"),
    ("Belediye başkanı tüm trafik ışıklarını yeşile sabitledi, tıkanıklık bitti", "FAKE", "TR"),
    ("Uzmanlar: Türk kahvesi içmek zekayı yüzde 300 artırıyor",            "FAKE", "TR"),
    ("Hükümet duyurdu: Artık herkes ücretsiz tatil yapabilecek",            "FAKE", "TR"),
    ("Kediler aslında oy kullanabiliyormuş, seçim kurulu araştırıyor",      "FAKE", "TR"),

    # --- English REAL (credible, neutral tone) ---
    ("Federal Reserve holds interest rates steady amid inflation concerns",  "REAL", "EN"),
    ("WHO launches global initiative to combat antimicrobial resistance",    "REAL", "EN"),
    ("European Parliament approves new AI regulation framework",             "REAL", "EN"),
    ("NASA confirms water ice deposits near lunar south pole",               "REAL", "EN"),
    ("G7 leaders agree on framework for global minimum corporate tax",       "REAL", "EN"),

    # --- English FAKE / Satirical ---
    ("Scientists prove the moon is made of compressed spreadsheets",         "FAKE", "EN"),
    ("Man reads terms and conditions, gains unlimited power",                "FAKE", "EN"),
    ("Government to replace all currency with pizza slices by 2027",        "FAKE", "EN"),
    ("Area CEO solves climate change after watching 10-minute YouTube video","FAKE", "EN"),
    ("Study finds 100% of people who drink water eventually die",            "FAKE", "EN"),
]


# ============================================================
# CELL 3 — Run predictions and print table
# ============================================================

def run_demo():
    COL_W = 68
    header = f"{'#':>3}  {'Lang':<4}  {'Expected':<8}  {'Predicted':<9}  {'Fake%':>6}  {'Real%':>6}  {'OK':>2}  Headline"
    sep = "-" * (len(header) + 20)
    print(header)
    print(sep)

    results = []
    for i, (text, expected, lang) in enumerate(HEADLINES, 1):
        label, fp, rp = predict(text)
        ok = "✓" if label == expected else "✗"
        results.append((lang, expected, label))
        snippet = text[:COL_W] + ("…" if len(text) > COL_W else "")
        print(f"{i:>3}  {lang:<4}  {expected:<8}  {label:<9}  {fp:>5.1%}  {rp:>5.1%}  {ok:>2}  {snippet}")

    total = len(results)
    correct = sum(1 for _, e, p in results if e == p)
    tr_correct = sum(1 for l, e, p in results if l == "TR" and e == p)
    tr_total   = sum(1 for l, _, _ in results if l == "TR")
    en_correct = sum(1 for l, e, p in results if l == "EN" and e == p)
    en_total   = sum(1 for l, _, _ in results if l == "EN")
    print(sep)
    print(f"\nAccuracy on demo set: {correct}/{total} = {correct/total:.1%}")
    print(f"  Turkish : {tr_correct}/{tr_total}  |  English : {en_correct}/{en_total}")


run_demo()
