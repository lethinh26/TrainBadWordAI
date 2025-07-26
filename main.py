from sentence_transformers import SentenceTransformer, util
import pickle
import os
import json
from rapidfuzz import fuzz

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = "vio_words.pkl"
NON_DATA_FILE = "non_vio_words.pkl"
TXT_FILE = "luat_data.txt"
TOXIC_JSON = "toxic_abb.json"
SIMILARITY_THRESHOLD = 0.60
FUZZY_MATCH_THRESHOLD = 80 

print("Đang tải mô hình AI...")
model = SentenceTransformer(MODEL_NAME)

# Load toxic abb
try:
    with open(TOXIC_JSON, "r", encoding="utf-8") as f:
        TOXIC_ABBREVIATIONS = json.load(f)
    print(f"✅ Đã nạp {len(TOXIC_ABBREVIATIONS)} từ viết tắt từ {TOXIC_JSON}")
except Exception as e:
    print(f"❌ Lỗi đọc {TOXIC_JSON}: {e}")
    TOXIC_ABBREVIATIONS = {}

def check_abb(text):
    words = text.split()
    for word in words:
        for abbr in TOXIC_ABBREVIATIONS.keys():
            if fuzz.ratio(word.lower(), abbr.lower()) >= FUZZY_MATCH_THRESHOLD:
                return True, word, abbr
    return False, None, None

# Expand word
def expand_abb(text):
    words = text.split()
    for i, word in enumerate(words):
        for abbr, full in TOXIC_ABBREVIATIONS.items():
            similarity = fuzz.ratio(word.lower(), abbr.lower())
            if similarity >= FUZZY_MATCH_THRESHOLD:
                print(f"🔍 Nhận dạng '{word}' ~ '{abbr}' ({similarity}%) → thay bằng '{full}'")
                words[i] = full
    return " ".join(words)

def load_vio_from_txt(file_path):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if " - " in line:
                    text, level = line.strip().split(" - ")
                    data.append((text.strip(), level.strip()))
    except Exception as e:
        print(f"❌ Lỗi đọc file TXT: {e}")
    return data

def get_embed(data):
    texts = [x[0] for x in data]
    labels = [x[1] for x in data]
    embs = model.encode(texts, convert_to_tensor=True)
    return texts, labels, embs

default_rule = [
    ("Spam tin nhắn gây phiền hà", "Mức 3"),
    ("Chat nội dung 18+", "Mức 4"),
    ("Chửi tục, dùng những từ toxic", "Mức 4"),
    ("Dùng những từ không phù hợp với tiêu chuẩn cộng đồng", "Mức 4"),
    ("Xúc phạm người khác", "Mức 4"),
    ("Kêu gọi, sát sinh", "Mức 5"),
    ("Quảng cáo server khác (nội dung có chứa bạn có muốn vào server discord mình không, /discord.gg/...)- Mức 5", "Mức 5"),
]

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        training_data = pickle.load(f)
elif os.path.exists(TXT_FILE):
    training_data = load_vio_from_txt(TXT_FILE)
    if not training_data:
        print("⚠️ File TXT rỗng hoặc lỗi, dùng mẫu mặc định.")
        training_data = default_rule
    with open(DATA_FILE, "wb") as f:
        pickle.dump(training_data, f)
else:
    print("⚠️ Không có file TXT hoặc pkl, dùng mẫu mặc định.")
    training_data = default_rule
    with open(DATA_FILE, "wb") as f:
        pickle.dump(training_data, f)

texts, labels, embeddings = get_embed(training_data)

def detectVio(user_input):
    input_emb = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = float(scores[best_idx])
    print(f"🔍 Điểm tương đồng: {best_score:.2f} (ngưỡng: {SIMILARITY_THRESHOLD})")
    if best_score < SIMILARITY_THRESHOLD:
        return None
    return {
        "input": user_input,
        "match": texts[best_idx],
        "level": labels[best_idx],
        "score": best_score
    }

def learnVio(user_input, correct_level):
    training_data.append((user_input, correct_level))
    with open(DATA_FILE, "wb") as f:
        pickle.dump(training_data, f)
    global texts, labels, embeddings
    texts, labels, embeddings = get_embed(training_data)
    print("✅ Đã học thêm dữ liệu vi phạm.")

def learnNonVio(user_input):
    try:
        with open(NON_DATA_FILE, "rb") as f:
            non_violations = pickle.load(f)
    except FileNotFoundError:
        non_violations = []

    if user_input not in non_violations:
        non_violations.append(user_input)
        with open(NON_DATA_FILE, "wb") as f:
            pickle.dump(non_violations, f)
        print("✅ Bot đã học đây là câu hợp lệ.")
    else:
        print("ℹ️ Câu này đã được đánh dấu là hợp lệ trước đó.")

print("\n🤖 Chatbot kiểm tra vi phạm đã sẵn sàng.")
print("Gõ 'exit' để thoát.\n")

while True:
    user_msg_raw = input("👤 Người dùng: ").strip()
    if user_msg_raw.lower() == "exit":
        print("👋 Tạm biệt!")
        break
    if not user_msg_raw:
        continue

    is_abbr, word_found, matched_abbr = check_abb(user_msg_raw)
    if is_abbr:
        print(f"⚠️ Phát hiện viết tắt/toxic: '{word_found}' ~ '{matched_abbr}' -> mở rộng trước khi kiểm tra AI.")

    user_msg = expand_abb(user_msg_raw)

    try:
        with open(NON_DATA_FILE, "rb") as f:
            non_violations = pickle.load(f)
        if user_msg_raw in non_violations:
            print("✅ Câu này đã được đánh dấu là không vi phạm. Bỏ qua.")
            continue
    except FileNotFoundError:
        pass

    result = detectVio(user_msg)

    if result is None:
        print("✅ Không phát hiện vi phạm. Câu hợp lệ.")
        confirm = input("❓ Theo bạn, câu này có vi phạm không? (y/n): ").lower().strip()
        if confirm == "y":
            level = input("👉 Nhập mức đúng (1–5): ").strip()
            if level.isdigit() and 1 <= int(level) <= 5:
                learnVio(user_msg, f"Mức {level}")
            else:
                print("❌ Vui lòng nhập mức hợp lệ (1–5).")
        else:
            learnNonVio(user_msg)
    else:
        print(f"\n⚠️  Vi phạm nghi ngờ: {result['match']}")
        print(f"🧺  Mức cảnh cáo: {result['level']} (tương đồng: {result['score']:.2f})")

        feedback = input("🤔 Cảnh báo đúng không? (y/n): ").lower().strip()
        if feedback == "n":
            true_level = input("👉 Nhập mức đúng (1–5) hoặc 0 nếu KHÔNG vi phạm: ").strip()
            if true_level.isdigit():
                level_int = int(true_level)
                if level_int == 0:
                    learnNonVio(user_msg)
                elif 1 <= level_int <= 5:
                    learnVio(user_msg, f"Mức {level_int}")
                else:
                    print("❌ Mức không hợp lệ.")
            else:
                print("❌ Vui lòng nhập số.")
    print("\n" + "-"*50 + "\n")
