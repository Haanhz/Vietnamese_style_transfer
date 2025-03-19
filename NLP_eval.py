from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import torch

model_dir = r"D:\Ha Anh\Group16\Sourcecode\model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")


def preprocess_informal(text, min_length=6):
    # Xử lý khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+([?.!,])', r'\1', text)  # Xóa khoảng trắng dư trước dấu câu
    text = re.sub(r'([?.!])\1{2,}', r'\1', text)  # Giảm thiểu dấu câu lặp

    # Thay thế từ viết tắt tạm thời để không bị tách nhầm
    text = re.sub(r'\b(mr|dr|ms|mrs)\.', r'\1<ABBR>', text, flags=re.IGNORECASE)

    # Tách câu dựa trên dấu câu kết thúc
    sentences = re.split(r'(?<=\.|\?|\!)\s', text)

    # Phục hồi từ viết tắt
    sentences = [re.sub(r'<ABBR>', '.', s, flags=re.IGNORECASE) for s in sentences]

    # Chỉ giữ câu dài hơn min_length
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]

    return sentences

def informal_to_formal(text):
    # Tách câu và thêm <EOS> đã được xử lý trong preprocess_informal
    sentences = preprocess_informal(text)  # Output đã có <EOS> cho mỗi câu
    formal_sentences = []

    for sentence in sentences:
        # Chuyển câu sang định dạng tensor cho model
        inputs = tokenizer(
            sentence, return_tensors="pt", max_length=128, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate output từ model cho từng câu
        outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)

        # Chuyển output từ model về dạng văn bản và thêm vào danh sách formal_sentences
        formal_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        formal_sentences.append(formal_sentence)

    # Nối các câu formal lại thành một văn bản hoàn chỉnh và trả về
    return ' '.join(formal_sentences)

def predict(text):
    model.eval()
    return informal_to_formal(text)


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data["text"]
    output_text = predict(text)
    return jsonify({"output": output_text})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
