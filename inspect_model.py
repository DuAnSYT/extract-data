import gradio as gr
import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
from underthesea import sent_tokenize
import numpy as np

# ==========================================
# PHẦN 1: CLASS INFERENCE CỦA BẠN (GIỮ NGUYÊN LOGIC)
# ==========================================

class VietnameseNERPredictor:
    """Simple Vietnamese NER Predictor"""
    
    def __init__(self, model_name_or_path: str):
        """Initialize predictor with model"""
        print(f"Loading model from: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.eval()

        self._TOKENIZER = self.tokenizer
        self.MAX_LENGTH = 512
        self.tokenizer.model_max_length = self.MAX_LENGTH
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU")

    def fix_bio_tags(self, tags):
        """Fix invalid BIO tag sequences"""
        fixed_tags = list(tags)
        for i in range(len(fixed_tags)):
            if fixed_tags[i].startswith('I-'):
                entity_type = fixed_tags[i][2:]
                if i == 0 or (not fixed_tags[i-1].startswith('B-' + entity_type) and 
                            not fixed_tags[i-1].startswith('I-' + entity_type)):
                    fixed_tags[i] = 'B-' + entity_type
        return fixed_tags

    def smart_chunk_text(self, item: dict, max_length: int = 510) -> List[dict]:
        text = item['text']
        original_labels = item.get('label', [])
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0 
        search_cursor = 0
        
        def flush_chunk(sent_buffer):
            if not sent_buffer: return
            chunk_start = sent_buffer[0][0]
            chunk_end = sent_buffer[-1][1]
            chunk_text = text[chunk_start:chunk_end]
            new_labels = [] # Logic label gốc của bạn (nếu có)
            chunks.append({"text": chunk_text, "label": new_labels})

        for sent in sentences:
            sent_start = text.find(sent, search_cursor)
            if sent_start == -1: sent_start = search_cursor
            sent_end = sent_start + len(sent)
            sent_token_ids = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_token_count = len(sent_token_ids)
            
            if current_token_count + sent_token_count <= max_length:
                current_chunk_sentences.append((sent_start, sent_end, sent))
                current_token_count += sent_token_count
            else:
                flush_chunk(current_chunk_sentences)
                current_chunk_sentences = [(sent_start, sent_end, sent)]
                current_token_count = sent_token_count
            search_cursor = sent_end

        if current_chunk_sentences:
            flush_chunk(current_chunk_sentences)
        return chunks

    def _predict_chunk(self, text: str) -> List[Dict]:
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.MAX_LENGTH, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = inputs.pop("offset_mapping")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
        
        predictions = predictions.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        offset_mapping = offset_mapping.cpu().numpy()[0]
        
        raw_labels = []
        valid_indices = []
        
        for i, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            if start == end: continue 
            label = self.model.config.id2label[int(pred_id)]
            raw_labels.append(label)
            valid_indices.append((i, conf, start, end))
        
        fixed_labels = self.fix_bio_tags(raw_labels)
        entities = []
        current_entity = None
        
        for (i, conf, start, end), label in zip(valid_indices, fixed_labels):
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            entity_type = label[2:]
            if label.startswith("B-"):
                if current_entity: entities.append(current_entity)
                current_entity = {
                    "text": text[start:end], "label": entity_type,
                    "start": int(start), "end": int(end), "confidence": float(conf)
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == entity_type:
                current_entity["text"] += text[start:end]
                current_entity["end"] = int(end)
                current_entity["confidence"] = max(current_entity["confidence"], float(conf))
        
        if current_entity: entities.append(current_entity)
        return entities

    def predict_text(self, text: str) -> Dict:
        chunks = self.smart_chunk_text({'text': text}, max_length=self.MAX_LENGTH)
        all_entities = []
        current_offset = 0
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_entities = self._predict_chunk(chunk_text)
            
            chunk_start = text.find(chunk_text, current_offset)
            if chunk_start != -1:
                for entity in chunk_entities:
                    entity["start"] += chunk_start
                    entity["end"] += chunk_start
                    all_entities.append(entity)
                current_offset = chunk_start + len(chunk_text)
        
        return {"text": text, "entities": all_entities}

# ==========================================
# PHẦN 2: GRADIO VISUALIZATION HELPERS
# ==========================================

def generate_token_html(tokens, labels, confidences):
    """
    Hàm helper tạo HTML (Đã thêm màu cho ADDR)
    """
    html_parts = []
    
    # === CẬP NHẬT BẢNG MÀU TẠI ĐÂY ===
    colors = {
        "PER": "#ffdad9",   # Đỏ nhạt
        "ORG": "#d7e3ff",   # Xanh dương nhạt
        "ADDR": "#e9dff7",  # Tím nhạt (MỚI THÊM)
        "O": "#f5f5f5"      # Xám (Không phải entity)
    }
    
    # Màu viền tương ứng (đậm hơn nền chút)
    border_colors = {
        "PER": "#ffb4ab",
        "ORG": "#abc7ff",
        "ADDR": "#d0bcff",  # Viền Tím (MỚI THÊM)
        "O": "#cccccc"
    }

    for token, label, conf in zip(tokens, labels, confidences):
        if token in ["<s>", "</s>", "<pad>"]:
            continue
            
        # Tách lấy phần type (ví dụ B-ADDR -> ADDR)
        entity_type = label.split("-")[-1] if "-" in label else label
        
        # Lấy màu, nếu không có trong list thì lấy màu của "O"
        bg_color = colors.get(entity_type, colors["O"])
        bd_color = border_colors.get(entity_type, border_colors["O"])
        
        tag_style = "font-size: 0.65em; opacity: 0.7; font-weight: bold; display: block; margin-top: 2px;"
        token_style = "font-family: 'Consolas', 'Monaco', monospace; font-size: 1.1em; font-weight: 600;"
        
        # Box style (Vẫn giữ color: #333333 để chống mù màu trên Darkmode)
        box_style = (
            f"display: inline-block; text-align: center; margin: 3px; padding: 4px 8px; "
            f"background-color: {bg_color}; border: 1px solid {bd_color}; border-radius: 6px; "
            f"line-height: 1.2; vertical-align: top; position: relative; "
            f"min-width: 20px; color: #333333;" 
        )
        
        tooltip_attr = f'title="Tag: {label} | Conf: {conf:.4f}"'

        html_parts.append(
            f'<div style="{box_style}" {tooltip_attr}>'
            f'<span style="{token_style}">{token}</span>'
            f'<span style="{tag_style}">{label}</span>'
            f'</div>'
        )
        
    return "".join(html_parts)

def visualize_token_map(text):
    """
    Hiển thị Full Text dưới dạng Token Blocks giống Tokenizer Playground
    nhưng có thêm màu Entity và Tag dự đoán.
    """
    if not text.strip():
        return ""

    # 1. Chunking text để xử lý văn bản dài
    chunks = predictor.smart_chunk_text({'text': text}, max_length=predictor.MAX_LENGTH)
    
    full_html = ['<div style="font-family: sans-serif; padding: 10px; line-height: 2.5;">']
    
    for chunk in chunks:
        chunk_text = chunk["text"]
        
        # 2. Inference từng chunk
        inputs = predictor.tokenizer(chunk_text, truncation=True, max_length=512, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = predictor.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            confidences = torch.softmax(outputs.logits, dim=2).max(dim=2)[0][0].cpu().numpy()
            
        # 3. Lấy tokens và mapping nhãn
        tokens = predictor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        id2label = predictor.model.config.id2label
        
        labels = []
        for p in predictions:
            l = str(id2label[p]) if id2label else str(p)
            labels.append("O" if l == "0" or l == "O" else l)
            
        # 4. Generate HTML cho chunk này
        full_html.append(generate_token_html(tokens, labels, confidences))
        
        # Thêm dấu ngắt dòng visual giữa các chunk (nếu muốn)
        # full_html.append('<div style="width: 100%; height: 10px;"></div>')
        
    full_html.append('</div>')
    return "".join(full_html)

# Khởi tạo predictor toàn cục
# LƯU Ý: Đổi đường dẫn model ở đây nếu chạy local
MODEL_NAME = "visobert_model/final" 
# Nếu bạn chưa train NER, model này sẽ load weight random cho head classification
# Nếu bạn có folder model đã train, thay path vào đây: e.g., "visobert_model/final"
try:
    predictor = VietnameseNERPredictor(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Vui lòng kiểm tra lại đường dẫn model.")

def inspect_tokens_detailed(text):
    """
    Hàm debug Tokenizer có hỗ trợ Chunking.
    Nó sẽ chạy y hệt logic cắt đoạn của model chính để bạn kiểm tra xem
    câu bị cắt ở đâu, có bị mất ngữ nghĩa không.
    """
    # 1. Sử dụng chính logic chunking của class predictor
    chunks = predictor.smart_chunk_text({'text': text}, max_length=predictor.MAX_LENGTH)
    
    all_rows = []
    
    # 2. Lặp qua từng chunk để inspect
    for chunk_idx, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        
        # Tokenize lại từng chunk (để lấy input_ids và attention_mask cho model)
        inputs = predictor.tokenizer(chunk_text, truncation=True, max_length=512, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Predict từng chunk
        with torch.no_grad():
            outputs = predictor.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
            confidences = torch.softmax(logits, dim=2).max(dim=2)[0][0].cpu().numpy()

        # Convert IDs to Tokens
        tokens = predictor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        id2label = predictor.model.config.id2label

        # 3. Lưu thông tin từng token vào list
        for i, (token, pred_id, conf) in enumerate(zip(tokens, predictions, confidences)):
            # Fix lỗi label 0 vs O
            raw_label = str(id2label[pred_id]) if id2label else str(pred_id)
            label = "O" if raw_label == "0" or raw_label == "O" else raw_label
            
            # Highlight logic
            is_entity = label != "O" and token not in ["<s>", "</s>", "<pad>"]
            
            all_rows.append({
                "Chunk #": chunk_idx + 1,      # Để biết token thuộc đoạn cắt thứ mấy
                "Index": i,                    # Vị trí trong chunk
                "Token": token,
                "Token ID": inputs["input_ids"][0][i].item(),
                "Predicted Tag": label,
                "Confidence": round(float(conf), 4),
                "Is Entity": "✅" if is_entity else ""
            })
            
    # 4. Trả về DataFrame gộp
    return pd.DataFrame(all_rows)

def visualize_ner(text):
    """
    Phiên bản Fix lỗi hiển thị Gradio:
    1. Ép kiểu mạnh mẽ (str, int) để tránh numpy types.
    2. In ra format cuối cùng để debug.
    """
    if not text.strip():
        return []

    # 1. Lấy kết quả từ model
    result = predictor.predict_text(text)
    entities = result['entities']
    
    # Sort entity theo vị trí
    entities.sort(key=lambda x: x['start'])

    formatted_output = []
    cursor = 0
    
    # 2. Loop để cắt ghép chuỗi
    for ent in entities:
        # Ép kiểu int tuyệt đối cho start/end
        start = int(ent['start'])
        end = int(ent['end'])
        
        # Ép kiểu string tuyệt đối cho label
        label = str(ent['label'])
        
        # Cắt text (đảm bảo index nằm trong giới hạn)
        start = max(0, start)
        end = min(len(text), end)
        
        # Phần Text thường (Label là None)
        if start > cursor:
            sub_text = str(text[cursor:start]) # Ép kiểu str
            if sub_text:
                formatted_output.append((sub_text, None))
        
        # Phần Entity (Label có giá trị)
        ent_text = str(text[start:end]) # Ép kiểu str
        if ent_text:
            formatted_output.append((ent_text, label))
            
        cursor = end
        
    # Phần Text còn lại sau entity cuối
    if cursor < len(text):
        remainder = str(text[cursor:])
        formatted_output.append((remainder, None))

    
    return formatted_output
# ==========================================
# PHẦN 3: GRADIO UI
# ==========================================

with gr.Blocks(title="Vietnamese NER Inspector") as demo:
    gr.Markdown("# 🕵️ Vietnamese NER Inspector (ViSoBERT)")
    gr.Markdown("Tool này giúp visualize kết quả NER và inspect chi tiết cách model tokenize dữ liệu.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text", 
            lines=5, 
            placeholder="Nhập văn bản tiếng Việt vào đây...",
            value="Bác sĩ Nguyễn Văn A làm việc tại Bệnh viện Chợ Rẫy TP.HCM."
        )
    
    with gr.Row():
        btn_submit = gr.Button("🚀 Analyze", variant="primary")

    with gr.Tabs():
        with gr.TabItem("🎨 Visual Entities"):
            gr.Markdown("*Kết quả sau khi đã ghép các tokens (Logic của `predict_text`)*")
            output_highlight = gr.HighlightedText(
                label="Named Entities",
                combine_adjacent=True,
                show_legend=True,
            )
            
        with gr.TabItem("🔬 Token Inspector"):
            gr.Markdown("""
            *Bảng này hiển thị cách **Tokenizer** cắt từ và **Raw Tag** mà model dự đoán cho từng token.*
            - Quan sát cột `Token` để xem sub-word (ví dụ `_Hồ`, `_Chí`).
            - Quan sát cột `Predicted Tag` để xem B-TAG, I-TAG.
            """)
            output_df = gr.Dataframe(
                # Thêm cột "Chunk #" vào đầu danh sách headers
                headers=["Chunk #", "Index", "Token", "Token ID", "Predicted Tag", "Confidence", "Is Entity"],
                interactive=False
            )

        with gr.TabItem("🧩 Token Map & NER"):
            gr.Markdown("""
            *Giao diện này hiển thị từng token trong một hộp riêng biệt.*
            - **Màu nền**: Loại thực thể (Đỏ=PER, Xanh=ORG...).
            - **Dòng trên**: Token (có dấu `_` nếu là đầu từ).
            - **Dòng dưới**: Tag dự đoán (B-..., I-...).
            - *Rê chuột vào hộp để xem độ tin cậy (Confidence score).*
            """)
            # Dùng gr.HTML để render
            output_token_map = gr.HTML(label="Token Map Visualization")


    # Sự kiện click
    btn_submit.click(
        fn=lambda x: (visualize_ner(x), inspect_tokens_detailed(x), visualize_token_map(x)),
        inputs=[input_text],
        outputs=[output_highlight, output_df, output_token_map]
    )

if __name__ == "__main__":
    demo.launch()