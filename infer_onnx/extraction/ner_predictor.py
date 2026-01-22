import os
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict
from underthesea import sent_tokenize

# Hàm softmax numpy (thay thế torch.softmax)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class VietnameseNERPredictor:
    """ONNX-based Vietnamese NER Predictor (No PyTorch dependency)"""
    
    def __init__(self, model_path: str = "weights/visobert_onnx_quantized", onnx_filename: str = "model.onnx"):
        """Initialize predictor with ONNX model"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(current_dir, model_path)
        else:
            self.model_path = model_path
        
        # 1. Load Tokenizer
        # Tokenizer vẫn dùng của HF (nhẹ, không cần torch)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.MAX_LENGTH = 512
        self.tokenizer.model_max_length = self.MAX_LENGTH
        
        # 2. Load Config thủ công (Quan trọng: Để lấy id2label)
        # Vì không có model.config của PyTorch, ta phải đọc file json
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
            
        # 3. Load ONNX Session
        onnx_file_path = os.path.join(self.model_path, onnx_filename)
        
        # Tối ưu thread cho Celery (Tránh tranh chấp CPU)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        print(f"Loading ONNX model from {onnx_file_path}...")
        self.session = ort.InferenceSession(onnx_file_path, sess_options, providers=["CPUExecutionProvider"])
        
        # Lấy tên input/output dynamic
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def smart_chunk_text(self, item: dict, max_length: int = 510) -> List[dict]:
        """
        (Giữ nguyên logic cũ của bạn - không thay đổi gì)
        """
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
            new_labels = []
            for lbl in original_labels:
                if lbl['start'] >= chunk_start and lbl['end'] <= chunk_end:
                    new_lbl = lbl.copy()
                    new_lbl['start'] = lbl['start'] - chunk_start
                    new_lbl['end'] = lbl['end'] - chunk_start
                    new_labels.append(new_lbl)
            chunks.append({"text": chunk_text, "label": new_labels})

        for sent in sentences:
            sent_start = text.find(sent, search_cursor)
            if sent_start == -1: sent_start = search_cursor
            sent_end = sent_start + len(sent)
            
            # Tokenize để đếm token (Tokenizer HF chạy tốt mà k cần torch)
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

    def fix_bio_tags(self, tags):
        """(Giữ nguyên logic cũ của bạn)"""
        fixed_tags = list(tags)
        for i in range(len(fixed_tags)):
            if fixed_tags[i].startswith('I-'):
                entity_type = fixed_tags[i][2:]
                if i == 0 or (not fixed_tags[i-1].startswith('B-' + entity_type) and 
                              not fixed_tags[i-1].startswith('I-' + entity_type)):
                    fixed_tags[i] = 'B-' + entity_type
        return fixed_tags

    def predict_text(self, text: str) -> Dict:
        """(Giữ nguyên logic cũ - chỉ gọi _predict_chunk mới)"""
        chunks = self.smart_chunk_text({'text': text}, max_length=self.MAX_LENGTH)
        all_entities = []
        current_offset = 0
        
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_entities = self._predict_chunk(chunk_text)
            
            # Logic map lại vị trí offset trong văn bản gốc
            chunk_start = text.find(chunk_text, current_offset)
            if chunk_start != -1:
                for entity in chunk_entities:
                    entity["start"] = int(entity["start"] + chunk_start)
                    entity["end"] = int(entity["end"] + chunk_start)
                    all_entities.append(entity)
                current_offset = chunk_start + len(chunk_text)
        
        entities_by_type = {}
        for entity in all_entities:
            label = entity["label"]
            if label not in entities_by_type: entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        return {
            "text": text,
            "entities": all_entities,
            "entities_by_type": entities_by_type,
            "total_entities": len(all_entities)
        }

    def _predict_chunk(self, text: str) -> List[Dict]:
        """
        Hàm inference chính đã được viết lại cho ONNX + Numpy
        """
        # 1. Tokenize (return numpy)
        inputs = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="np" # Quan trọng: Trả về Numpy array
        )
        
        # Tách offset_mapping ra xử lý riêng
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        # 2. Chuẩn bị input cho ONNX (cast sang int64)
        ort_inputs = {
            k: v.astype(np.int64) 
            for k, v in inputs.items() 
            if k in self.input_names
        }
        
        # 3. Chạy Inference (Thay cho model(**inputs))
        logits = self.session.run([self.output_name], ort_inputs)[0]
        
        # 4. Post-process bằng Numpy
        # Softmax trên trục cuối cùng (dim=-1)
        probs = softmax(logits)
        
        # Argmax để lấy class ID
        predictions = np.argmax(logits, axis=-1)[0]
        
        # Max để lấy confidence score
        confidences = np.max(probs, axis=-1)[0]
        
        # 5. Decode Labels (Logic y hệt cũ nhưng dùng biến python thường)
        raw_labels = []
        valid_indices = []
        
        # zip các mảng numpy
        for i, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            if start == end:  # Bỏ qua special tokens
                continue
            
            # Map ID sang Label text
            label = self.id2label.get(int(pred_id), "O")
            raw_labels.append(label)
            valid_indices.append((i, conf, start, end))
        
        # Fix BIO tag sequences
        fixed_labels = self.fix_bio_tags(raw_labels)
        
        # 6. Gom tokens thành Entities (Giữ nguyên logic cũ)
        entities = []
        current_entity = None
        
        for (i, conf, start, end), label in zip(valid_indices, fixed_labels):
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            entity_type = label[2:]  # Remove B- or I-
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": int(start),
                    "end": int(end),
                    "confidence": float(conf)
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == entity_type:
                current_entity["text"] += text[start:end]
                current_entity["end"] = int(end)
                # Lấy confidence cao nhất trong chuỗi token
                current_entity["confidence"] = max(current_entity["confidence"], float(conf))
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def format_results(self, results: Dict) -> str:
        """(Giữ nguyên logic cũ)"""
        output = [f"📝 Text: {results['text'][:100]}..."]
        if results['entities']:
            for i, entity in enumerate(results['entities'], 1):
                output.append(f"   {i}. '{entity['text']}' -> {entity['label']} ({entity['confidence']:.2f})")
        else:
            output.append("❌ No entities detected")
        return "\n".join(output)