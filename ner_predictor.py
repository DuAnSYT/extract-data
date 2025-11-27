#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Vietnamese NER Prediction
Chỉ có function cơ bản để dự đoán NER từ model đã train
"""

import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
from underthesea import sent_tokenize


class VietnameseNERPredictor:
    """Simple Vietnamese NER Predictor"""
    
    def __init__(self, model_path: str = "visobert_model/final"):
        """Initialize predictor with model"""
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()

        self._TOKENIZER = self.tokenizer # alias
        self.MAX_LENGTH = 512
        self.tokenizer.model_max_length = self.MAX_LENGTH
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")

    def smart_chunk_text(self, item: dict, max_length: int = 510) -> List[dict]:
        """
        Chia nhỏ văn bản thành các chunk dựa trên số lượng TOKEN của model Visobert.
        max_length mặc định là 510 (để chừa chỗ cho 2 special tokens đầu và cuối).
        """
        text = item['text']
        original_labels = item.get('label', [])
        
        # Tách câu
        sentences = sent_tokenize(text)
        
        chunks = []
        
        # Biến lưu trữ các câu trong chunk hiện tại
        current_chunk_sentences = [] # List các tuple (start, end, text)
        current_token_count = 0      # Đổi tên biến để rõ nghĩa hơn
        
        # Con trỏ tìm kiếm trong văn bản gốc
        search_cursor = 0
        
        def flush_chunk(sent_buffer):
            if not sent_buffer:
                return
                
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
            
            chunks.append({
                "text": chunk_text, 
                "label": new_labels
            })

        for sent in sentences:
            # Tìm vị trí chính xác của câu
            sent_start = text.find(sent, search_cursor)
            if sent_start == -1: 
                sent_start = search_cursor
                
            sent_end = sent_start + len(sent)
            
            sent_token_ids = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_token_count = len(sent_token_ids)
            
            # Kiểm tra giới hạn token
            if current_token_count + sent_token_count <= max_length:
                current_chunk_sentences.append((sent_start, sent_end, sent))
                current_token_count += sent_token_count
            else:
                # Đóng gói chunk cũ
                flush_chunk(current_chunk_sentences)
                
                # Tạo chunk mới
                current_chunk_sentences = [(sent_start, sent_end, sent)]
                current_token_count = sent_token_count
                
            # Cập nhật con trỏ tìm kiếm
            search_cursor = sent_end

        # Đóng gói chunk cuối cùng
        if current_chunk_sentences:
            flush_chunk(current_chunk_sentences)
                
        return chunks

    def fix_bio_tags(self, tags):
        """
        Fix invalid BIO tag sequences by converting first I- tags without preceding B- to B- tags
        """
        fixed_tags = list(tags)
        
        for i in range(len(fixed_tags)):
            if fixed_tags[i].startswith('I-'):
                entity_type = fixed_tags[i][2:]
                
                if i == 0 or (not fixed_tags[i-1].startswith('B-' + entity_type) and 
                            not fixed_tags[i-1].startswith('I-' + entity_type)):
                    fixed_tags[i] = 'B-' + entity_type
        
        return fixed_tags

    def predict_text(self, text: str) -> Dict:
        """
        Dự đoán entities trong văn bản
        
        Args:
            text: Văn bản cần dự đoán
        """
        chunks = self.smart_chunk_text({'text': text}, max_length=self.MAX_LENGTH)
        all_entities = []
        current_offset = 0
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_entities = self._predict_chunk(chunk_text)
            
            # Find the chunk in the original text starting from current_offset
            chunk_start = text.find(chunk_text, current_offset)
            if chunk_start != -1:
                for entity in chunk_entities:
                    entity["start"] = int(entity["start"] + chunk_start)
                    entity["end"] = int(entity["end"] + chunk_start)
                    all_entities.append(entity)
                
                # Update current_offset to after this chunk to avoid matching earlier occurrences
                current_offset = chunk_start + len(chunk_text)
            else:
                # If not found, skip adjusting offsets for this chunk
                print("Warning: chunk not found in original text while reconstructing offsets")
        
        entities_by_type = {}
        for entity in all_entities:
            label = entity["label"]
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        return {
            "text": text,
            "entities": all_entities,
            "entities_by_type": entities_by_type,
            "total_entities": len(all_entities),
            "entity_types": list(entities_by_type.keys())
        }

    def _predict_chunk(self, text: str) -> List[Dict]:
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = inputs.pop("offset_mapping")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
        
        # Convert to CPU
        predictions = predictions.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        offset_mapping = offset_mapping.cpu().numpy()[0]
        
        # Convert predictions to labels
        raw_labels = []
        valid_indices = []
        
        for i, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            if start == end:  # Skip special tokens
                continue
            
            label = self.model.config.id2label[int(pred_id)]
            raw_labels.append(label)
            valid_indices.append((i, conf, start, end))
        
        # Fix BIO tag sequences
        fixed_labels = self.fix_bio_tags(raw_labels)
        
        # Extract entities from fixed labels
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
                current_entity["confidence"] = max(current_entity["confidence"], float(conf))
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def format_results(self, results: Dict, show_confidence: bool = True) -> str:
        """Format kết quả để hiển thị"""
        output = []
        output.append(f"📝 Text: {results['text'][:100]}{'...' if len(results['text']) > 100 else ''}")
        output.append(f"📊 Found {results['total_entities']} entities")
        output.append("")
        
        if results['entities']:
            output.append("🏷️ Detected Entities:")
            for i, entity in enumerate(results['entities'], 1):
                confidence_str = f" (confidence: {entity['confidence']:.3f})" if show_confidence else ""
                output.append(f"   {i}. '{entity['text']}' → {entity['label']}{confidence_str}")
            
            output.append("")
            output.append("📋 By Category:")
            for entity_type in sorted(results['entities_by_type'].keys()):
                entities = results['entities_by_type'][entity_type]
                output.append(f"   {entity_type}: {len(entities)} entities")
                for entity in entities:
                    output.append(f"      • {entity['text']}")
        else:
            output.append("❌ No entities detected")
        
        return "\n".join(output)


def main():
    """Test function đơn giản"""
    # Simple test
    predictor = VietnameseNERPredictor()
    
    test_text ="""[MEGA LIVE 14.11] DEAL ĐỈNH THÌNH LÌNH - LÀM ĐẸP SIÊU DÍNH Cùng Ca sĩ THU THỦY Săn deal ĐỒNG GIÁ từ 99K - Quà tặng Làm đẹp siêu khủng----------Danh sách các dịch vụ siêu Hot sẽ xuất hiện trong phiên live lần này, với mức giá ĐỘC QUYỀN siêu giảm:- Chăm Sóc Da Cao Cấp 3in1- Dr.Vip Chăm Sóc Da Lão Hoá ECM- Dr.Vip Ủ Trắng Face Collagen- Dr.Vip Chăm Sóc Vùng Mắt ECM - Xoá nhăn vết chân chim- Dr.Vip Collagen Thuỷ Phân - Ức Chế Đốm Nâu- Dr. Acne Trị Mụn Chuẩn Y Khoa- Dr.Seoul Laser Pico 5.0- Dr.Slim Giảm Mỡ Exilis Detox- Dr. White Tắm Trắng Hoàng Gia- Phun mày- Phun mí- Phun môiNgoài ra, các hoạt động cộng hưởng tại phiên live: Giao lưu, trò chuyện, chia sẻ kiến thức làm đẹp cùng ca sĩ Thu Thủy Tư vấn & giải đáp về dịch vụ cùng Seoul Center Tham gia minigame - Nhận quà độc quyền thương hiệuTất cả DEAL hời đã sẵn sàng "lên kệ" vào lúc 19h00 | 14.11.2024 tại FB/ Tiktok Seoul Center và Fb/tiktok ca sĩ Thu Thủy Giảm giá kịch sàn, chỉ có trên live Đặt lịch săn ngay làm đẹp đón tết cùng Thu Thủy nhé!-------------Hệ Thống Thẩm Mỹ Quốc Tế Seoul CenterSẵn sàng lắng nghe mọi ý kiến của khách hàng: 1800 3333Đặt lịch ngay với Top dịch vụ đặc quyền: Website: Zalo: Tiktok: Youtube: Top 10 Thương Hiệu Xuất Sắc Châu Á 2022 & 2023Huy Chương Vàng Sản Phẩm, Dịch Vụ Chất Lượng Châu Á 2023Thương Hiệu Thẩm Mỹ Dẫn Đầu Việt Nam 2024SEOUL CENTER - PHỤNG SỰ TỪ TÂM#SeoulCenter #ThamMyVien"""
    
    results = predictor.predict_text(test_text)
    print(predictor.format_results(results))


if __name__ == "__main__":
    main()