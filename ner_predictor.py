#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Vietnamese NER Prediction
Chỉ có function cơ bản để dự đoán NER từ model đã train
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict


class VietnameseNERPredictor:
    """Simple Vietnamese NER Predictor"""
    
    def __init__(self, model_path: str = "./mdeberta_ner_model/final"):
        """Initialize predictor with model"""
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
    
    def smart_chunk_text(self, text: str, max_length: int = 400) -> List[str]:
        chunks = []
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text)
        text_without_hashtags = re.sub(hashtag_pattern, '', text).strip()
        
        lines = text_without_hashtags.split('\n')
        current_chunk = ""
        contact_block = ""
        in_contact_block = False
        contact_keywords = ['hotline', 'địa chỉ', 'website', 'email', 'zalo', 'facebook', 'tel', 'fax', 'tổng đài']
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            line_lower = line.lower()
            is_contact_line = any(keyword in line_lower for keyword in contact_keywords)
            has_address = bool(re.search(r'(số\s+\d+|đường|phường|quận|tỉnh|thành phố)', line_lower))
            has_org_pattern = bool(re.search(r'<ORG>.*?</ORG>', line))
            has_addr_pattern = bool(re.search(r'<ADDR>.*?</ADDR>', line))
            
            if is_contact_line or has_address or has_org_pattern or has_addr_pattern:
                if not in_contact_block:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    in_contact_block = True
                    contact_block = line
                else:
                    contact_block += " " + line
            else:
                if in_contact_block:
                    chunks.append(contact_block.strip())
                    contact_block = ""
                    in_contact_block = False
                
                if re.match(r'^[\-\*]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    list_item = line
                    j = i + 1
                    while j < len(lines) and lines[j].strip():
                        next_line = lines[j].strip()
                        if not (re.match(r'^[\-\*]\s+', next_line) or re.match(r'^\d+[\.\)]\s+', next_line)):
                            next_line_lower = next_line.lower()
                            is_next_contact = any(keyword in next_line_lower for keyword in contact_keywords)
                            
                            if not is_next_contact:
                                list_item += " " + next_line
                                j += 1
                            else:
                                break
                        else:
                            break
                    
                    chunks.append(list_item.strip())
                    i = j - 1
                else:
                    if len(current_chunk) + len(line) + 1 <= max_length:
                        if current_chunk:
                            current_chunk += " " + line
                        else:
                            current_chunk = line
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line
            
            i += 1
        
        if in_contact_block and contact_block:
            chunks.append(contact_block.strip())
        elif current_chunk:
            chunks.append(current_chunk.strip())
        
        if hashtags and chunks:
            hashtag_text = " " + " ".join(hashtags)
            chunks[-1] += hashtag_text
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                sentences = re.split(r'[.!?]\s+|[.!?]$|(?<=\.)\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ])', chunk)
                
                current_subchunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                        
                    if len(current_subchunk) + len(sent) + 2 <= max_length:
                        if current_subchunk:
                            current_subchunk += ". " + sent
                        else:
                            current_subchunk = sent
                    else:
                        if current_subchunk:
                            final_chunks.append(current_subchunk.strip())
                        
                        if len(sent) > max_length:
                            clauses = re.split(r',\s+|;\s+|\s+và\s+|\s+hoặc\s+|\s+nhưng\s+|\s+mà\s+', sent)
                            current_subchunk = ""
                            for clause in clauses:
                                clause = clause.strip()
                                if len(current_subchunk) + len(clause) + 2 <= max_length:
                                    if current_subchunk:
                                        current_subchunk += ", " + clause
                                    else:
                                        current_subchunk = clause
                                else:
                                    if current_subchunk:
                                        final_chunks.append(current_subchunk.strip())
                                    current_subchunk = clause
                        else:
                            current_subchunk = sent
                
                if current_subchunk:
                    final_chunks.append(current_subchunk.strip())
        
        return [chunk for chunk in final_chunks if chunk.strip()]

    def predict_text(self, text: str) -> Dict:
        """
        Dự đoán entities trong văn bản
        
        Args:
            text: Văn bản cần dự đoán
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        chunks = self.smart_chunk_text(text, max_length=400)
        all_entities = []
        
        current_offset = 0
        for chunk in chunks:
            chunk_entities = self._predict_chunk(chunk)
            
            chunk_start = text.find(chunk, current_offset)
            if chunk_start != -1:
                for entity in chunk_entities:
                    entity["start"] += chunk_start
                    entity["end"] += chunk_start
                    all_entities.append(entity)
            
            current_offset = text.find(chunk, current_offset) + len(chunk)
        
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
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=2)
            confidences = torch.softmax(outputs.logits, dim=2).max(dim=2)[0]
        
        # Convert to CPU
        predictions = predictions.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        offset_mapping = offset_mapping.cpu().numpy()[0]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            if start == end:  # Skip special tokens
                continue
                
            label = self.model.config.id2label[int(pred_id)]
            
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