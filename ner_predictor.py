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
from underthesea import sent_tokenize as vn_sent_tokenize


class VietnameseNERPredictor:
    """Simple Vietnamese NER Predictor"""
    
    def __init__(self, model_path: str = "mdeberta_ner_model/final"):
        """Initialize predictor with model"""
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()

        self._TOKENIZER = self.tokenizer # alias
        self.MAX_LENGTH = 512
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")

    def _vn_sentences(self, s: str) -> List[str]:
        raise NotImplementedError
        return [t.strip() for t in vn_sent_tokenize(s) if t.strip()]
    
    def remove_hrules(self, text: str) -> str:
        """Xóa chuỗi kẻ ngang (----, ————, ____) ≥ 4 ký tự, giữ nguyên các ký tự khác."""
        return re.sub(r"[-–—_]{4,}", "", text)
    
    def smart_chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Chunking văn bản tối ưu để đạt gần max_length (tính theo TOKEN)
        
        Quy tắc 1: Các mục trong danh sách có thể ghép lại với nhau
        Quy tắc 2: Các khối liên hệ và siêu dữ liệu là không thể chia cắt  
        Quy tắc 3: Xử lý Hashtag
        """
        # Helper: đếm token (kể cả special tokens để sát giới hạn mô hình)
        def _tok_len(s: str) -> int:
            # Lưu ý: encode cả special tokens để mỗi chunk khi đưa vào model không vượt quá max_length
            return len(self._TOKENIZER.encode(s, add_special_tokens=True))

        # Validate input
        if not text or not text.strip():
            return []
        
        chunks: List[str] = []
        
        try:
            # Quy tắc 3: Xử lý hashtag - tách ra và lưu trữ riêng
            hashtag_pattern = r'#\w+'
            hashtags = re.findall(hashtag_pattern, text)
            text_without_hashtags = re.sub(hashtag_pattern, '', text).strip()
            
            # Xoá các chuỗi kẻ ngang trước khi chunk
            text_without_hashtags = self.remove_hrules(text_without_hashtags)
            
            # Tách văn bản thành các segments theo loại
            segments = []
            lines = text_without_hashtags.split('\n')
            
            # Các từ khóa để nhận diện khối liên hệ
            contact_keywords = ['hotline', 'địa chỉ', 'website', 'email', 'zalo', 'facebook', 'tel', 'fax', 'tổng đài']
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                    
                # Quy tắc 2: Phát hiện khối liên hệ/siêu dữ liệu
                line_lower = line.lower()
                is_contact_line = any(keyword in line_lower for keyword in contact_keywords)
                has_address = bool(re.search(r'(số\s+\d+|đường|phường|quận|tỉnh|thành phố)', line_lower))
                has_org_pattern = bool(re.search(r'<ORG>.*?</ORG>', line))
                has_addr_pattern = bool(re.search(r'<ADDR>.*?</ADDR>', line))
                has_phone_pattern = bool(re.search(r'<PER>.*?</PER>', line))
                
                if is_contact_line or has_address or has_org_pattern or has_addr_pattern or has_phone_pattern:
                    # Gom khối liên hệ liền nhau (không cắt nhỏ)
                    contact_block = line
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                            
                        next_line_lower = next_line.lower()
                        is_next_contact = any(keyword in next_line_lower for keyword in contact_keywords)
                        has_next_address = bool(re.search(r'(số\s+\d+|đường|phường|quận|tỉnh|thành phố)', next_line_lower))
                        has_next_org = bool(re.search(r'<ORG>.*?</ORG>', next_line))
                        has_next_addr = bool(re.search(r'<ADDR>.*?</ADDR>', next_line))
                        has_next_phone = bool(re.search(r'<PER>.*?</PER>', next_line))
                        
                        if (is_next_contact or has_next_address or has_next_org or 
                            has_next_addr or has_next_phone or
                            re.match(r'^\d+', next_line) or  # có thể là số điện thoại
                            'www.' in next_line_lower or '.com' in next_line_lower):  # website
                            contact_block += " " + next_line
                            j += 1
                        else:
                            break
                    
                    segments.append({"type": "contact", "content": contact_block})
                    i = j - 1
                    
                # Quy tắc 1: Xử lý mục danh sách
                elif re.match(r'^[\-\*]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
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
                    segments.append({"type": "list_item", "content": list_item})
                    i = j - 1

                else:
                    segments.append({"type": "normal", "content": line})
                i += 1
            
            # Ghép segments thành chunks theo token
            current_chunk = ""
            segment_buffer = []
            
            def _join_contents(items) -> str:
                return " ".join([s["content"] for s in items]).strip()

            def try_add_segment(segment) -> bool:
                # Thử thêm vào buffer + current_chunk, kiểm tra theo token
                test_content = _join_contents(segment_buffer + [segment])
                test_chunk = (current_chunk + " " + test_content).strip() if current_chunk else test_content
                return _tok_len(test_chunk) <= max_length
            
            def flush_buffer():
                nonlocal current_chunk, segment_buffer
                if not segment_buffer:
                    return
                buffer_content = _join_contents(segment_buffer)
                if current_chunk:
                    combined = (current_chunk + " " + buffer_content).strip()
                    if _tok_len(combined) <= max_length:
                        current_chunk = combined
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = buffer_content
                else:
                    current_chunk = buffer_content
                segment_buffer = []
            
            def flush_current():
                nonlocal current_chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            for segment in segments:
                if segment["type"] == "contact":
                    flush_buffer()
                    flush_current()
                    if _tok_len(segment["content"]) <= max_length:
                        chunks.append(segment["content"].strip())
                    else:
                        # cắt contact “an toàn” theo câu/cụm (giữ các phần liền kề)
                        def _split_contact(s: str) -> List[str]:
                            out, cur = [], ""
                            sents = self._vn_sentences(s) or [s]
                            for sent in sents:
                                cand = (cur + (" " if cur else "") + sent).strip()
                                if _tok_len(cand) <= max_length:
                                    cur = cand
                                else:
                                    if cur:
                                        out.append(cur.strip())
                                    # tách mệnh đề, rồi fallback theo từ nếu cần
                                    clauses = re.split(r',\s+|;\s+|\s+\-\s+|\s+\|\s+', sent)
                                    buf = ""
                                    for c in clauses:
                                        c = c.strip()
                                        if not c:
                                            continue
                                        cand2 = (buf + (", " if buf else "") + c).strip()
                                        if _tok_len(cand2) <= max_length:
                                            buf = cand2
                                        else:
                                            if buf:
                                                out.append(buf.strip())
                                            wbuf = ""
                                            for w in c.split():
                                                cand3 = (wbuf + (" " if wbuf else "") + w).strip()
                                                if _tok_len(cand3) <= max_length:
                                                    wbuf = cand3
                                                else:
                                                    if wbuf:
                                                        out.append(wbuf.strip())
                                                    wbuf = w
                                            if wbuf:
                                                out.append(wbuf.strip())
                                            buf = ""
                                    if buf:
                                        out.append(buf.strip())
                                    cur = ""
                            if cur:
                                out.append(cur.strip())
                            return out

                        chunks.extend(_split_contact(segment["content"]))
            
            # Flush phần còn lại
            flush_buffer()
            flush_current()
            
            # Thêm hashtag (nếu có) vào các chunk mà vẫn không vượt token
            if hashtags and chunks:
                hashtag_text = " " + " ".join(hashtags)
                # thử vào chunk cuối trước
                candidate = (chunks[-1] + hashtag_text).strip()
                if _tok_len(candidate) <= max_length:
                    chunks[-1] = candidate
                else:
                    # thử lùi dần các chunk trước đó
                    placed = False
                    for k in range(len(chunks) - 2, -1, -1):
                        test = (chunks[k] + hashtag_text).strip()
                        if _tok_len(test) <= max_length:
                            chunks[k] = test
                            placed = True
                            break
                    if not placed:
                        chunks.append(" ".join(hashtags))
            elif hashtags:
                chunks.append(" ".join(hashtags))
            
            # Merge các chunk còn nhỏ nếu gộp lại không vượt token
            optimized_chunks: List[str] = []
            for ch in chunks:
                if optimized_chunks:
                    merged = (optimized_chunks[-1] + " " + ch).strip()
                    if _tok_len(merged) <= max_length:
                        optimized_chunks[-1] = merged
                    else:
                        optimized_chunks.append(ch)
                else:
                    optimized_chunks.append(ch)
            
            # Xử lý các chunk vẫn quá dài (token) bằng cách tách câu/cụm (regex)
            final_chunks: List[str] = []
            for ch in optimized_chunks:
                if _tok_len(ch) <= max_length:
                    final_chunks.append(ch)
                    continue
                
                # Tách câu, sau đó gộp lại theo token
                sentences = self._vn_sentences(ch)
                if not sentences:
                    sentences = [ch]
                
                current_subchunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    # thử thêm câu vào subchunk (thêm ". " để giữ dấu kết thúc)
                    test_sub = (current_subchunk + (" " if current_subchunk else "") + sent).strip()
                    if _tok_len(test_sub) <= max_length:
                        current_subchunk = test_sub
                    else:
                        if current_subchunk:
                            final_chunks.append(current_subchunk.strip())
                        # nếu câu quá dài so với max_length, tách tiếp theo clause
                        if _tok_len(sent) > max_length:
                            def find_best_split_point(text, max_tokens):
                                """Tìm điểm cắt tốt nhất để giữ nguyên ý nghĩa"""
                                # Ưu tiên cắt ở cuối câu hoàn chỉnh
                                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
                                for pos in reversed(sentence_ends):
                                    if _tok_len(text[:pos]) <= max_tokens:
                                        return pos
                                
                                # Nếu không có, cắt ở dấu phẩy + từ nối
                                clause_ends = [m.end() for m in re.finditer(r'[,;]\s+(?:và|hoặc|nhưng|mà|nên|nếu|khi|để)\s+', text)]
                                for pos in reversed(clause_ends):
                                    if _tok_len(text[:pos]) <= max_tokens:
                                        return pos
                                
                                # Cuối cùng mới cắt ở dấu phẩy thường
                                comma_ends = [m.end() for m in re.finditer(r',\s+', text)]
                                for pos in reversed(comma_ends):
                                    if _tok_len(text[:pos]) <= max_tokens:
                                        return pos
                                
                                return None

                            # Sử dụng hàm này thay vì split đơn giản
                            clauses = []
                            remaining = sent
                            while remaining and _tok_len(remaining) > max_length:
                                split_pos = find_best_split_point(remaining, max_length)
                                if split_pos:
                                    clauses.append(remaining[:split_pos].strip())
                                    remaining = remaining[split_pos:].strip()
                                else:
                                    # Fallback: cắt theo từ
                                    words = remaining.split()
                                    current = ""
                                    for word in words:
                                        test = (current + " " + word).strip() if current else word
                                        if _tok_len(test) <= max_length:
                                            current = test
                                        else:
                                            if current:
                                                clauses.append(current)
                                            current = word
                                            break
                                    remaining = " ".join(words[len(current.split()):])
                            if remaining:
                                clauses.append(remaining)
                            current_subchunk = ""
                            for clause in clauses:
                                clause = clause.strip()
                                if not clause:
                                    continue
                                test_clause = (current_subchunk + (", " if current_subchunk else "") + clause).strip()
                                if _tok_len(test_clause) <= max_length:
                                    current_subchunk = test_clause
                                else:
                                    if current_subchunk:
                                        final_chunks.append(current_subchunk.strip())
                                    current_subchunk = clause
                        else:
                            current_subchunk = sent
                
                if current_subchunk:
                    final_chunks.append(current_subchunk.strip())
            
            # Loại bỏ chunk rỗng
            return [c for c in final_chunks if c.strip()]
    
        except Exception:
            # Fallback: tách theo câu trước, sau đó mới theo từ
            try:
                sentences = self._vn_sentences(text)
                chunks = []
                current_chunk = ""
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                        
                    test_chunk = (current_chunk + " " + sent).strip() if current_chunk else sent
                    if _tok_len(test_chunk) <= max_length:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # Nếu câu đơn lẻ vẫn quá dài, tách theo từ cẩn thận
                        if _tok_len(sent) > max_length:
                            words = sent.split()
                            word_chunk = ""
                            for word in words:
                                test_word = (word_chunk + " " + word).strip() if word_chunk else word
                                if _tok_len(test_word) <= max_length:
                                    word_chunk = test_word
                                else:
                                    if word_chunk:
                                        chunks.append(word_chunk.strip())
                                    word_chunk = word
                            if word_chunk:
                                current_chunk = word_chunk.strip()
                            else:
                                current_chunk = ""
                        else:
                            current_chunk = sent
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [c for c in chunks if c.strip()]
            
            except:
                # Ultimate fallback: trả về text gốc nếu không thể xử lý
                if _tok_len(text) <= max_length:
                    return [text.strip()]
                else:
                    # Cắt cứng theo từ nhưng cẩn thận hơn
                    words = text.split()
                    chunks = []
                    current = ""
                    
                    for word in words:
                        test = (current + " " + word).strip() if current else word
                        if _tok_len(test) <= max_length:
                            current = test
                        else:
                            if current:
                                chunks.append(current.strip())
                                current = word
                            else:
                                # Từ đơn quá dài, buộc phải cắt
                                chunks.append(word)
                                current = ""
                    
                    if current:
                        chunks.append(current.strip())
                    
                    return [c for c in chunks if c.strip()]

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
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        chunks = self.smart_chunk_text(text)
        for idx, chunk in enumerate(chunks):
            print(f"chunk {idx}: {chunk}")
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