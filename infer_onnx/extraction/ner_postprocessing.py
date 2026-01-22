#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved NER Postprocessing - Handle fuzzy substring matching
"""

import re
from typing import Dict, List
from collections import Counter
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    print("Warning: rapidfuzz not installed. Install with: pip install rapidfuzz")
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False


class NERPostProcessor:
    """Improved Postprocessing cho kết quả NER với fuzzy substring matching"""
    
    def __init__(self):
        # Danh sách các prefix học hàm học vị cần loại bỏ
        self.academic_prefixes = [
            # Học vị tiếng Việt
            r'(?i)\b(?:bác\s*sĩ?|bs\.?)\b\s*',
            r'(?i)\b(?:cử\s*nhân|cn\.?)\b\s*',
            r'(?i)\b(?:thạc\s*sĩ?|ths\.?)\b\s*',
            r'(?i)\b(?:tiến\s*sĩ?|ts\.?)\b\s*',
            r'(?i)\b(?:giáo\s*sư|gs\.?)\b\s*',
            r'(?i)\b(?:phó\s*giáo\s*sư|pgs\.?)\b\s*',
            
            # Chức danh y tế
            # ===== Chuyên khoa (đủ cả dạng chữ & viết tắt) =====
            r'(?i)\b(?:chuyên\s*khoa\s*(?:i|1)|cki\.?|ck1\.?)\b\s*',   # Chuyên khoa I / CKI / CK1
            r'(?i)\b(?:chuyên\s*khoa\s*(?:ii|2)|ckii\.?|ck2\.?)\b\s*', # Chuyên khoa II / CKII / CK2
            r'(?i)\b(?:chuyên\s*gia)\b\s*',                            # Chuyên gia
            
            # ===== Bác sĩ chuyên khoa (BSCKI, BSCKII) =====
            r'(?i)\b(?:bscki|bsck\s*i|bsck\s*1|bs\.?ck\.?i|bs\.?ck\.?1)\b\s*',   # BSCKI / Bsck I / BSCK 1
            r'(?i)\b(?:bsckii|bsck\s*ii|bsck\s*2|bs\.?ck\.?ii|bs\.?ck\.?2)\b\s*', # BSCKII / Bsck II / BSCK 2

            r'(?i)\bbscc\.?\b',             # BSCC.
            
            # Học vị tiếng Anh
            r'(?i)\b(?:dr\.?|doctor)\b\s*',
            r'(?i)\b(?:prof\.?|professor)\b\s*',
            r'(?i)\b(?:mr\.?|mrs\.?|ms\.?|miss\.?)\b\s*',
            
            # Các chức danh khác
            r'(?i)\b(?:dược\s*sĩ?|ds\.?)\b\s*',
            r'(?i)\b(?:y\s*tá|điều\s*dưỡng)\b\s*',
            r'(?i)\b(?:kỹ\s*thuật\s*viên|ktv\.?)\b\s*',
            r'(?i)\b(?:thầy\s*thuốc|tt\.?)\b\s*',
            r'(?i)\b(?:giám\s*đốc|gd\.?|gđ\.?)\b\s*',
        ]
        
        # Compile regex patterns cho academic prefixes
        self.prefix_patterns = [re.compile(pattern) for pattern in self.academic_prefixes]
        
        # Regex patterns cho các địa chỉ cho phép
        self.allowed_locations = [
            # TPHCM - các cách viết đầy đủ
            r'(?i).*(?:tp\s*\.?\s*)?h[ồô]\s*ch[íi]\s*minh.*',
            r'(?i).*sài\s*g[òon].*',
            r'(?i).*hcm.*',
            r'(?i).*tphcm.*',
            r'(?i).*thành\s*phố\s*hồ\s*chí\s*minh.*',
            r'(?i).*ho\s*chi\s*minh.*',
            
            # TPHCM - Các quận/huyện cụ thể
            r'(?i).*(?:q\s*\.?\s*|quận\s+)[1-9](?:\d{1,2})?(?:\s|$|,|\.|\-|\/).*',  # Q.1, Q.12, Quận 1, etc.
            r'(?i).*(?:quận|q\s*\.?\s*)(?:bình\s*thạnh|tân\s*bình|gò\s*vấp|phú\s*nhuận|tân\s*phú|bình\s*tân).*',
            r'(?i).*(?:quận|q\s*\.?\s*)(?:thủ\s*đức).*',
            r'(?i).*thủ\s*đức.*',
            r'(?i).*(?:huyện|h\s*\.?\s*)(?:củ\s*chi|hóc\s*môn|bình\s*chánh|nhà\s*bè|cần\s*giờ).*',
            
            # Bình Dương
            r'(?i).*b[ìi]nh\s*d[ươuong].*',
            r'(?i).*binh\s*duong.*',
            r'(?i).*(?:tp\s*\.?\s*|thành\s*phố\s+)?thủ\s*dầu\s*một.*',
            r'(?i).*(?:tp\s*\.?\s*|thành\s*phố\s+)?dĩ\s*an.*',
            r'(?i).*(?:tp\s*\.?\s*|thành\s*phố\s+)?thuận\s*an.*',
            r'(?i).*(?:thị\s*xã\s+|tx\s*\.?\s*)?tân\s*uyên.*',
            r'(?i).*(?:thị\s*xã\s+|tx\s*\.?\s*)?bến\s*cát.*',
            r'(?i).*(?:huyện|h\s*\.?\s*)(?:bàu\s*bàng|dầu\s*tiếng|phú\s*giáo|bắc\s*tân\s*uyên).*',
            
            # Bà Rịa - Vũng Tàu
            r'(?i).*b[àa]\s*r[ịi]a\s*v[ũu]ng\s*t[àa]u.*',
            r'(?i).*ba\s*ria\s*vung\s*tau.*',
            r'(?i).*brvt.*',
            r'(?i).*(?:tp\s*\.?\s*|thành\s*phố\s+)?vũng\s*tàu.*',
            r'(?i).*(?:tp\s*\.?\s*|thành\s*phố\s+)?bà\s*rịa.*',
            r'(?i).*(?:thị\s*xã\s+|tx\s*\.?\s*)?phú\s*mỹ.*',
            r'(?i).*(?:huyện|h\s*\.?\s*)(?:châu\s*đức|xuyên\s*mộc|đất\s*đỏ|tân\s*thành|long\s*điền).*',
            r'(?i).*côn\s*đảo.*',
        ]
        
        # Compile regex patterns
        self.location_patterns = [re.compile(pattern) for pattern in self.allowed_locations]
    
    def fuzzy_similarity(self, a: str, b: str) -> float:
        """Tính độ tương tự fuzzy giữa 2 string với RapidFuzz"""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(a.lower().strip(), b.lower().strip()) / 100.0
        else:
            # Fallback to difflib
            return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
    
    def fuzzy_partial_similarity(self, a: str, b: str) -> float:
        """Tính độ tương tự partial (substring) fuzzy"""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.partial_ratio(a.lower().strip(), b.lower().strip()) / 100.0
        else:
            # Fallback: tự implement partial ratio đơn giản
            a_clean = a.lower().strip()
            b_clean = b.lower().strip()
            
            if len(a_clean) > len(b_clean):
                longer, shorter = a_clean, b_clean
            else:
                longer, shorter = b_clean, a_clean
            
            best_ratio = 0
            for i in range(len(longer) - len(shorter) + 1):
                substring = longer[i:i + len(shorter)]
                ratio = SequenceMatcher(None, shorter, substring).ratio()
                best_ratio = max(best_ratio, ratio)
            
            return best_ratio
    
    def is_substring_or_similar(self, short: str, long: str, threshold: float = 0.8, partial_threshold: float = 0.9) -> bool:
        """
        Kiểm tra xem short có phải substring hoặc tương tự với long không
        Bao gồm cả fuzzy substring matching
        
        Args:
            short: Chuỗi ngắn hơn
            long: Chuỗi dài hơn  
            threshold: Ngưỡng cho full similarity
            partial_threshold: Ngưỡng cho partial similarity (cao hơn vì chỉ so sánh 1 phần)
        """
        short_clean = short.lower().strip()
        long_clean = long.lower().strip()
        
        # Case 1: Exact substring
        if short_clean in long_clean:
            return True
        
        # Case 2: Full fuzzy similarity (toàn bộ string tương tự)
        if self.fuzzy_similarity(short_clean, long_clean) >= threshold:
            return True
        
        # Case 3: Partial fuzzy similarity (substring nhưng viết khác 1 tí)
        # Ví dụ: "HANA" trong "Hệ thống thẩm mỹ HANNA"
        if self.fuzzy_partial_similarity(short_clean, long_clean) >= partial_threshold:
            return True
            
        return False
    
    def is_similar_organizations(self, org1: str, org2: str, threshold: float = 0.8, partial_threshold: float = 0.9) -> bool:
        """
        Kiểm tra 2 organizations có tương tự nhau không
        Xử lý cả trường hợp substring và fuzzy matching
        """
        org1_clean = org1.lower().strip()
        org2_clean = org2.lower().strip()
        
        # Xác định cái nào ngắn hơn, dài hơn
        if len(org1_clean) <= len(org2_clean):
            shorter, longer = org1_clean, org2_clean
        else:
            shorter, longer = org2_clean, org1_clean
        
        # Kiểm tra substring hoặc similar
        return self.is_substring_or_similar(shorter, longer, threshold, partial_threshold)
    
    def is_allowed_location(self, location: str) -> bool:
        """Kiểm tra xem địa chỉ có thuộc vùng cho phép không"""
        location_clean = location.strip()
        
        for pattern in self.location_patterns:
            if pattern.search(location_clean):
                return True
        
        return False
    
    def remove_academic_prefix(self, person_name: str) -> str:
        """
        Loại bỏ các prefix học hàm học vị từ tên người
        
        Args:
            person_name: Tên người có thể chứa prefix
            
        Returns:
            Tên người đã loại bỏ prefix
        """
        cleaned_name = person_name.strip()
        
        # Áp dụng từng pattern để loại bỏ prefix
        for pattern in self.prefix_patterns:
            cleaned_name = pattern.sub('', cleaned_name).strip()
        
        # Loại bỏ khoảng trắng thừa và normalize
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
        
        # Nếu sau khi loại bỏ prefix mà tên trống thì trả về tên gốc
        if not cleaned_name:
            return person_name.strip()
        
        return cleaned_name
    
    def expand_medical_abbreviations(self, org_name: str) -> str:
        """
        Mở rộng các từ viết tắt trong tên cơ sở y tế
        
        Args:
            org_name: Tên cơ sở có thể chứa viết tắt
            
        Returns:
            Tên cơ sở đã mở rộng viết tắt
        """
        if not org_name:
            return org_name
            
        # Danh sách các viết tắt y tế phổ biến
        abbreviations = {
            # ===== Viết tắt bệnh viện & đa khoa =====
            r'\bbvdk\b': 'bệnh viện đa khoa',  # đặt trước để tránh xung đột với bv + dk
            r'\bbvđk\b': 'bệnh viện đa khoa',   # thêm dạng có đ
            r'\bbv\b': 'bệnh viện',
            r'\bdk\b': 'đa khoa',
            r'\bđk\b': 'đa khoa',
            r'\bttyt\b': 'trung tâm y tế',

            # ===== Các viết tắt chuyên khoa =====
            r'\bpk\b': 'phòng khám',
            r'\btm\b': 'thẩm mỹ',
            r'\btw\b': 'trung ương',
            r'\bqt\b': 'quốc tế',
            r'\bđhyd?\b': 'đại học y dược',
            r'\bdhyd?\b': 'đại học y dược',
            r'\btkb\b': 'tai - mũi - họng',
            r'\btmh\b': 'tai - mũi - họng',
            r'\brhm?\b': 'răng - hàm - mặt',
            r'\bphcn\b': 'phục hồi chức năng',
            r'\bcchs\b': 'cấp cứu hồi sức',

            # ===== Viết tắt địa danh =====
            r'\btp\.?\s*(?:hồ\s*chí\s*minh|hcm)\b': 'thành phố hồ chí minh',
            r'\btp\.?(?=\s|$)': 'thành phố',
            r'\bhcm\b': 'hồ chí minh',
            r'\bq\.?\s*(\d+)\b': r'quận \1',

            #==== Các viết tắt khác =====
            r'\bCP\b': 'cổ phần',
            r'\bcty\b': 'công ty',
            r'\btnhh\b': 'trách nhiệm hữu hạn',
        }
        
        expanded = org_name.lower().strip()
        
        # Áp dụng từng pattern để mở rộng viết tắt
        for abbrev_pattern, full_form in abbreviations.items():
            expanded = re.sub(abbrev_pattern, full_form, expanded, flags=re.IGNORECASE)
        
        # Normalize khoảng trắng
        expanded = re.sub(r'\s+', ' ', expanded).strip()
        
        # Chuyển về title case
        if expanded:
            words = expanded.split()
            capitalized_words = []
            for word in words:
                if word:
                    capitalized_words.append(word.capitalize())
            expanded = ' '.join(capitalized_words)
        
        return expanded if expanded else org_name
    
    def process_organizations(self, org_entities: List[Dict], threshold: float = 0.8, partial_threshold: float = 0.9) -> str:
        """
        Xử lý danh sách organizations với improved fuzzy matching
        Trả về list organization name
        
        Args:
            org_entities: List entities của ORG
            threshold: Ngưỡng cho full similarity  
            partial_threshold: Ngưỡng cho partial similarity
        """
        if not org_entities:
            return []
        
        # Lấy tất cả text của organizations và mở rộng viết tắt
        org_texts = [self.expand_medical_abbreviations(entity['text'].strip()) for entity in org_entities]
        
        # Filter bỏ các tên tổ chức quá ngắn hoặc lỗi token
        # Logic: Nếu có <= 3 từ VÀ mỗi từ < 3 ký tự thì loại bỏ
        # Ví dụ loại: "Ệ Th" (2 từ, mỗi từ 1-2 ký tự)
        # Ví dụ giữ: "VinMec" (1 từ, 6 ký tự), "Vin AI" (2 từ, 1 từ >= 3 ký tự)
        filtered_orgs = []
        for org in org_texts:
            components = org.split()
            if len(components) <= 3:
                # Kiểm tra xem có ít nhất 1 component >= 3 ký tự không
                has_valid_component = any(len(comp) >= 3 for comp in components)
                if has_valid_component:
                    filtered_orgs.append(org)
            else:
                # Nếu có > 3 từ thì giữ lại (tên dài, khả năng cao là hợp lệ)
                filtered_orgs.append(org)
        
        org_texts = filtered_orgs
        
        if len(org_texts) == 1:
            return org_texts
        
        # Đếm frequency của từng organization (exact match)
        org_counter = Counter(org_texts)
        
        # Group các organizations tương tự nhau
        grouped_orgs = {}
        processed = set()
        
        for i, org in enumerate(org_texts):
            if org in processed:
                continue
                
            # Tìm tất cả organizations tương tự với org này
            similar_orgs = [org]
            
            for j, other_org in enumerate(org_texts):
                if i != j and other_org not in processed:
                    # Kiểm tra xem có tương tự nhau không (bao gồm fuzzy substring)
                    if self.is_similar_organizations(org, other_org, threshold, partial_threshold):
                        similar_orgs.append(other_org)
            
            # Chọn cái dài nhất trong group
            longest_org = max(similar_orgs, key=len)
            
            # Tính tổng count cho group này
            total_count = sum(org_counter[similar_org] for similar_org in similar_orgs)
            
            grouped_orgs[longest_org] = total_count
            
            # Đánh dấu đã xử lý
            for similar_org in similar_orgs:
                processed.add(similar_org)
        
        # # Chọn organization có count cao nhất
        # if grouped_orgs:
        #     best_org = max(grouped_orgs.items(), key=lambda x: x[1])[0]
        #     return best_org.strip()

        if grouped_orgs:
            sorted_orgs = sorted(grouped_orgs.items(), key=lambda x: x[1], reverse=True)
            return [org for org, count in sorted_orgs]

        return []
    
    def process_addresses(self, loc_entities: List[Dict]) -> List[str]:
        """
        Xử lý danh sách địa chỉ, chỉ giữ lại những địa chỉ thuộc vùng cho phép và dài hơn 10 ký tự
        """
        if not loc_entities:
            return []
        
        allowed_addresses = []
        seen_addresses = set()
        
        for entity in loc_entities:
            address = entity['text'].strip()
            address_lower = address.lower()
            
            # Kiểm tra trùng lặp, location hợp lệ và độ dài >= 10 ký tự
            if (address_lower not in seen_addresses and 
                self.is_allowed_location(address) and 
                len(address) >= 10):
                allowed_addresses.append(address)
                seen_addresses.add(address_lower)
        
        return allowed_addresses
    
    def clean_person_name(self, name: str) -> str:
        """
        Làm sạch tên người - loại bỏ ký tự dư thừa, prefix, etc.
        """
        if not name:
            return ""
            
        cleaned = name.strip()
        
        # Loại bỏ prefix học hàm học vị
        cleaned = self.remove_academic_prefix(cleaned)
        
        # Loại bỏ TẤT CẢ các ký tự không phải chữ cái (alphabet) và khoảng trắng
        # Giữ lại: chữ cái tiếng Việt (a-z, A-Z, À-ỹ) và khoảng trắng
        cleaned = re.sub(r'[^a-zA-ZÀ-ỹ\s]+', ' ', cleaned)
        
        # Normalize khoảng trắng (loại bỏ khoảng trắng thừa)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Normalize case - chuyển về title case cho tên người Việt
        if cleaned and re.search(r'[a-zA-ZÀ-ỹ]', cleaned):
            cleaned = self.normalize_vietnamese_name(cleaned)
        
        return cleaned
    
    def normalize_vietnamese_name(self, name: str) -> str:
        """
        Chuẩn hóa tên người Việt về dạng title case
        """
        # Split thành các từ và capitalize từng từ
        words = name.split()
        normalized_words = []
        
        for word in words:
            # Capitalize từng từ (chữ đầu viết hoa, còn lại viết thường)
            if word:
                normalized_words.append(word.capitalize())
        
        return ' '.join(normalized_words)
    
    def is_substring_name(self, short_name: str, long_name: str) -> bool:
        """
        Kiểm tra xem short_name có phải là substring của long_name không
        (để dedup các trường hợp như "Vũ" trong "Hồ Cao Vũ")
        """
        if not short_name or not long_name:
            return False
            
        short_clean = short_name.lower().strip()
        long_clean = long_name.lower().strip()
        
        # Kiểm tra substring exact
        if short_clean in long_clean and len(short_clean) < len(long_clean):
            return True
            
        return False
    
    def process_persons(self, per_entities: List[Dict]) -> List[str]:
        """
        Xử lý danh sách persons với advanced cleaning và case-insensitive substring dedup
        """
        if not per_entities:
            return []
        
        # Bước 1: Clean tất cả tên
        cleaned_persons = []
        for entity in per_entities:
            person_raw = entity['text'].strip()
            person_cleaned = self.clean_person_name(person_raw)
            
            # Chỉ giữ tên có ý nghĩa (>= 2 ký tự, không chỉ toàn ký tự đặc biệt)
            if person_cleaned and len(person_cleaned) >= 2 and re.search(r'[a-zA-ZÀ-ỹ]', person_cleaned):
                cleaned_persons.append(person_cleaned)
        
        if not cleaned_persons:
            return []
        
        # Bước 2: Loại bỏ exact duplicates (case-insensitive)
        seen_lower = {}
        unique_persons = []
        
        for person in cleaned_persons:
            person_lower = person.lower()
            if person_lower not in seen_lower:
                seen_lower[person_lower] = person
                unique_persons.append(person)
        
        # Bước 3: Loại bỏ substring duplicates (case-insensitive, giữ chuỗi dài nhất)
        final_persons = []
        
        for person in unique_persons:
            is_substring = False
            
            # Kiểm tra xem person này có phải substring của ai đã có trong final_persons không
            for existing in final_persons:
                if self.is_substring_name(person, existing):
                    is_substring = True
                    break
            
            if not is_substring:
                # Kiểm tra ngược lại: person này có chứa substring nào trong final_persons không
                # Nếu có thì thay thế (giữ cái dài hơn)
                to_remove = []
                for i, existing in enumerate(final_persons):
                    if self.is_substring_name(existing, person):
                        to_remove.append(i)
                
                # Xóa các substring cũ
                for i in reversed(to_remove):
                    final_persons.pop(i)
                
                # Thêm person hiện tại
                final_persons.append(person)
        
        return final_persons
    
    def postprocess(self, ner_results: Dict, threshold: float = 0.8, partial_threshold: float = 0.9) -> Dict:
        """
        Main postprocessing function với improved fuzzy matching
        
        Args:
            ner_results: Output từ VietnameseNERPredictor
            threshold: Ngưỡng cho full similarity (0.0-1.0)
            partial_threshold: Ngưỡng cho partial similarity (0.0-1.0)
            
        Returns:
            Dict với format: {
                'organization_name': List[str],
                'address': List[str],
                'person': List[str]
            }
        """
        entities_by_type = ner_results.get('entities_by_type', {})
        
        # Xử lý organizations với improved logic
        org_entities = entities_by_type.get('ORG', [])
        organization_names = self.process_organizations(org_entities, threshold, partial_threshold)
        
        # Xử lý locations/addresses
        loc_entities = entities_by_type.get('ADDR', [])
        addresses = self.process_addresses(loc_entities)
        
        # Xử lý persons
        per_entities = entities_by_type.get('PER', [])
        persons = self.process_persons(per_entities)
        
        # Có thể thêm xử lý cho các entity type khác nếu cần
        gpe_entities = entities_by_type.get('GPE', [])
        if gpe_entities:
            gpe_addresses = self.process_addresses(gpe_entities)
            # Merge và loại bỏ trùng lặp
            all_addresses = addresses + gpe_addresses
            addresses = list(dict.fromkeys(all_addresses))
        
        return {
            'organization_names': organization_names,
            'addresses': addresses,
            'persons': persons
        }


# 🚀 QUICK FUNCTION - Global instance
_processor = None

def get_processor():
    """Lazy initialization của processor"""
    global _processor
    if _processor is None:
        _processor = NERPostProcessor()
    return _processor


def quick_postprocess(ner_results: Dict, threshold: float = 0.8, partial_threshold: float = 0.9) -> Dict:
    """
    🚀 QUICK FUNCTION: Improved postprocess với fuzzy substring matching
    
    Args:
        ner_results: Output từ VietnameseNERPredictor
        threshold: Ngưỡng cho full similarity (0.0-1.0)
        partial_threshold: Ngưỡng cho partial similarity (0.0-1.0, thường cao hơn threshold)
        
    Returns:
        Dict với format: {
            'organization_name': str,
            'address': List[str],
            'person': List[str]
        }
    """
    processor = get_processor()
    return processor.postprocess(ner_results, threshold, partial_threshold)


def test_improved_fuzzy():
    """🧪 Test improved fuzzy substring matching và academic prefix removal"""
    
    processor = NERPostProcessor()
    
    # Test cases cho medical abbreviation expansion
    print("=== Test Medical Abbreviation Expansion ===")
    test_orgs = [
        "BV Chợ Rẫy",
        "BVDK Tân Tạo", 
        "BV TM Hana",
        "Phòng khám PK ABC",
        "BV NT Q7",
        "DHYD TPHCM",
        "BV Răng Hàm Mặt RHMM",
        "Phòng khám TMH Q1",
        "BV Tim TPHCM",
        "Trung tâm PHCN",
        "BV CCHS 115",
        "Bệnh viện Từ Dũ",  # Không có viết tắt
        "BV Q.1",
        "BVDK Q 10"
    ]
    
    for org in test_orgs:
        expanded = processor.expand_medical_abbreviations(org)
        print(f"'{org}' → '{expanded}'")
    print()
    
    # Test cases cho address cleaning
    print("=== Test Address Cleaning ===")
    test_addresses = [
        "123 Nguyễn Văn A, Quận 1, TP.HCM (gần chợ Bến Thành)",
        "456 Lê Lợi, Q.3, TPHCM [tầng 2, toà nhà ABC]",
        "789 Trần Hưng Đạo, Q1 {cạnh ngân hàng Vietcombank}",
        "321 Hai Bà Trưng, Quận 3, gần siêu thị BigC",
        "654 Võ Văn Tần, Q.3, đối diện bệnh viện Chợ Rẫy",
        "987 Nam Kỳ Khởi Nghĩa, phía sau chợ Nguyễn Thiện Thuật",
        "159 Pasteur, Quận 1, lầu 3 block A",
        "753 Cách Mạng Tháng 8, tầng 5 toà nhà Diamond Plaza",
        "852 Nguyễn Thị Minh Khai, toà nhà Landmark 81",
        "951 Đường D1, Quận 7, số 25A căn hộ B1-08",
        "- 147 Nguyễn Du, Q.1, TPHCM.,",  # Có ký tự dư thừa
        ". 258 Lý Tự Trọng, Q.1 ;;",      # Có ký tự dư thừa
        "123 ABC ()",                      # Ngoặc rỗng
        "456 XYZ []",                      # Ngoặc vuông rỗng
        "789 DEF {}",                      # Ngoặc nhọn rỗng
        "Bệnh viện Chợ Rẫy, 201B Nguyễn Chí Thanh, Quận 5, TP.HCM",  # Địa chỉ bình thường
    ]
    
    for address in test_addresses:
        cleaned = processor.clean_address(address)
        print(f"'{address}'")
        print(f"→ '{cleaned}'")
        print()
    
    # Test cases cho person name cleaning và academic prefix removal
    print("=== Test Person Name Cleaning ===")
    test_persons = [
        "Bác sĩ Nguyễn Văn A",
        "BS. Trần Thị B", 
        "Thạc sĩ Lê Minh C",
        "TS. Phạm Đức D",
        "Giáo sư Hoàng Thị E",
        "PGS.TS. Vũ Văn F",
        "Dr. John Smith",
        "CKI Nguyễn Thành G",
        "CKI. Lê Văn M", 
        "CKII Lý Thị H",
        "CKII. Phạm Thị N",
        "CK1. Trần Minh O",
        "CK2. Hoàng Thị P",
        "Dược sĩ Cao Minh I",
        "Y tá Đinh Thị J",
        "Mr. David Wilson",
        "Nguyễn Văn K",  # Không có prefix
        "Bác sĩ",        # Chỉ có prefix
        "- Hồ Cao Vũ",   # Có dấu - ở đầu
        ". Hồ Cao Vũ",   # Có dấu . ở đầu
        ". Vũ",          # Chỉ có tên ngắn
        "Hồ Cao Vũ",     # Tên bình thường
    ]
    
    for person in test_persons:
        cleaned = processor.clean_person_name(person)
        print(f"'{person}' → '{cleaned}'")
    print()
    
    # Test substring deduplication
    print("=== Test Substring Deduplication ===")
    test_entities = [
        {'text': '- Hồ Cao Vũ', 'label': 'PER', 'confidence': 0.95},
        {'text': '. Hồ Cao Vũ', 'label': 'PER', 'confidence': 0.92},
        {'text': '. Vũ', 'label': 'PER', 'confidence': 0.88},
        {'text': 'Hồ Cao Vũ', 'label': 'PER', 'confidence': 0.90},
        {'text': 'Bác sĩ Nguyễn Văn A', 'label': 'PER', 'confidence': 0.93},
        {'text': 'Nguyễn Văn A', 'label': 'PER', 'confidence': 0.89},
        {'text': 'A', 'label': 'PER', 'confidence': 0.85},
    ]
    
    result_persons = processor.process_persons(test_entities)
    print("Input entities:")
    for entity in test_entities:
        print(f"  - '{entity['text']}'")
    print(f"After processing: {result_persons}")
    print()
    
    # Test case-insensitive deduplication (từ data thực của user)
    print("=== Test Case-Insensitive Deduplication (Real Data) ===")
    real_test_entities = [
        {'text': 'BÁC VĂN', 'label': 'PER', 'confidence': 0.95},
        {'text': 'bác Văn', 'label': 'PER', 'confidence': 0.92},
        {'text': 'trưởng khoa Lê Viết Văn', 'label': 'PER', 'confidence': 0.90},
    ]
    
    real_result_persons = processor.process_persons(real_test_entities)
    print("Input entities:")
    for entity in real_test_entities:
        print(f"  - '{entity['text']}'")
    print(f"After processing: {real_result_persons}")
    print()
    
    # Test cases cho fuzzy substring
    test_cases = [
        ("HANA", "Hệ thống thẩm mỹ HANNA"),  # Substring nhưng viết khác 1 tí
        ("VinMart", "Siêu thị VinMart Plus"), # Substring exact
        ("BigC", "BigC Thăng Long"),          # Substring exact
        ("Lotte", "Lotte Cinema"),            # Substring exact
        ("ABC", "XYZ Company"),               # Không liên quan
        ("Vincom", "Vincom Center"),          # Substring exact
        ("TGDD", "Thế Giới Di Động TGDD"),   # Substring exact
    ]
    
    print("=== Test Fuzzy Substring Matching ===")
    for short, long in test_cases:
        is_similar = processor.is_similar_organizations(short, long)
        partial_sim = processor.fuzzy_partial_similarity(short, long)
        full_sim = processor.fuzzy_similarity(short, long)
        
        print(f"'{short}' vs '{long}':")
        print(f"  Similar: {is_similar}")
        print(f"  Partial similarity: {partial_sim:.3f}")
        print(f"  Full similarity: {full_sim:.3f}")
        print()
    
    # Test address processing với real data
    print("=== Test Address Processing (Real Data) ===")
    test_address_entities = [
        {'text': '123 Nguyễn Văn A, Quận 1, TP.HCM (gần chợ Bến Thành)', 'label': 'ADDR', 'confidence': 0.95},
        {'text': '456 Lê Lợi, Q.3, TPHCM [tầng 2]', 'label': 'ADDR', 'confidence': 0.92},
        {'text': 'Bệnh viện Chợ Rẫy, 201B Nguyễn Chí Thanh, Quận 5, TP.HCM', 'label': 'ADDR', 'confidence': 0.90},
        {'text': 'Hà Nội', 'label': 'ADDR', 'confidence': 0.85},  # Không thuộc vùng cho phép
        {'text': '789 ABC, gần siêu thị', 'label': 'ADDR', 'confidence': 0.88},  # Quá ngắn sau khi clean
    ]
    
    result_addresses = processor.process_addresses(test_address_entities)
    print("Input address entities:")
    for entity in test_address_entities:
        print(f"  - '{entity['text']}'")
    print(f"After processing: {result_addresses}")
    print()
    
    # Test với real data
    test_ner_results = {
        'entities_by_type': {
            'ORG': [
                {'text': 'HANA', 'label': 'ORG', 'confidence': 0.95},
                {'text': 'Hệ thống thẩm mỹ HANNA', 'label': 'ORG', 'confidence': 0.92},
                {'text': 'HANA', 'label': 'ORG', 'confidence': 0.93},
                {'text': 'VinMart', 'label': 'ORG', 'confidence': 0.90},
                {'text': 'Siêu thị VinMart Plus', 'label': 'ORG', 'confidence': 0.88},
            ],
            'ADDR': [
                {'text': '123 Nguyễn Văn A, Quận 1, TP.HCM (gần chợ)', 'label': 'ADDR', 'confidence': 0.89},
                {'text': 'Thủ Đức, TP.HCM [khu vực trung tâm]', 'label': 'ADDR', 'confidence': 0.87},
                {'text': 'Hà Nội', 'label': 'ADDR', 'confidence': 0.85}
            ],
            'PER': [
                {'text': 'Bác sĩ Nguyễn Văn A', 'label': 'PER', 'confidence': 0.92},
                {'text': 'TS. Trần Thị B', 'label': 'PER', 'confidence': 0.88},
                {'text': 'Nguyễn Văn A', 'label': 'PER', 'confidence': 0.90},
                {'text': 'CKI Lê Minh C', 'label': 'PER', 'confidence': 0.85},
                {'text': 'Dr. John Smith', 'label': 'PER', 'confidence': 0.87}
            ]
        }
    }
    
    print("=== Test Real NER Data ===")
    result = quick_postprocess(test_ner_results)
    print(f"Organizations: {result['organization_names']}")
    print(f"Addresses: {result['addresses']}")
    print(f"Persons: {result['persons']}")
    
    print(f"\n=== Using RapidFuzz: {RAPIDFUZZ_AVAILABLE} ===")


if __name__ == "__main__":
    test_improved_fuzzy()