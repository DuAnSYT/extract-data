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
        # Regex patterns cho các địa chỉ cho phép
        self.allowed_locations = [
            r'(?i).*(?:tp\s*\.?\s*)?h[ồô]\s*ch[íi]\s*minh.*',
            r'(?i).*sài\s*g[òon].*',
            r'(?i).*b[àa]\s*r[ịi]a\s*v[ũu]ng\s*t[àa]u.*',
            r'(?i).*b[ìi]nh\s*d[ươuong].*',
            r'(?i).*hcm.*',
            r'(?i).*tphcm.*',
            r'(?i).*thành\s*phố\s*hồ\s*chí\s*minh.*',
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
    
    def process_organizations(self, org_entities: List[Dict], threshold: float = 0.8, partial_threshold: float = 0.9) -> str:
        """
        Xử lý danh sách organizations với improved fuzzy matching
        Trả về organization name tốt nhất
        
        Args:
            org_entities: List entities của ORG
            threshold: Ngưỡng cho full similarity  
            partial_threshold: Ngưỡng cho partial similarity
        """
        if not org_entities:
            return ""
        
        # Lấy tất cả text của organizations
        org_texts = [entity['text'].strip() for entity in org_entities]
        
        if len(org_texts) == 1:
            return org_texts[0]
        
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
        
        # Chọn organization có count cao nhất
        if grouped_orgs:
            best_org = max(grouped_orgs.items(), key=lambda x: x[1])[0]
            return best_org.strip()
        
        return ""
    
    def process_addresses(self, loc_entities: List[Dict]) -> List[str]:
        """
        Xử lý danh sách địa chỉ, chỉ giữ lại những địa chỉ thuộc vùng cho phép
        """
        if not loc_entities:
            return []
        
        allowed_addresses = []
        seen_addresses = set()
        
        for entity in loc_entities:
            address = entity['text'].strip()
            address_lower = address.lower()
            
            # Kiểm tra trùng lặp và location hợp lệ
            if address_lower not in seen_addresses and self.is_allowed_location(address):
                allowed_addresses.append(address)
                seen_addresses.add(address_lower)
        
        return allowed_addresses
    
    def postprocess(self, ner_results: Dict, threshold: float = 0.8, partial_threshold: float = 0.9) -> Dict:
        """
        Main postprocessing function với improved fuzzy matching
        
        Args:
            ner_results: Output từ VietnameseNERPredictor
            threshold: Ngưỡng cho full similarity (0.0-1.0)
            partial_threshold: Ngưỡng cho partial similarity (0.0-1.0)
            
        Returns:
            Dict với format: {
                'organization_name': str,
                'address': List[str]
            }
        """
        entities_by_type = ner_results.get('entities_by_type', {})
        
        # Xử lý organizations với improved logic
        org_entities = entities_by_type.get('ORG', [])
        organization_name = self.process_organizations(org_entities, threshold, partial_threshold)
        
        # Xử lý locations/addresses
        loc_entities = entities_by_type.get('LOC', [])
        addresses = self.process_addresses(loc_entities)
        
        # Có thể thêm xử lý cho các entity type khác nếu cần
        gpe_entities = entities_by_type.get('GPE', [])
        if gpe_entities:
            gpe_addresses = self.process_addresses(gpe_entities)
            # Merge và loại bỏ trùng lặp
            all_addresses = addresses + gpe_addresses
            addresses = list(dict.fromkeys(all_addresses))
        
        return {
            'organization_name': organization_name,
            'address': addresses
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
            'address': List[str]
        }
    """
    processor = get_processor()
    return processor.postprocess(ner_results, threshold, partial_threshold)


def test_improved_fuzzy():
    """🧪 Test improved fuzzy substring matching"""
    
    processor = NERPostProcessor()
    
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
            'LOC': [
                {'text': 'TP Hồ Chí Minh', 'label': 'LOC', 'confidence': 0.89},
                {'text': 'Bình Dương', 'label': 'LOC', 'confidence': 0.87},
                {'text': 'Hà Nội', 'label': 'LOC', 'confidence': 0.85}
            ]
        }
    }
    
    print("=== Test Real NER Data ===")
    result = quick_postprocess(test_ner_results)
    print(f"Organization: '{result['organization_name']}'")
    print(f"Addresses: {result['address']}")
    
    print(f"\n=== Using RapidFuzz: {RAPIDFUZZ_AVAILABLE} ===")


if __name__ == "__main__":
    test_improved_fuzzy()