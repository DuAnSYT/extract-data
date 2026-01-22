import re

def extract_phone(text):
    """
    Extract and normalize Vietnamese phone numbers from text.
    
    Args:
        text (str): Input text to search
        
    Returns:
        list: Normalized phone numbers starting with '0'
    """
    phone_regex = r'''(?:
                        (?:(?:(?:\+84|0084)|0)[\s.-]?(?:[13456789][\s.-]?(?:\d[\s.-]?){7}\d))
                        |
                        (?:(?:(?:02)(?:\d[\s.-]?){8,9}))
                        |
                        \((?:0\d{2,3})\)[\s.-]?(?:\d[\s.-]?){7}
                        |
                        (?:(?:1800|1900)[\s.-]?(?:\d[\s.-]?){4,6})
                        |
                        \((?:\+84|0084)\)[\s.-]?(?:\d[\s.-]?){8}\d
                        )'''

    found_numbers = re.findall(phone_regex, text, flags=re.VERBOSE)

    phone_lists=[]
    for number in found_numbers:
        clean_num=re.sub(r'[^\d]', '', number)
        
        if clean_num.startswith('84'):
            clean_num = '0' + clean_num[2:]
        if clean_num.startswith('0084'):
            clean_num = '0' + clean_num[4:]
        
        if clean_num:
            phone_lists.append(clean_num)
    return list(set(phone_lists))

def extract_licenses_and_certificates(text):
    """
    Extract and normalize Vietnamese operating licenses and medical certificates.
    Args:
        text (str): Input text to search
        
    Returns:
        dict: Dictionary with separate lists for licenses and certificates
    """
    operating_licenses = []
    medical_certificates = []
    
    # Danh sách các cơ quan được chấp nhận
    valid_agencies = ['HCM', 'SYT', 'BYT']
    
    # Pattern cho Giấy phép hoạt động
    # Ví dụ: 09384/HCM, 12345/SYT, 12345/BYT, 12345/HCM-GPHĐ
    license_pattern = r'[Gg]iấy\s*phép\s*hoạt\s*động[:\s]*(\d{2,7})\s*/\s*([A-Za-z]+)(?:-[A-Za-zĐ]+)?(?:/[A-Za-z0-9]+)?'
    
    # Pattern cho Chứng chỉ hành nghề
    # Ví dụ: 13949/SYT, 008933/SYT, 12345/HCM-CCHN
    certificate_pattern = r'[Cc]hứng\s*chỉ\s*hành\s*nghề[:\s]*(\d{2,7})\s*/\s*([A-Za-z]+)(?:-[A-Za-zĐ]+)?(?:/[A-Za-z0-9]+)?'
    
    # Tìm Giấy phép hoạt động
    for match in re.finditer(license_pattern, text):
        number = match.group(1).strip()
        agency = match.group(2).strip().upper()
        
        # Chỉ lấy nếu thuộc HCM, SYT, BYT
        if agency in valid_agencies:
            license_num = f"{number}/{agency}"
            operating_licenses.append(license_num)
    
    # Tìm Chứng chỉ hành nghề
    for match in re.finditer(certificate_pattern, text):
        number = match.group(1).strip()
        agency = match.group(2).strip().upper()
        
        # Chỉ lấy nếu thuộc HCM, SYT, BYT
        if agency in valid_agencies:
            cert_num = f"{number}/{agency}"
            medical_certificates.append(cert_num)
    
    return {
        'operating_licenses': list(set(operating_licenses)),
        'medical_certificates': list(set(medical_certificates))
    }

# Keep separate functions for backward compatibility
def operating_license(text):
    """Extract operating licenses only"""
    result = extract_licenses_and_certificates(text)
    return result['operating_licenses']

def medical_certificate(text):
    """Extract medical certificates only"""
    result = extract_licenses_and_certificates(text)
    return result['medical_certificates']