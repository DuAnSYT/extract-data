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
    # Combined pattern for both types
    combined_regex = r'\b\d{2,6}\s*/\s*(HCM|SYT|BYT)(?:-(GPHĐ|CCHN))?\b'
    
    operating_licenses = []
    medical_certificates = []
    
    for match in re.finditer(combined_regex, text):
        full_match = match.group(0)
        number_part = match.group(0).split('/')[0].strip()
        agency_part = match.group(1)  # HCM, SYT, BYT
        suffix_part = match.group(2)  # GPHĐ or CCHN
        
        # Normalize spacing
        base_format = f"{number_part}/{agency_part}"
        
        if suffix_part:
            # Already has suffix, determine type
            if suffix_part == 'GPHĐ':
                operating_licenses.append(f"{base_format}-GPHĐ")
            elif suffix_part == 'CCHN' and agency_part == 'HCM':
                medical_certificates.append(f"{base_format}-CCHN")
        else:
            # No suffix, check context to determine type
            context = text[max(0, match.start()-50):match.end()+50].lower()
            
            # Keywords for operating license
            license_keywords = ['giấy phép hoạt động', 'gphđ', 'phép hoạt động', 'hoạt động']
            
            # Keywords for medical certificate (only for HCM)
            medical_keywords = ['chứng chỉ hành nghề', 'cchn', 'hành nghề', 'chứng chỉ']
            
            # Determine type based on context
            has_license_context = any(keyword in context for keyword in license_keywords)
            has_medical_context = any(keyword in context for keyword in medical_keywords)
            
            if has_license_context and not has_medical_context:
                operating_licenses.append(f"{base_format}-GPHĐ")
            elif has_medical_context and not has_license_context and agency_part == 'HCM':
                medical_certificates.append(f"{base_format}-CCHN")
            elif not has_license_context and not has_medical_context:
                # Default behavior: operating license for all agencies, medical certificate only for HCM
                if agency_part in ['SYT', 'BYT']:
                    operating_licenses.append(f"{base_format}-GPHĐ")
                elif agency_part == 'HCM':
                    # For HCM without context, could be either - add both possibilities
                    operating_licenses.append(f"{base_format}-GPHĐ")
    
    return {
        'operating_licenses': list(set(operating_licenses)),
        'medical_certificates': list(set(medical_certificates))
    }

# Keep separate functions for backward compatibility
def extract_operating_license(text):
    """Extract operating licenses only"""
    result = extract_licenses_and_certificates(text)
    return result['operating_licenses']

def extract_medical_certificate(text):
    """Extract medical certificates only"""
    result = extract_licenses_and_certificates(text)
    return result['medical_certificates']