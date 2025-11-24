#!/usr/bin/env python3
"""
Clean Excel file content using the clean_text function.
Processes all text cells in all sheets and outputs a cleaned version.

Usage:
    python clean_text.py
"""

import unicodedata
import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd


def is_emoji_or_symbol(char):
    """
    Check if a character is an emoji, symbol, or icon that should be removed.
    """
    code = ord(char)
    
    # Common emoji and symbol ranges
    emoji_ranges = [
        # Miscellaneous Symbols (includes phone ☎, etc.)
        (0x2600, 0x26FF),   # ☀☁☂...☎...⚡⚽ etc.
        
        # Dingbats
        (0x2700, 0x27BF),   # ✀✁✂...✉...➡ etc.
        
        # Emoticons  
        (0x1F600, 0x1F64F), # 😀😁😂...😍😎 etc.
        
        # Miscellaneous Symbols and Pictographs
        (0x1F300, 0x1F5FF), # 🌀🌁🌂...📱📞 etc.
        
        # Transport and Map Symbols
        (0x1F680, 0x1F6FF), # 🚀🚁🚂...🛸 etc.
        
        # Supplemental Symbols and Pictographs
        (0x1F900, 0x1F9FF), # 🤀🤁🤂...🦸🦹 etc.
        
        # Geometric Shapes
        (0x25A0, 0x25FF),   # ■□▲△ etc.
        
        # Miscellaneous Technical
        (0x2300, 0x23FF),   # ⌀⌁⌂...⏰⏱ etc.
        
        # Arrows
        (0x2190, 0x21FF),   # ←↑→↓ etc.
        
        # Mathematical Operators (some symbols)
        (0x2200, 0x22FF),   # ∀∁∂...≤≥ etc. (but keep basic math)
        
        # Box Drawing (might be from tables)
        (0x2500, 0x257F),   # ─━│┃ etc.
        
        # Block Elements
        (0x2580, 0x259F),   # ▀▁▂ etc.
    ]
    
    # Check if character falls in any emoji range
    for start, end in emoji_ranges:
        if start <= code <= end:
            return True
    
    # Additional specific symbols commonly found in text
    specific_symbols = {
        0x260E,  # ☎ (Black Telephone)
        0x2709,  # ✉ (Envelope)
        0x2764,  # ❤ (Heavy Black Heart)
        0x2665,  # ♥ (Black Heart Suit)
        0x2666,  # ♦ (Black Diamond Suit)
        0x2663,  # ♣ (Black Club Suit)
        0x2660,  # ♠ (Black Spade Suit)
        0x2605,  # ★ (Black Star)
        0x2606,  # ☆ (White Star)
        0x25CF,  # ● (Black Circle)
        0x25CB,  # ○ (White Circle)
        0x25A0,  # ■ (Black Square)
        0x25A1,  # □ (White Square)
    }
    
    return code in specific_symbols


def clean_text(text):
    """
    Remove or replace characters that cause Excel errors.
    Handles multiple styles of Unicode mathematical alphabetic symbols, icons, and emojis.
    Also removes separator lines/strings like '---', '====', '___'.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'[-=_]{3,}', '', text)
    # Dictionary of Unicode mathematical alphabetic symbols and their replacements
    replacements = {
        range(0x1D400, 0x1D433): lambda c: chr(ord(c) - 0x1D400 + ord('A')),  # Bold A-Z and a-z
        range(0x1D7CE, 0x1D7FF): lambda c: chr(ord(c) - 0x1D7CE + ord('0')),  # Bold numbers
        range(0x1D434, 0x1D467): lambda c: chr(ord(c) - 0x1D434 + ord('A')),  # Italic A-Z and a-z
        range(0x1D468, 0x1D49B): lambda c: chr(ord(c) - 0x1D468 + ord('A')),  # Bold Italic A-Z and a-z
        range(0x1D49C, 0x1D4CF): lambda c: chr(ord(c) - 0x1D49C + ord('A')),  # Script A-Z and a-z
        range(0x1D4D0, 0x1D503): lambda c: chr(ord(c) - 0x1D4D0 + ord('A')),  # Bold Script A-Z and a-z
        range(0x1D504, 0x1D537): lambda c: chr(ord(c) - 0x1D504 + ord('A')),  # Fraktur A-Z and a-z
        range(0x1D538, 0x1D56B): lambda c: chr(ord(c) - 0x1D538 + ord('A')),  # Double-struck A-Z and a-z
        range(0x1D56C, 0x1D59F): lambda c: chr(ord(c) - 0x1D56C + ord('A')),  # Bold Fraktur A-Z and a-z
        range(0x1D5A0, 0x1D5D3): lambda c: chr(ord(c) - 0x1D5A0 + ord('A')),  # Sans-serif A-Z and a-z
        range(0x1D5D4, 0x1D607): lambda c: chr(ord(c) - 0x1D5D4 + ord('A')),  # Sans-serif Bold A-Z and a-z
        range(0x1D7EC, 0x1D7F6): lambda c: chr(ord(c) - 0x1D7EC + ord('0')),  # Sans-serif Bold numbers
        range(0x1D608, 0x1D63B): lambda c: chr(ord(c) - 0x1D608 + ord('A')),  # Sans-serif Italic A-Z and a-z
    }
    
    try:
        normalized = unicodedata.normalize('NFC', text)
        result = ""
        for char in normalized:
            code = ord(char)

            # Skip control characters except tab, LF, CR
            if code < 32 and code not in (9, 10, 13):
                continue

            # Remove emojis and symbols (common ranges)
            if is_emoji_or_symbol(char):
                continue

            # Try replacing mathematical symbols
            replaced = False
            for char_range, replacement_func in replacements.items():
                if code in char_range:
                    result += replacement_func(char)
                    replaced = True
                    break

            # Keep the character if it wasn't replaced and is in BMP
            if not replaced and code < 65536:
                result += char
        return result
    except Exception:
        return text


def remove_crawl_artifacts(text):
    """
    Remove HTML crawling artifacts and unwanted elements.
    """
    if not isinstance(text, str):
        return text
    
    # Patterns to remove crawling artifacts
    patterns_to_remove = [
        # Image placeholders - các dạng [IMAGE-X], [IMG-X], [image-X]
        r'\[IMAGE-\d+\]',
        r'\[IMG-\d+\]', 
        r'\[image-\d+\]',
        r'\[IMAGE_\d+\]',
        r'\[IMG_\d+\]',
        r'\[image_\d+\]',
        
        # Video placeholders
        r'\[VIDEO-\d+\]',
        r'\[VID-\d+\]',
        r'\[video-\d+\]',
        
        # HTML tags that might remain
        r'<[^>]+>',
        
        # Common crawling artifacts
        r'\[AD\]',
        r'\[ADVERTISEMENT\]',
        r'\[SPONSORED\]',
        r'\[PROMO\]',
        r'\[Banner\]',
        r'\[banner\]',
        r'\[BANNER\]',
        
        # Social media artifacts
        r'\[TWEET\]',
        r'\[FACEBOOK\]',
        r'\[INSTAGRAM\]',
        r'\[SHARE\]',
        r'\[LIKE\]',
        r'\[FOLLOW\]',
        
        # Navigation artifacts
        r'\[MENU\]',
        r'\[NAV\]',
        r'\[NAVIGATION\]',
        r'\[BREADCRUMB\]',
        r'\[HOME\]',
        r'\[BACK\]',
        r'\[NEXT\]',
        r'\[PREV\]',
        r'\[Previous\]',
        r'\[Continue\]',
        
        # Comment/interaction artifacts
        r'\[COMMENT\]',
        r'\[REPLY\]',
        r'\[COMMENTS\]',
        r'\[read more\]',
        r'\[Read More\]',
        r'\[read_more\]',
        r'\[show more\]',
        r'\[Show More\]',
        r'\[xem thêm\]',
        r'\[Xem thêm\]',
        r'\[XEM THÊM\]',
        
        # Chart/table artifacts
        r'\[CHART-\d+\]',
        r'\[TABLE-\d+\]',
        r'\[GRAPH-\d+\]',
        r'\[FIGURE-\d+\]',
        r'\[FIG-\d+\]',
        
        # Loading/placeholder text
        r'\[LOADING\]',
        r'\[Loading\]',
        r'\[loading\]',
        r'\[PLACEHOLDER\]',
        r'\[placeholder\]',
        
        # Common Vietnamese crawling artifacts
        r'\[Ảnh\]',
        r'\[ảnh\]',
        r'\[ẢNH\]',
        r'\[Hình\]',
        r'\[hình\]',
        r'\[HÌNH\]',
        r'\[Video\]',
        r'\[video\]',
        r'\[VIDEO\]',
        r'\[Quảng cáo\]',
        r'\[quảng cáo\]',
        r'\[QUẢNG CÁO\]',
        
        # Generic numbered placeholders
        r'\[ITEM-\d+\]',
        r'\[ELEMENT-\d+\]',
        r'\[BLOCK-\d+\]',
        r'\[DIV-\d+\]',
        r'\[SECTION-\d+\]',
        
        # Empty brackets or with spaces
        r'\[\s*\]',
        r'\[\s+\]',
        
        # Double spaces and excessive whitespace
        r'\s+',  # Replace multiple spaces with single space
    ]
    
    try:
        cleaned_text = text
        
        # Apply all removal patterns except the last one (whitespace)
        for pattern in patterns_to_remove[:-1]:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Handle whitespace normalization separately
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Remove empty parentheses that might be left behind
        cleaned_text = re.sub(r'\(\s*\)', '', cleaned_text)
        cleaned_text = re.sub(r'\[\s*\]', '', cleaned_text)
        cleaned_text = re.sub(r'\{\s*\}', '', cleaned_text)
        
        # Final cleanup - remove multiple spaces again
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    except Exception:
        return text


def clean_excel_file(input_file: Path, output_file: Path) -> Dict:
    """Clean an Excel file and return statistics."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"📂 Reading Excel file: {input_file}")
    
    # Read all sheets
    excel_data = pd.read_excel(input_file, sheet_name=None, engine="openpyxl")
    
    print(f"📋 Processing {len(excel_data)} sheet(s)")
    
    # Clean each sheet
    cleaned_data = {}
    total_cells = 0
    changed_cells = 0
    
    for sheet_name, df in excel_data.items():
        print(f"  🧹 Cleaning sheet: {sheet_name}")
        
        if df.empty:
            cleaned_data[sheet_name] = df
            continue
        
        # Apply both cleaning functions to all columns
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            # First apply clean_text, then remove crawl artifacts
            df_cleaned[col] = df_cleaned[col].apply(clean_text)
            df_cleaned[col] = df_cleaned[col].apply(remove_crawl_artifacts)
        
        cleaned_data[sheet_name] = df_cleaned
        
        # Calculate changes
        sheet_cells = df.size
        sheet_changes = 0
        for col in df.columns:
            original_series = df[col].astype(str)
            cleaned_series = df_cleaned[col].astype(str)
            sheet_changes += (original_series != cleaned_series).sum()
        
        total_cells += sheet_cells
        changed_cells += sheet_changes
        
        change_pct = round((sheet_changes / sheet_cells) * 100, 2) if sheet_cells > 0 else 0
        print(f"    📊 Changed {sheet_changes}/{sheet_cells} cells ({change_pct}%)")
    
    # Save cleaned Excel file
    print(f"💾 Saving cleaned file: {output_file}")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in cleaned_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Return statistics - CONVERT TO NATIVE PYTHON TYPES
    overall_change_pct = round((changed_cells / total_cells) * 100, 2) if total_cells > 0 else 0
    
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "sheets_processed": int(len(cleaned_data)),  # Convert to int
        "total_cells": int(total_cells),             # Convert to int
        "changed_cells": int(changed_cells),         # Convert to int
        "change_percentage": float(overall_change_pct)  # Convert to float
    }


def main():
    # Fixed file names
    INPUT_FILE = "data.xlsx"
    OUTPUT_FILE = "cleaned_data.xlsx"
    STATS_FILE = "cleaning_stats.json"
    
    input_file = Path(INPUT_FILE)
    output_file = Path(OUTPUT_FILE)
    stats_file = Path(STATS_FILE)
    
    print(f"🔧 Processing:")
    print(f"  - Input: {input_file}")
    print(f"  - Output: {output_file}")
    print()
    
    try:
        # Clean the Excel file
        stats = clean_excel_file(input_file, output_file)
        
        # Print summary
        print("\n✅ Cleaning completed successfully!")
        print(f"📊 Statistics:")
        print(f"  - Sheets processed: {stats['sheets_processed']}")
        print(f"  - Total cells: {stats['total_cells']:,}")
        print(f"  - Changed cells: {stats['changed_cells']:,}")
        print(f"  - Change percentage: {stats['change_percentage']}%")
        
        # Save statistics
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📈 Statistics saved to: {stats_file}")
        
        print(f"\n📄 Output files:")
        print(f"  - Cleaned Excel: {output_file}")
        print(f"  - Statistics JSON: {stats_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())