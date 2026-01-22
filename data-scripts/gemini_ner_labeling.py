import pandas as pd
import re
from underthesea import sent_tokenize
from typing import List, Dict, Tuple
from google import genai
import os
from tqdm import tqdm
import time
import argparse
from transformers import AutoTokenizer
import logging
from datetime import datetime
import traceback
import unicodedata
import html
from ftfy import fix_text

def setting_logger():
    global logger
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ner_labeling.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("NER")
    logger.setLevel(logging.INFO)

# Fancy to normal mapping (có thể mở rộng thêm nếu cần)
fancy_char_map = str.maketrans({
    'ᴬ': 'A', 'ᴭ': 'a', 'ᴮ': 'B', 'ᴯ': 'b', 'ᴰ': 'D', 'ᴱ': 'E',
    'ᴲ': 'e', 'ᴳ': 'G', 'ᴴ': 'H', 'ᴵ': 'I', 'ᴶ': 'J', 'ᴷ': 'K',
    'ᴸ': 'L', 'ᴹ': 'M', 'ᴺ': 'N', 'ᴼ': 'O', 'ᴾ': 'P', 'ᴿ': 'R',
    'ᵀ': 'T', 'ᵁ': 'U', 'ⱽ': 'V', 'ᵂ': 'W',
    'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c', 'ᵈ': 'd', 'ᵉ': 'e', 'ᶠ': 'f',
    'ᵍ': 'g', 'ʰ': 'h', 'ᶦ': 'i', 'ʲ': 'j', 'ᵏ': 'k', 'ˡ': 'l',
    'ᵐ': 'm', 'ⁿ': 'n', 'ᵒ': 'o', 'ᵖ': 'p', 'ʳ': 'r', 'ˢ': 's',
    'ᵗ': 't', 'ᵘ': 'u', 'ᵛ': 'v', 'ʷ': 'w', 'ˣ': 'x', 'ʸ': 'y', 'ᶻ': 'z',
    'ᴜ': 'u', 'ᴎ': 'n', 'ᴇ': 'e', 'ᴏ': 'o', 'ᴅ': 'd', 'ᴛ': 't', 'ᴍ': 'm',
    'ɴ': 'n', 'ʀ': 'r', 'ʏ': 'y', 'ʜ': 'h', 'ɪ': 'i', 'ᴄ': 'c', 'ᴋ': 'k',
    'ᴀ': 'a',
    # Math bold/italic (một vài ví dụ)
    '𝓐': 'A', '𝓑': 'B', '𝓒': 'C', '𝓓': 'D', '𝓔': 'E', '𝓕': 'F',
    '𝓖': 'G', '𝓗': 'H', '𝓘': 'I', '𝓙': 'J', '𝓚': 'K', '𝓛': 'L',
    '𝓜': 'M', '𝓝': 'N', '𝓞': 'O', '𝓟': 'P', '𝓠': 'Q', '𝓡': 'R',
    '𝓢': 'S', '𝓣': 'T', '𝓤': 'U', '𝓥': 'V', '𝓦': 'W', '𝓧': 'X',
    '𝓨': 'Y', '𝓩': 'Z',
    '𝓪': 'a', '𝓫': 'b', '𝓬': 'c', '𝓭': 'd', '𝓮': 'e', '𝓯': 'f',
    '𝓰': 'g', '𝓱': 'h', '𝓲': 'i', '𝓳': 'j', '𝓴': 'k', '𝓵': 'l',
    '𝓶': 'm', '𝓷': 'n', '𝓸': 'o', '𝓹': 'p', '𝓺': 'q', '𝓻': 'r',
    '𝓼': 's', '𝓽': 't', '𝓾': 'u', '𝓿': 'v', '𝔀': 'w', '𝔁': 'x',
    '𝔂': 'y', '𝔃': 'z',
})

def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    
    # Sửa lỗi Unicode thường gặp (ftfy)
    text = fix_text(str(text))

    # Normalize Unicode tổ hợp (dấu tiếng Việt)
    text = unicodedata.normalize('NFC', text)

    # Chuyển ký tự fancy → ký tự Latin thông thường
    text = text.translate(fancy_char_map)

    # HTML entity decode
    text = html.unescape(text)

    # Xóa link và tag HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Xóa các ký tự điều khiển (invisible, null,...)
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F-\x9F]', '', text)

    # Xử lý từng dòng
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        clean_line = ''.join(
            c if (unicodedata.category(c)[0] in 'LNZ' or
                  c in '.,!?;:()[]{}"\'`-_/+=\\~@#$%^&*|' or
                  c.isspace())
            else ' '
            for c in line
        )
        clean_line = re.sub(r'[ \t]+', ' ', clean_line).strip()
        if clean_line:
            if not re.search(r'[.!?…:;]$', clean_line):
                clean_line += '.'
            processed_lines.append(clean_line)

    # Ghép lại thành đoạn văn
    text = ' '.join(processed_lines)
    return text.strip()

# Cấu hình Rate Limiting
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Seconds between requests
last_request_time = 0

# Hàm để đảm bảo tốc độ request không vượt quá giới hạn
def respect_rate_limit():
    global last_request_time
    current_time = time.time()
    elapsed = current_time - last_request_time
    
    if elapsed < REQUEST_INTERVAL:
        sleep_time = REQUEST_INTERVAL - elapsed
        logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    last_request_time = time.time()

class TextChunker:
    def __init__(self, model_name: str = "vinai/vit5-base", max_length: int = 900):
        """
        Initialize text chunker for Vietnamese text
        
        Args:
            model_name: ViT5 model name
            max_length: Maximum length in tokens (slightly less than 1024 to allow for tags)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in Vietnamese text."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def _find_safe_split_point(self, text: str, max_tokens: int) -> int:
        token_count = self._count_tokens(text)  # đếm tổng số token
        if token_count <= max_tokens:
            return len(text)  # không cần chia

        approx_char_pos = int(len(text) * (max_tokens / token_count))  # ước lượng vị trí nên chia theo độ dài
        sentences = sent_tokenize(text)  # tách câu

        current_pos = 0  # dùng để đánh dấu vị trí đang duyệt trong text gốc
        for sentence in sentences:
            # tìm câu hiện tại trong text gốc (tính từ current_pos)
            match = re.search(re.escape(sentence), text[current_pos:])
            if not match:
                continue  # nếu không tìm thấy thì bỏ qua (hiếm khi xảy ra)

            next_pos = current_pos + match.end()  # vị trí kết thúc câu đó trong text gốc

            if next_pos > approx_char_pos:
                # kiểm tra nếu đang trong đoạn có tag chưa đóng
                text_so_far = text[:current_pos]
                if "<" in text_so_far and ">" not in text_so_far:
                    return max(0, current_pos - len(sentence))  # tránh cắt giữa tag

                # tránh cắt giữa từ
                if current_pos < len(text) and not text[current_pos].isspace():
                    last_space = text[:current_pos].rfind(" ")
                    if last_space != -1:
                        return last_space

                return current_pos

            current_pos = next_pos  # cập nhật vị trí cho vòng lặp tiếp theo

        # fallback: nếu không tìm được câu hợp lý
        last_space = text[:approx_char_pos].rfind(" ")
        return last_space if last_space != -1 else approx_char_pos
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that respect the max_length constraint."""
        chunks = []
        remaining_text = text
        
        while remaining_text:
            # Get a safe split point
            split_point = self._find_safe_split_point(remaining_text, self.max_length)
            
            chunk = remaining_text[:split_point]
            chunks.append(chunk)
            remaining_text = remaining_text[split_point:].strip()
        
        return chunks


class GeminiNERTagger:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.0-flash-lite", max_requests_per_key: int = 1400):
        """
        Initialize Gemini API for NER tagging with multiple API keys for rotation
        
        Args:
            api_keys: List of Google API keys for Gemini
            model_name: Gemini model to use
            max_requests_per_key: Maximum number of requests per API key before rotation
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        self.request_count = 0
        self.max_requests_per_key = max_requests_per_key
        logger.info(f"Initialized Gemini NER tagger with {len(api_keys)} API keys")
        
    def _rotate_api_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        self.request_count = 0
        logger.info(f"Rotated to API key #{self.current_key_index+1}")
        
    def generate_ner_prompt(self, text: str, entity_types: List[str]) -> str:
        """
        Generate a prompt for NER tagging in Vietnamese
        
        Args:
            text: Text to annotate
            entity_types: List of entity types to identify
        
        Returns:
            Prompt for Gemini API
        """
        entity_types_str = ", ".join(entity_types)
        
        prompt = f"""Mục tiêu: Gắn nhãn thực thể (NER) cho bài quảng cáo y tế, với các loại thực thể cần nhận diện là:
1. PER (Person): Tên cá nhân (người cung cấp/bán sản phẩm).
2. ORG (Organization): Tên tổ chức/đơn vị/shop cung cấp/bán sản phẩm.
3. ADDR (Address): Địa chỉ của tổ chức/cá nhân/shop.
4. PHONE (Phone number): Số điện thoại liên lạc.

Quy tắc đánh dấu:
- Bao quanh mỗi thực thể với thẻ <LABEL></LABEL>, trong đó LABEL là loại thực thể (PER, ORG, ADDR, PHONE).
- Đánh dấu toàn bộ cụm thực thể, không chỉ một phần.
- Không thay đổi văn bản gốc, chỉ thêm thẻ vào.
- Các tên tổ chức (như "Công ty ABC", "Shop XYZ") được gắn nhãn là ORG.
- Các tên cá nhân (như "Nguyễn Văn A", "Mai") được gắn nhãn là PER.Tên cá nhân sẽ bao gồm luôn chức danh, học hàm học vị nếu có
- Các địa chỉ (như "123 Nguyễn Thị Minh Khai, TP.HCM”) được gắn nhãn là ADDR.
- Các số điện thoại (như "0901234567", "02838383838") được gắn nhãn là PHONE.
- Địa chỉ chỉ lấy cụm địa chỉ đầy đủ.
- CHỈ trả về văn bản đã được đánh dấu thực thể, không thêm giải thích.

Ví dụ: 
- Văn bản: "Nguyễn Văn A làm việc tại Công ty FPT ở Hà Nội." 
- Gán nhãn: "<PER>Nguyễn Văn A</PER> làm việc tại <ORG>Công ty FPT</ORG> ở Hà Nội."

Lưu ý: 
- Khi tên Cá nhân bao gồm cả chức vụ/nghề nghiệp thì cụm tên cá nhân sẽ bao gồm cả chức vụ/nghề nghiệp.
Ví dụ:  <PER>Ca sĩ THU THỦY</PER> -> không chỉ gán mỗi “THU THỦY”.
- Các từ viết tắt của tên tổ chức sẽ không gán nhãn cho nó.
Ví dụ <ORG>Thế giới skinfood<\ORG> => Không gán tag ORG cho cụm từ TGSF nếu có trong văn bản.

Văn bản cần đánh dấu:
"{text}"
"""
        return prompt
    
    def tag_entities(self, text: str, entity_types: List[str], max_retries: int = 1) -> str:
        """
        Tag entities in text using Gemini API
        
        Args:
            text: Text to tag
            entity_types: List of entity types to identify
            max_retries: Maximum number of retry attempts
        
        Returns:
            Text with entity tags
        """
        prompt = self.generate_ner_prompt(text, entity_types)
        
        for attempt in range(max_retries):
            try:
                respect_rate_limit()  # Add rate limiting before API call
                
                # Check if we need to rotate API key
                if self.request_count >= self.max_requests_per_key:
                    if self.current_key_index == len(self.api_keys) - 1:
                        logger.warning("WARNING: All API keys have reached their quota limit!")
                    self._rotate_api_key()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt]
                )
                
                # Increment request counter
                self.request_count += 1
                logger.debug(f"Request count for current key: {self.request_count}/{self.max_requests_per_key}")

                tagged_text = response.text.strip()
                if tagged_text.startswith('"') and tagged_text.endswith('"'):
                    tagged_text = tagged_text[1:-1]
                
                # Simple validation to check if the response contains tags
                if "<" in tagged_text and ">" in tagged_text:
                    return tagged_text
                else:
                    logger.warning(f"No tags found in the response, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"Error calling Gemini API: {e}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
        
        # If all retries fail, return the original text
        logger.error("Failed to tag entities, returning original text")
        return text

# Tạo tên file cho checkpoint dựa trên timestamp
def get_checkpoint_filename(base_filename, checkpoint_num):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, ext = os.path.splitext(base_filename)
    return f"{filename}_checkpoint_{checkpoint_num}_{timestamp}{ext}"

def process_excel_for_ner(
    input_file: str,
    output_file: str,
    api_keys: List[str],
    content_column: str,
    entity_types: List[str],
    model_name: str = "VietAI/vit5-base",
    max_length: int = 900,
    batch_size: int = 10,
    max_requests_per_key: int = 1400
) -> None:
    """
    Process Excel file for NER tagging
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file
        api_keys: List of Google API keys for Gemini
        content_column: Column name containing text to tag
        entity_types: List of entity types to identify
        model_name: ViT5 model name for tokenization
        max_length: Maximum sequence length for chunking
        batch_size: Number of rows to process at once for saving progress
        max_requests_per_key: Maximum number of requests per API key before rotation
    """
    start_time = time.time()
    
    # Initialize chunker and tagger
    chunker = TextChunker(model_name=model_name, max_length=max_length)
    tagger = GeminiNERTagger(api_keys=api_keys, max_requests_per_key=max_requests_per_key)

    setting_logger()
    
    # Load the Excel file
    df = pd.read_excel(input_file)
    
    # Validate content column exists
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found in the Excel file")
    
    # Create output dataframe with columns for original and tagged content
    result_data = []
    
    # Check for existing data
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            if len(existing_df) > 0:
                logger.info(f"Found existing output file with {len(existing_df)} entries. Will append new results.")
                result_data = existing_df.to_dict('records')
                
                # Find which rows were already processed
                processed_rows = set()
                for row in result_data:
                    if 'original_index' in row:
                        processed_rows.add(row['original_index'])
                
                logger.info(f"Found {len(processed_rows)} already processed rows.")
        except Exception as e:
            logger.warning(f"Error reading existing output file: {e}. Starting fresh.")
    
    # Track checkpoints
    checkpoint_count = 0
    total_rows = len(df)
    
    # Process each row
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows"):
        # Skip if this row was already processed
        if result_data and any(r.get('original_index') == index for r in result_data):
            logger.debug(f"Skipping already processed row {index}")
            continue
            
        text = str(row[content_column])
        
        # Skip empty cells
        if not text or text.lower() == "nan":
            continue
        
        # Preprocess the text before chunking
        preprocessed_text = preprocess_text(text)
        
        logger.info(f"Processing row {index+1}/{total_rows}")
        
        # Check if text needs chunking
        chunks = chunker.chunk_text(preprocessed_text)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Call Gemini API to tag entities
            logger.info(f"Tagging chunk {i+1}/{len(chunks)} for row {index+1}")
            tagged_text = tagger.tag_entities(chunk, entity_types)
            
            # Create a new row for the result dataframe
            result_row = {
                "original_index": index,
                "chunk_index": i,
                "original_content": text,
                "preprocessed_content": preprocessed_text if i == 0 else "",
                "content": chunk,
                "tagged_content": tagged_text
            }
            
            # Add other columns from the original dataframe
            for col in df.columns:
                if col != content_column:
                    result_row[f"original_{col}"] = row[col]
            
            result_data.append(result_row)
        
        # Save intermediate results every batch_size rows
        if (index + 1) % batch_size == 0:
            checkpoint_count += 1
            temp_df = pd.DataFrame(result_data)
            
            # Save checkpoint
            checkpoint_file = get_checkpoint_filename(output_file, checkpoint_count)
            temp_df.to_excel(checkpoint_file, index=False)
            temp_df.to_excel(output_file, index=False)
            
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (index + 1)
            remaining_rows = total_rows - (index + 1)
            estimated_time = remaining_rows * avg_time_per_row
            
            logger.info(f"Checkpoint {checkpoint_count} saved to {checkpoint_file}")
            logger.info(f"Processed {index+1}/{total_rows} rows ({(index+1)/total_rows*100:.1f}%)")
            logger.info(f"Average time per row: {avg_time_per_row:.2f} seconds")
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Create final result dataframe and save
    result_df = pd.DataFrame(result_data)
    result_df.to_excel(output_file, index=False)
    
    # Remove temporary file if it exists
    if os.path.exists(f"{output_file}.tmp"):
        os.remove(f"{output_file}.tmp")
    
    logger.info(f"Processing complete. Tagged data saved to {output_file}")
    logger.info(f"Original rows: {len(df)}, Result rows (after chunking): {len(result_df)}")


def main():
    """Main function to execute the NER tagging process with predefined parameters"""
    try:
        # Define your parameters here
        input_file = "google_crawl_26_04-part2.xlsx"
        output_file = "Gemini_google_crawl_26_04-part2.xlsx"
        
        # List of API keys to rotate through when quota is reached
        api_keys = [
            # "YOUR_FIRST_API_KEY",
            # "YOUR_SECOND_API_KEY",
            # "YOUR_THIRD_API_KEY",
        ]
        
        content_column = "content"  # Column containing the text to be tagged
        entity_types = ["PER cho cá nhân", "ORG cho tổ chức", "LOC cho địa chỉ", "PHONE cho sdt"]  # Entity types to identify
        
        setting_logger()
        
        logger.info(f"Starting NER tagging process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Entity types: {', '.join(entity_types)}")
        logger.info(f"Using {len(api_keys)} API keys with max {1400} requests per key")
        
        # Process the file
        process_excel_for_ner(
            input_file=input_file,
            output_file=output_file,
            api_keys=api_keys,
            content_column=content_column,
            entity_types=entity_types,
            model_name="VietAI/vit5-base",
            max_length=900,
            batch_size=10,
            max_requests_per_key=1400  # Set request limit to 1400 per key (safety margin below 1500)
        )
        
        logger.info(f"NER tagging process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        print("\nProcess interrupted by user. Partial results have been saved.")
        
    except Exception as e:
        logger.error(f"Error in main function: {traceback.format_exc()}")
        print(f"An error occurred: {e}")
        print("Check the log file for details.")


if __name__ == "__main__":
    main()