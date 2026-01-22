import re
import json
import uuid
from typing import List, Tuple, Dict, Union


LABELS = ["PER", "ORG", "ADDR", "PHONE"]


def extract_entities(tagged_text: str, labels: List[str] = LABELS) -> Tuple[str, List[Dict]]:
    """
    Chuyển văn bản có tag thành (text_cleaned, spans) dùng cho NER trong Label Studio

    Args:
        tagged_text (str): Văn bản có chứa tag như <PER>...</PER>
        labels (List[str]): Danh sách nhãn cần trích xuất

    Returns:
        Tuple[str, List[Dict]]: Văn bản đã loại tag, danh sách span gồm (start, end, text, label)
    """
    spans = []
    text = tagged_text

    for label in labels:
        pattern = fr"<{label}>(.*?)</{label}>"
        for match in re.finditer(pattern, text):
            raw_entity = match.group(1)
            # Tính offset không tính tag
            pre_text = text[:match.start()]
            clean_start = len(re.sub(fr"<[^>]+>", "", pre_text))
            clean_end = clean_start + len(raw_entity)
            spans.append({
                "start": clean_start,
                "end": clean_end,
                "text": raw_entity,
                "labels": [label]
            })

        text = re.sub(pattern, r"\1", text)  # Bỏ tag, giữ lại text

    return text, spans


def create_labelstudio_item(
    tagged_text: str,
    source_id: Union[int, str] = None,
    model_tag: str = "gemini-auto",
    labels: List[str] = LABELS
) -> Dict:
    """
    Tạo một item NER kiểu Label Studio từ văn bản tag

    Args:
        tagged_text (str): Văn bản có tag
        source_id (Union[int, str]): ID gốc nếu cần tracking
        model_tag (str): Tag của mô hình đã sinh ra label
        labels (List[str]): Danh sách nhãn

    Returns:
        Dict: Dict item chuẩn cho Label Studio import
    """
    clean_text, spans = extract_entities(tagged_text, labels)
    return {
        "id": str(source_id),  # Thay đổi: dùng source_id làm ID
        "data": {"text": clean_text},
        "predictions": [
            {
                "model_version": model_tag,
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": span["start"],
                            "end": span["end"],
                            "text": span["text"],
                            "labels": span["labels"]
                        }
                    } for span in spans
                ],
                "score": 0.5
            }
        ],
        "annotations": []  # Để trống → user chưa chỉnh sửa
    }


def convert_excel_column_to_labelstudio_json(
    excel_path: str,
    text_column: str,
    output_path: str,
    model_tag: str = "gemini-auto"
):
    """
    Chuyển toàn bộ cột văn bản trong Excel sang file JSON cho Label Studio,
    bao gồm cả những văn bản không có tag

    Args:
        excel_path (str): Đường dẫn đến file Excel
        text_column (str): Tên cột chứa văn bản tag
        output_path (str): Đường dẫn để lưu file JSON kết quả
        model_tag (str): Tag tên mô hình tạo prediction
    """
    import pandas as pd
    df = pd.read_excel(excel_path)

    output = []
    for idx, row in df.iterrows():
        tagged_text = row[text_column]
        # Xử lý tất cả các dòng text có giá trị
        if isinstance(tagged_text, str):
            item = create_labelstudio_item(tagged_text, source_id=idx, model_tag=model_tag)
            output.append(item)
        else:
            # Xử lý giá trị không phải string (NaN, None, etc.) bằng cách tạo item rỗng
            item = {
                "id": str(uuid.uuid4()),
                "data": {"text": ""},  # Text rỗng nếu không phải string
                "predictions": [{"model_version": model_tag, "result": [], "score": 0}],
                "annotations": []
            }
            output.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_excel_column_to_labelstudio_json(
        "drive-download-20250512T022115Z-1-001/merged_part5.xlsx", 
        "human_tagged_content", 
        "drive-download-20250512T022115Z-1-001/part5.json"
    )