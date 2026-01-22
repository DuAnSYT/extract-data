import json
import pandas as pd


def rebuild_tagged_content(text: str, spans: list) -> str:
    """
    Xây dựng lại văn bản được tag từ text gốc và danh sách spans

    Args:
        text (str): Văn bản gốc
        spans (list): Danh sách span chứa thông tin {start, end, text, labels}

    Returns:
        str: Văn bản được tag lại với các nhãn dạng <LABEL>...</LABEL>
    """
    spans = sorted(spans, key=lambda x: x["start"], reverse=True)  # Sắp xếp từ cuối lên đầu để tránh lệch offset
    for span in spans:
        label = span["labels"][0]
        start, end = span["start"], span["end"]
        entity_text = text[start:end]
        tagged_entity = f"<{label}>{entity_text}</{label}>"
        text = text[:start] + tagged_entity + text[end:]
    return text


def convert_json_to_excel(json_path: str, output_excel_path: str):
    """
    Chuyển đổi file JSON từ Label Studio sang Excel với 3 cột:
    content, gemini_tagged_content, human_tagged_content

    Args:
        json_path (str): Đường dẫn đến file JSON
        output_excel_path (str): Đường dẫn lưu file Excel
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data):
        # Nội dung gốc
        content = item.get("data", {}).get("text", "")

        # Gemini prediction (từ trường prediction.result)
        gemini_result = item.get("annotations", [{}])[0].get("prediction", {}).get("result", [])
        gemini_spans = [
            {
                "start": span["value"]["start"],
                "end": span["value"]["end"],
                "labels": span["value"]["labels"],
            }
            for span in gemini_result
            if span.get("type") == "labels"
        ]
        gemini_tagged_content = rebuild_tagged_content(content, gemini_spans)

        # Human labels (từ trường annotations.result)
        human_result = item.get("annotations", [{}])[0].get("result", [])
        human_spans = [
            {
                "start": span["value"]["start"],
                "end": span["value"]["end"],
                "labels": span["value"]["labels"],
            }
            for span in human_result
            if span.get("type") == "labels"
        ]
        human_tagged_content = rebuild_tagged_content(content, human_spans)

        # Thêm vào danh sách hàng
        rows.append({
            "content": content,
            "gemini_tagged_content": gemini_tagged_content,
            "human_tagged_content": human_tagged_content,
        })

    # Ghi ra file Excel
    df = pd.DataFrame(rows)
    df.to_excel(output_excel_path, index=False, engine="openpyxl")


if __name__ == "__main__":
    # Đường dẫn file JSON input và Excel output
    json_path = "fb_post_test_labeled.json"
    output_excel_path = "Labeled_facebook_posts_24_04_ner_test.xlsx"

    convert_json_to_excel(json_path, output_excel_path)