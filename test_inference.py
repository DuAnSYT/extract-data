from ner_predictor import VietnameseNERPredictor
from ner_postprocessing import quick_postprocess as ner_postprocess
from rule import extract_phone, extract_licenses_and_certificates
from clean_text import clean_text, remove_crawl_artifacts


def main(ad_content):
    content = clean_text(ad_content)
    content = remove_crawl_artifacts(content)
    entity_extractor = VietnameseNERPredictor()
    entity_extraction = entity_extractor.predict_text(content)
    entity_extraction = ner_postprocess(entity_extraction)

    phone_numbers = extract_phone(content)
    license_extraction = extract_licenses_and_certificates(content)

    # Tạo kết quả đầy đủ
    result = {}
    information = {
        "organization_names": entity_extraction.get('organization_names', []),
        "addresses": entity_extraction.get('addresses', []),
        "doctor_names": entity_extraction.get('persons', []),
        "phone_numbers": phone_numbers, #list
        "operating_licenses": license_extraction.get('operating_licenses', []),
        "medical_certificates": license_extraction.get('medical_certificates', []),
        "working_hours": None # dummy value, can be updated later
    }
    print(information)

if __name__ == "__main__":
    ad_content = """
❤ HỆ LỤY TỪ VIỆC CẮT 2/3 DẠ DÀY ĐỂ GIẢM C.Â.N & GIẢI PHÁP AN TOÀN TẠI THERAX

❌ Giảm c.â.n bằng phẫu thuật (cắt 2/3 dạ dày) có thể đi kèm nhiều rủi ro:
• Rối loạn tiêu hoá, trào ngược, thiếu vi chất
• Đau kéo dài, thời gian hồi phục lâu
• Chi phí cao, nguy cơ để lại biến chứng

💯 Giải pháp tại THERAX: Phác đồ chuẩn Y Khoa, cá nhân hoá theo bác sĩ, hỗ trợ dinh dưỡng & vận động. Kết hợp máy móc Công nghệ cao Robot AI 6D – SlimPro Tech định vị mô mỡ chính xác, giúp cơ thể săn chắc – siết eo một cách thoải mái.

✅ Hỗ trợ cải thiện vóc dáng rõ rệt chỉ sau 01 liệu trình
✅ Giảm bệnh nền gây ảnh hưởng tới sức khỏe
✅ Công nghệ an toàn – không đau, không nghỉ dưỡng
👉 Chỉ với 60 phút, bạn sẽ cảm nhận được sự thay đổi: vóc dáng gọn gàng, tự tin hơn.

✅  CAM KẾT CHẤT LƯỢNG
- Bác sĩ trực tiếp thăm khám & thực hiện
- Không đau nhức, sưng tấy; không tổn thương mô da
- An toàn – lộ trình minh bạch – giấy tờ cam kết rõ ràng

📌 Đăng ký ngay hôm nay để được chuyên viên 10+ năm kinh nghiệm tư vấn chi tiết & nhận ưu đãi dành riêng cho 20 khách hàng đầu tiên.
-----------💦-----------
🏩 PHÒNG KHÁM QUỐC TẾ THERA X
☎ Hotline: 0918 680 551
⏰ Hoạt động: 08h30 – 20h00 (tất cả các ngày trong tuần)
📍 Địa chỉ: 13-E Đường Bạch Đằng (nối dài), P. Phú Cường, Thủ Dầu Một, Bình Dương
0:00 / 2:16
------------------------------------------------------------
Trung Tâm Giảm Cân - Kiểm Soát Bệnh Nền TheraX Bình Dương
    """
    main(ad_content)