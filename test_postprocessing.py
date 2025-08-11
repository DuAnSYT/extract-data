from ner_predictor import VietnameseNERPredictor
from ner_postprocessing import quick_postprocess


predictor = VietnameseNERPredictor()

text = """
    [MEGA LIVE 14.11] DEAL ĐỈNH THÌNH LÌNH - LÀM ĐẸP SIÊU DÍNH Cùng Ca sĩ THU THỦY Săn deal ĐỒNG GIÁ từ 99K - Quà tặng Làm đẹp siêu khủng----------Danh sách các dịch vụ siêu Hot sẽ xuất hiện trong phiên live lần này, với mức giá ĐỘC QUYỀN siêu giảm:- Chăm Sóc Da Cao Cấp 3in1- Dr.Vip Chăm Sóc Da Lão Hoá ECM- Dr.Vip Ủ Trắng Face Collagen- Dr.Vip Chăm Sóc Vùng Mắt ECM - Xoá nhăn vết chân chim- Dr.Vip Collagen Thuỷ Phân - Ức Chế Đốm Nâu- Dr. Acne Trị Mụn Chuẩn Y Khoa- Dr.Seoul Laser Pico 5.0- Dr.Slim Giảm Mỡ Exilis Detox- Dr. White Tắm Trắng Hoàng Gia- Phun mày- Phun mí- Phun môiNgoài ra, các hoạt động cộng hưởng tại phiên live: Giao lưu, trò chuyện, chia sẻ kiến thức làm đẹp cùng ca sĩ Thu Thủy Tư vấn & giải đáp về dịch vụ cùng Seoul Center Tham gia minigame - Nhận quà độc quyền thương hiệuTất cả DEAL hời đã sẵn sàng "lên kệ" vào lúc 19h00 | 14.11.2024 tại FB/ Tiktok Seoul Center và Fb/tiktok ca sĩ Thu Thủy Giảm giá kịch sàn, chỉ có trên live Đặt lịch săn ngay làm đẹp đón tết cùng Thu Thủy nhé!-------------Hệ Thống Thẩm Mỹ Quốc Tế Seoul CenterSẵn sàng lắng nghe mọi ý kiến của khách hàng: 1800 3333Đặt lịch ngay với Top dịch vụ đặc quyền: Website: Zalo: Tiktok: Youtube: Top 10 Thương Hiệu Xuất Sắc Châu Á 2022 & 2023Huy Chương Vàng Sản Phẩm, Dịch Vụ Chất Lượng Châu Á 2023Thương Hiệu Thẩm Mỹ Dẫn Đầu Việt Nam 2024SEOUL CENTER - PHỤNG SỰ TỪ TÂM#SeoulCenter #ThamMyVien
"""

results = predictor.predict_text(text)

postprocessing_results = quick_postprocess(results)

print(postprocessing_results)