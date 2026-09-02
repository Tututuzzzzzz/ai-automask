# Bộ dữ liệu phân vùng cốc sứ — mô tả và quy trình gán nhãn

Tài liệu này mô tả nguồn gốc, quy trình gán nhãn và cách chia tập của bộ dữ liệu
dùng để huấn luyện mô hình phân vùng ảnh cho nhóm cốc sứ.

Tái lập toàn bộ bằng ba lệnh:

```bash
python scripts/build_mug_dataset.py     # gom + khử trùng lặp + sinh nhãn thô
# duyệt nhãn tại http://localhost:8000/review
python scripts/split_dataset.py         # chia train / val / test theo nhóm
```

---

## 1. Nguồn dữ liệu

Ảnh **không phải đi thu thập mới**. Hệ thống Mockup Generator đã lưu sẵn ảnh
chụp phôi cho từng base tại `src/main/resources/mask/<BASE_CODE>/`, dùng để ghép
mockup. Đây đúng là loại ảnh mô hình sẽ gặp khi vận hành, nên dùng chính chúng
làm dữ liệu huấn luyện là lựa chọn sát thực tế nhất.

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | `mockupgenerator/src/main/resources/mask/*/[màu]-[mặt]-model.jpg` |
| Độ phân giải | 1200 × 1200, đồng nhất toàn bộ |
| Nền | Nền trắng studio, có bóng đổ mềm |
| Số base code | 16 |
| Số màu | 16 (black, white, navy, maroon, bordeaux, gold, pink, red, orange, yellow, blue, green, light-blue, light-green, light-yellow, black-white) |
| Mặt chụp | front (1 cốc) và back (2 cốc cạnh nhau) |
| Bản quyền | Tài sản nội bộ của công ty, không có ràng buộc giấy phép bên ngoài |

### Từ 372 file xuống 121 ảnh

Con số thô gây hiểu nhầm, nên ghi rõ ở đây quá trình lọc:

| Bước | Số lượng | Lý do loại |
|---|---:|---|
| File khớp mẫu `*model*.jpg` | 372 | — |
| Loại: **không phải ảnh phôi** | −13 | Các file `1-Model-Mask.jpg`, `2-Model-Mask-Left.jpg`… là asset ghép sẵn: ảnh cốc **đã dán khối mask trắng lên**. Lọt vào vì Windows so khớp tên không phân biệt hoa thường. Nhãn AI sinh cho chúng là rác (bao cả khúc gỗ kê, cả khối mask) nên phải loại |
| Loại: **trùng lặp nội dung** | −238 | Nhiều base dùng chung y hệt một file ảnh. Kiểm bằng MD5. Nếu giữ, cùng một tấm ảnh sẽ nằm ở cả train lẫn test và mọi chỉ số đánh giá đều vô nghĩa |
| **Còn lại** | **121** | Ảnh duy nhất, đưa vào bộ dữ liệu |

Mỗi ảnh giữ trường `occurrences` và `used_by_bases` trong manifest để truy vết
được nó xuất hiện ở những base nào.

---

## 2. Quy trình gán nhãn

Bài toán là **phân vùng nhị phân**: pixel thuộc sản phẩm = 255, còn lại = 0.
Bóng đổ trên nền **không** thuộc sản phẩm. Lỗ quai cốc **không** thuộc sản phẩm.
Với ảnh mặt back có hai cốc, **cả hai** đều thuộc sản phẩm.

Quy trình hai giai đoạn, người vẫn là người quyết định cuối:

### Giai đoạn 1 — Máy đề xuất

`scripts/build_mug_dataset.py` chạy pipeline AI (BiRefNet + tinh chỉnh biên +
kiểm định chất lượng) trên từng ảnh, sinh:

- `labels_auto/<id>.png` — nhãn đề xuất, cùng độ phân giải ảnh gốc
- `overlays/<id>.jpg` — ảnh chồng lớp để mắt người soát nhanh
- một dòng trong `manifest.csv` kèm verdict, confidence và các chỉ số chất lượng

Kết quả trên 121 ảnh: **96 READY (79.3 %)**, **25 REVIEW (20.7 %)**, 0 FAILED.
Thời gian 235 giây (1.95 s/ảnh) trên GPU laptop 4 GB.

### Giai đoạn 2 — Người duyệt

Giao diện tại `http://localhost:8000/review`. Mỗi ảnh nhận một trong ba quyết
định, ghi thẳng vào `manifest.csv` ngay khi bấm:

| Quyết định | Ý nghĩa | Hệ quả |
|---|---|---|
| **accept** (phím `A`) | Nhãn máy đúng | Chép nguyên sang `labels/` |
| **fix** (phím `S`) | Người sửa lại bằng cọ | Bản đã sửa ghi vào `labels/` |
| **reject** (phím `R`) | Ảnh không dùng được | Loại khỏi bộ dữ liệu, vẫn ghi lại để thống kê trung thực |

Công cụ có cọ thêm/xoá vẽ trực tiếp ở **độ phân giải gốc**, hoàn tác, nút quay
về nhãn máy, và ba chế độ xem (chồng lớp / chỉ mask / chỉ ảnh) với thanh chỉnh
độ mờ. Server nhị phân hoá lại và kiểm tra kích thước trước khi ghi, nên nhãn
cuối luôn là 0/255 và luôn khớp pixel với ảnh.

**Thứ tự làm việc khuyến nghị:** lọc `REVIEW` trước — đó là nhóm máy tự báo là
không chắc, và cũng là nhóm đáng để mắt người ở lâu nhất. Sau đó soát lướt lưới
overlay của nhóm `READY`. Nếu nhóm READY sạch hoàn toàn, có thể nhận hàng loạt:

```bash
curl -X POST http://localhost:8000/v1/labeling/accept-all-ready
```

Thao tác này ghi `reviewer_note = "bulk-accept: verdict READY"` để về sau vẫn
phân biệt được ảnh nào duyệt từng cái, ảnh nào duyệt hàng loạt. **Chỉ dùng sau
khi đã soát mắt**, nếu không thì bộ nhãn không còn là "đã gán nhãn" nữa mà chỉ
là đầu ra của máy được đổi tên.

### Vì sao dùng nhãn máy làm điểm khởi đầu

Vẽ tay 121 mask ở độ chính xác pixel mất khoảng 2–4 giờ và vẫn chủ quan ở đúng
chỗ khó nhất: viền quai, ranh giới cốc/bóng đổ. Ảnh nền trắng sạch nên mô hình
cắt gần như hoàn hảo; việc của người chuyển từ *vẽ* sang *duyệt*, nhanh hơn hàng
chục lần mà chất lượng cao hơn.

Đánh đổi phải nói rõ: nhãn khởi tạo từ mô hình A rồi dùng để huấn luyện mô hình
B thì B thừa hưởng cả thiên kiến của A. Với bài toán này rủi ro thấp — BiRefNet
là mô hình tổng quát, không huấn luyện trên dữ liệu cốc sứ của công ty — nhưng
đây là lý do bước duyệt của người là **bắt buộc**, không phải tuỳ chọn.

---

## 3. Cách chia tập

**Chia theo nhóm, không chia ngẫu nhiên theo ảnh.**

Cùng một phôi cốc được chụp lại cho nhiều màu men. `USACM11/white-front` và
`USACM11/black-front` là *cùng một hình dáng*, chỉ khác màu. Chia ngẫu nhiên thì
bản trắng vào train, bản đen vào test, mô hình chỉ việc nhớ lại hình dáng đã
thấy — IoU trên test sẽ đẹp một cách giả tạo và không phản ánh năng lực thật.

Nhóm = **`(base_code, side)`**. Mọi màu của cùng một base và cùng một mặt luôn
nằm trọn trong một tập.

| Tham số | Giá trị |
|---|---|
| Tỉ lệ | 70 / 15 / 15 |
| Seed | 42 |
| Đơn vị chia | nhóm `(base_code, side)` |
| Thuật toán | duyệt nhóm từ lớn đến nhỏ, mỗi nhóm vào tập đang thiếu nhất so với hạn mức |

Không xáo trộn thuần tuý vì chỉ có ~32 nhóm: một cú xáo xấu có thể làm tập test
chỉ còn 5 % dữ liệu.

Kết quả (chạy thử trên 121 ảnh):

| Tập | Ảnh | Tỉ lệ | Nhóm | Base code |
|---|---:|---:|---:|---:|
| train | 84 | 69.4 % | 22 | 15 |
| val | 19 | 15.7 % | 5 | 5 |
| test | 18 | 14.9 % | 5 | 5 |

`scripts/split_dataset.py` tự kiểm tra rò rỉ: nếu bất kỳ nhóm nào nằm ở hai tập,
script báo lỗi và trả mã thoát khác 0. Script cũng **từ chối chạy** nếu còn ảnh
chưa duyệt, trừ khi truyền `--allow-unreviewed` (chỉ để thử, kết quả không dùng
để báo cáo).

---

## 4. Cấu trúc thư mục

```
data/mugs/
├── images/              121 ảnh JPG 1200×1200, tên: <stt>_<base>_<màu>_<mặt>.jpg
├── labels_auto/         nhãn máy đề xuất (PNG xám, 0/255)
├── labels/              nhãn ĐÃ DUYỆT — đây mới là ground truth
├── overlays/            ảnh chồng lớp để soát nhanh
├── splits/
│   ├── train.txt        mỗi dòng: <đường dẫn ảnh>\t<đường dẫn nhãn>
│   ├── val.txt
│   └── test.txt
├── manifest.csv         một dòng một ảnh, đầy đủ metadata + quyết định duyệt
├── build_summary.json   thống kê lúc gom dữ liệu
└── split_summary.json   thống kê lúc chia tập + kết quả kiểm tra rò rỉ
```

### Các cột trong `manifest.csv`

| Cột | Ý nghĩa |
|---|---|
| `id` | Định danh, cũng là tên file ảnh/nhãn |
| `image`, `label_auto`, `label_final`, `overlay` | Đường dẫn tương đối |
| `base_code`, `color`, `side` | Metadata trích từ đường dẫn nguồn |
| `group` | `base_code_side` — đơn vị chia tập |
| `verdict`, `confidence` | Máy tự chấm |
| `coverage`, `ensemble_iou`, `holes`, `solidity` | Chỉ số chất lượng, dùng để lọc khi duyệt |
| `review_status`, `reviewer_note` | Quyết định của người |
| `split` | train / val / test, điền sau khi chia |
| `md5`, `occurrences`, `used_by_bases`, `source_path` | Truy vết nguồn gốc |

---

## 5. Giới hạn cần biết

1. **121 ảnh là nhỏ.** Đủ để fine-tune một mô hình đã pre-train, **không** đủ để
   huấn luyện U-Net từ đầu mà không tăng cường dữ liệu mạnh. Kế hoạch huấn luyện
   phải tính đến điều này.
2. **Nền đồng nhất.** Toàn bộ là nền trắng studio. Mô hình huấn luyện trên đó sẽ
   yếu trên ảnh nền phức tạp. Nếu hệ thống cần xử lý ảnh lifestyle, phải bổ sung
   dữ liệu loại đó.
3. **Chỉ 16 base code.** Mức đa dạng hình dáng bị giới hạn bởi số phôi công ty
   đang bán, không phải bởi quy trình.
4. **Nhãn khởi tạo từ mô hình.** Xem mục 2. Bước duyệt của người là bắt buộc.
5. **Ảnh mặt back có hai cốc.** Nếu bài toán hạ nguồn giả định một sản phẩm mỗi
   ảnh, phải xử lý riêng — nhãn ở đây bao cả hai cốc, đúng như ảnh thể hiện.
