# Processed data

Thư mục này chỉ chứa artifact được sinh lại từ dữ liệu nguồn. Git lưu file quy
ước này và `video_summary.json` cuối của mỗi video để chia sẻ corpus tìm kiếm;
timeline, validation report và output trung gian vẫn được ignore.

## Cấu trúc runtime hiện tại

```text
data/processed/
└── video_understanding/
    └── <batch_id>/
        └── <video_id>/
            └── pilot/
                ├── timeline.json              # local
                ├── video_summary.json         # tracked
                └── validation_report.json      # local
```

Mỗi video luôn có đúng ba file cuối. Chạy lại pipeline sẽ cập nhật bộ output cố
định này; không tạo thư mục theo timestamp hoặc nhiều phiên bản pilot.

- `timeline.json`: story, event, temporal anchor và reference đến keyframe/ASR.
- `video_summary.json`: nội dung cấp video dùng để chọn đúng video khi retrieval.
- `validation_report.json`: provenance, input hash, thống kê và quality gate của
  chính video đó.

Semantic API đọc trực tiếp `video_understanding/`. Không đặt caption, ASR,
keyframe, object index, file merge trung gian hoặc benchmark report tại đây.
Caption và ASR nguồn tiếp tục nằm dưới `data/metadata/`; map và keyframe dùng các
thư mục chuẩn dưới `data/`.

## Quy tắc output

1. Artifact trung gian phải dùng thư mục tạm và được xóa sau khi publish.
2. Chỉ `pilot/video_summary.json` được commit; hai file pilot còn lại là local.
3. Báo cáo gắn với một video phải nằm trong `pilot/validation_report.json`.
4. Benchmark dùng trong lúc phát triển không được ghi vào `data/processed/`.
5. Chỉ thêm một loại output mới khi đã có consumer trong code và tài liệu schema.
