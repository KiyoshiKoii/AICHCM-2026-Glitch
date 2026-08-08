# Task 1: Trích xuất Ngữ nghĩa
# Pipeline mỗi ảnh:
#   - Florence-2 <DETAILED_CAPTION>  -> mô tả cảnh (tiếng Anh sẵn)
#   - PaddleOCR (detect vùng chữ) + VietOCR (đọc chữ, tiếng Việt chuẩn)
#   - Helsinki-NLP dịch dòng tiếng Việt sang tiếng Anh (dòng tiếng Anh giữ nguyên)
# Kết quả gom vào metadata.json theo schema cho Task 2 (BM25).

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    from common.frame_ref import frame_id as btc_frame_id
    from common.frame_ref import resolve as resolve_btc_frame
except ImportError:  # direct package import from repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.frame_ref import frame_id as btc_frame_id
    from common.frame_ref import resolve as resolve_btc_frame

FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Nới rộng box khi crop: dấu thanh tiếng Việt nằm PHÍA TRÊN chữ, crop sát viền
# box sẽ cắt mất dấu ("bắt buộc" -> "bát buộc").
CROP_PADDING = 4

VIETNAMESE_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def is_vietnamese(text: str) -> bool:
    return bool(VIETNAMESE_CHARS.search(text))


def parse_frame_info(filename: str | Path) -> dict:
    path = Path(filename)
    if re.fullmatch(r"L\d+_V\d+", path.parent.name) and path.stem.isdigit():
        ref = resolve_btc_frame(path.parent.name, int(path.stem))
        return {
            "schema_version": "1.1",
            "frame_id": btc_frame_id(ref),
            "video_name": ref.video_id,
            "frame_index": ref.frame_idx,
            "keyframe_n": ref.keyframe_n,
            "timestamp_ms": ref.timestamp_ms,
            "has_visual_text": True,
        }
    # "vid01_f0001.png" -> video_name="vid01.mp4", frame_index=1, frame_id="vid01_f0001"
    stem = path.stem
    video_name, frame_part = stem.rsplit("_f", 1)
    return {
        "frame_id": stem,
        "video_name": f"{video_name}.mp4",
        "frame_index": int(frame_part),
    }


class SemanticExtractor:
    """Gom 4 model: Florence-2 (caption) + PaddleOCR (detect) + VietOCR (rec) + translator."""

    def __init__(self):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, pipeline

        self.np = np
        self.Image = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[init] device={self.device}, dtype={self.dtype}")

        print(f"[init] Loading Florence-2 ({FLORENCE_MODEL_ID})...")
        self.florence = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )

        print("[init] Loading PaddleOCR (detect)...")
        from paddleocr import PaddleOCR

        # enable_mkldnn=False: né bug oneDNN+PIR của paddlepaddle 3.x trên CPU.
        self.detector = PaddleOCR(
            lang="vi",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print("[init] Loading VietOCR (recognize)...")
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["device"] = self.device
        cfg["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(cfg)

        print(f"[init] Loading translator ({TRANSLATION_MODEL_ID})...")
        self.translator = pipeline(
            "translation",
            model=TRANSLATION_MODEL_ID,
            device=0 if self.device == "cuda" else -1,
        )

    def _florence_task(self, image: Image.Image, task_prompt: str) -> str:
        inputs = self.processor(
            text=task_prompt, images=image, return_tensors="pt"
        ).to(self.device, self.dtype)
        generated_ids = self.florence.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed[task_prompt]

    def caption(self, image: Image.Image) -> str:
        return self._florence_task(image, "<DETAILED_CAPTION>")

    def ocr_lines(self, image: Image.Image, image_path: Path) -> list[str]:
        """Đọc chữ bằng ENSEMBLE hai model, chọn theo thế mạnh của từng cái.

        PaddleOCR vừa detect vùng chữ vừa tự đọc luôn (rec_texts) — ta tận dụng cả hai:
          - Dòng CÓ dấu tiếng Việt  -> lấy VietOCR   (PaddleOCR rớt dấu: "Nếu giữ" -> "Nu gi")
          - Dòng KHÔNG dấu (Anh/toán) -> lấy PaddleOCR (VietOCR đọc sai ký hiệu:
            "1 + 2 + 3 + 4 = 10" -> "1%2%344-10")
        """
        result = self.detector.predict(str(image_path))
        if not result:
            return []
        res = result[0]
        polys = res.get("rec_polys", res.get("dt_polys")) or []
        paddle_texts = res.get("rec_texts") or []

        width, height = image.size
        crops = []
        kept_paddle_texts = []  # phải song song với crops, vì có box bị bỏ qua
        for poly, paddle_text in zip(polys, paddle_texts):
            pts = self.np.array(poly).reshape(-1, 2)
            x1 = max(int(pts[:, 0].min()) - CROP_PADDING, 0)
            y1 = max(int(pts[:, 1].min()) - CROP_PADDING, 0)
            x2 = min(int(pts[:, 0].max()) + CROP_PADDING, width)
            y2 = min(int(pts[:, 1].max()) + CROP_PADDING, height)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            kept_paddle_texts.append(paddle_text)

        if not crops:
            return []

        # Đọc cả batch 1 lần thay vì tuần tự từng crop (mỗi ảnh có 24-50 crop).
        try:
            viet_texts = self.recognizer.predict_batch(crops)
        except Exception as exc:
            # Batch lỗi -> lùi về đọc từng crop, để 1 crop hỏng không mất cả ảnh.
            print(f"  [warn] predict_batch failed ({exc}), fallback per-crop")
            viet_texts = []
            for crop in crops:
                try:
                    viet_texts.append(self.recognizer.predict(crop))
                except Exception:
                    viet_texts.append("")

        lines = []
        for paddle_text, viet_text in zip(kept_paddle_texts, viet_texts):
            viet_text = (viet_text or "").strip()
            paddle_text = (paddle_text or "").strip()

            chosen = viet_text if is_vietnamese(viet_text) else paddle_text
            if not chosen:  # model được chọn đọc ra rỗng -> lấy model còn lại
                chosen = viet_text or paddle_text
            if chosen:
                lines.append(chosen)
        return lines

    def translate_lines(self, lines: list[str]) -> list[str]:
        # Chỉ dịch dòng có chữ tiếng Việt; dòng tiếng Anh/mã (ActualStartTime...) giữ nguyên.
        vi_idx = [i for i, line in enumerate(lines) if is_vietnamese(line)]
        out = list(lines)
        if vi_idx:
            translated = self.translator([lines[i] for i in vi_idx], max_length=512)
            for i, tr in zip(vi_idx, translated):
                out[i] = tr["translation_text"]
        return out

    def process_image(self, image_path: Path) -> dict:
        with self.Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            caption = self.caption(image)
            raw_lines = self.ocr_lines(image, image_path)
        finally:
            image.close()
        en_lines = self.translate_lines(raw_lines)

        info = parse_frame_info(image_path)
        info["caption"] = caption
        info["ocr_text"] = " ".join(en_lines)  # tiếng Anh -> nạp BM25 (Task 2)
        info["ocr_text_raw"] = " ".join(raw_lines)  # VietOCR gốc -> đối chiếu (Task 5)
        return info


def extract_metadata(input_dir: str, output_path: str, limit: int | None = None, single_image: str | None = None):
    extractor = SemanticExtractor()

    if single_image:
        image_paths = [Path(single_image)]
        print(f"[run] Processing single image: {single_image}")
    else:
        image_paths = sorted(
            p for p in Path(input_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            image_paths = image_paths[:limit]
        print(f"[run] Found {len(image_paths)} images in {input_dir}")

    results = []
    for i, img_path in enumerate(image_paths, start=1):
        start = time.time()
        try:
            record = extractor.process_image(img_path)
            results.append(record)
            print(f"[{i}/{len(image_paths)}] {img_path.name} done in {time.time() - start:.1f}s")
        except Exception as exc:  # 1 ảnh hỏng không làm chết cả batch
            print(f"[{i}/{len(image_paths)}] {img_path.name} FAILED: {exc}")

    Path(output_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[run] Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất caption (Florence-2) + OCR đã dịch (PaddleOCR+VietOCR+MT) ra metadata.json"
    )
    parser.add_argument("--input-dir", default="src/semantic_pipeline/sample_frames")
    parser.add_argument("--output", default="src/semantic_pipeline/sample_frames/metadata.json")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu (test nhanh)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến 1 tấm ảnh cụ thể cần test")
    args = parser.parse_args()

    extract_metadata(args.input_dir, args.output, args.limit, args.image)
