# AIC 2026 stress-test corpus

This corpus is a deterministic engineering stress test built from the official batch-1 keyframes, object detections, frame maps, and video metadata.

> Important: labels are pseudo-ground truth derived from the provided Faster R-CNN detections and metadata. They are not human annotations and must not be reported as official retrieval accuracy.

## Corpus

- Frames: **1200**
- Videos represented: **267**
- Textual KIS queries: **1200**
- Q&A queries: **267**
- TRAKE sequences: **267**
- Dataset distribution: L21=120, L22=120, L23=120, L24=120, L25=120, L26=120, L27=120, L28=120, L29=120, L30=120

The official preliminary-round scoring accepts at most 100 answers and uses R@k at k = 1, 5, 20, 50, and 100. The included evaluator follows those cutoffs.

## Files

- `manifest.jsonl`: complete frame metadata, keywords, detections, and provenance.
- `manifest.csv`: spreadsheet-friendly frame manifest.
- `queries/textual_kis.jsonl`: one retrieval query per sampled frame.
- `queries/qa.jsonl`: one detector-count question per represented video when possible.
- `queries/trake.jsonl`: ordered 3-4 moment sequences per represented video.
- `queries/all_queries.jsonl`: concatenation of all query types.
- `frames/`: sampled JPEG/PNG files grouped by video ID.
- `objects/`: raw object JSON for each sampled frame when available.

## Representative queries

| ID | Type | Query | Target |
|---|---|---|---|
| tkis-0001 | textual_kis | Tìm khoảnh khắc có khuôn mặt người, người đàn ông, quần áo, and người. Bối cảnh video: 60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024. | L21_V001 / 2856 |
| tkis-0002 | textual_kis | Need exact frame please - person, clothing, human face; maybe from '60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V001 / 11000 |
| tkis-0003 | textual_kis | Find the moment showing human face, clothing, person, and man. Video context: 60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024. | L21_V001 / 18486 |
| tkis-0004 | textual_kis | Tìm khoảnh khắc có ti vi, giày dép, người phụ nữ, and khuôn mặt người. Bối cảnh video: 60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024. | L21_V001 / 26508 |
| tkis-0005 | textual_kis | Need exact frame please - boat, clothing, person, man; maybe from '60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V001 / 35034 |
| tkis-0006 | textual_kis | Find the moment showing flag, plant, and window. Video context: 60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024. | L21_V002 / 2331 |
| tkis-0007 | textual_kis | Tìm khoảnh khắc có trái cây, cây, and cà chua. Bối cảnh video: 60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024. | L21_V002 / 8373 |
| tkis-0008 | textual_kis | Need exact frame please - van, car, wheel, tree; maybe from '60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V002 / 15563 |
| tkis-0009 | textual_kis | Find the moment showing person and clothing. Video context: 60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024. | L21_V002 / 22127 |
| tkis-0010 | textual_kis | Tìm khoảnh khắc có quần áo, người đàn ông, khuôn mặt người, and người phụ nữ. Bối cảnh video: 60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024. | L21_V002 / 28860 |
| tkis-0011 | textual_kis | Tìm một khung hình thuộc video có thông tin: 60 Giây Official, HTV Tin tức, and HTV News. Ưu tiên đúng video và đúng mốc thời gian. | L21_V003 / 2130 |
| tkis-0012 | textual_kis | Find the moment showing plant, dinosaur, fish, and tree. Video context: 60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024. | L21_V003 / 8730 |
| tkis-0013 | textual_kis | Tìm khoảnh khắc có xe tải, phương tiện đường bộ, bánh xe, and người. Bối cảnh video: 60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024. | L21_V003 / 13975 |
| tkis-0014 | textual_kis | Need exact frame please - man, clothing, human face, skyscraper; maybe from '60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V003 / 20340 |
| tkis-0015 | textual_kis | Find the moment showing tower, clothing, person, and street light. Video context: 60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024. | L21_V003 / 27270 |
| tkis-0016 | textual_kis | Tìm khoảnh khắc có vehicle, toy, người, and quần áo. Bối cảnh video: 60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024. | L21_V005 / 2139 |
| tkis-0017 | textual_kis | Need exact frame please - person, boat, wheel, motorcycle; maybe from '60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V005 / 7781 |
| tkis-0018 | textual_kis | Find the moment showing footwear, person, clothing, and man. Video context: 60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024. | L21_V005 / 13908 |
| tkis-0019 | textual_kis | Tìm khoảnh khắc có cây. Bối cảnh video: 60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024. | L21_V005 / 20109 |
| tkis-0020 | textual_kis | Need exact frame please - man, clothing, skyscraper, person; maybe from '60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024'. Ignore unrelated clips. | L21_V005 / 26307 |
| qa-0001 | qa | In the target scene from '60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'person' are visible? | L21_V001 / 35034 |
| qa-0002 | qa | In the target scene from '60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'fruit' are visible? | L21_V002 / 8373 |
| qa-0003 | qa | In the target scene from '60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'land vehicle' are visible? | L21_V003 / 13975 |
| qa-0004 | qa | In the target scene from '60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'footwear' are visible? | L21_V005 / 13908 |
| qa-0005 | qa | In the target scene from '60 Giây Sáng - Ngày 06082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'man' are visible? | L21_V006 / 19206 |
| qa-0006 | qa | In the target scene from '60 Giây Sáng - Ngày 07082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'clothing' are visible? | L21_V007 / 1974 |
| qa-0007 | qa | In the target scene from '60 Giây Sáng - Ngày 08082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'person' are visible? | L21_V008 / 2855 |
| qa-0008 | qa | In the target scene from '60 Giây Sáng - Ngày 09082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'man' are visible? | L21_V009 / 10006 |
| qa-0009 | qa | In the target scene from '60 Giây Sáng - Ngày 10082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'person' are visible? | L21_V010 / 2200 |
| qa-0010 | qa | In the target scene from '60 Giây Sáng - Ngày 11082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'human face' are visible? | L21_V011 / 2077 |
| qa-0011 | qa | In the target scene from '60 Giây Sáng - Ngày 12082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'skyscraper' are visible? | L21_V012 / 17369 |
| qa-0012 | qa | In the target scene from '60 Giây Sáng - Ngày 13082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'skyscraper' are visible? | L21_V013 / 21858 |
| qa-0013 | qa | In the target scene from '60 Giây Sáng - Ngày 14082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'skyscraper' are visible? | L21_V014 / 20589 |
| qa-0014 | qa | In the target scene from '60 Giây Sáng - Ngày 15082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'car' are visible? | L21_V015 / 2733 |
| qa-0015 | qa | In the target scene from '60 Giây Sáng - Ngày 16082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'skyscraper' are visible? | L21_V016 / 9867 |
| qa-0016 | qa | In the target scene from '60 Giây Sáng - Ngày 17082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'footwear' are visible? | L21_V017 / 14969 |
| qa-0017 | qa | In the target scene from '60 Giây Sáng - Ngày 18082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'man' are visible? | L21_V018 / 26070 |
| qa-0018 | qa | In the target scene from '60 Giây Sáng - Ngày 19082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'man' are visible? | L21_V019 / 10637 |
| qa-0019 | qa | In the target scene from '60 Giây Sáng - Ngày 21082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'flower' are visible? | L21_V021 / 27879 |
| qa-0020 | qa | In the target scene from '60 Giây Sáng - Ngày 22082024 - HTV Tin Tức Mới Nhất 2024', how many detected instances of 'tree' are visible? | L21_V022 / 1830 |
| trake-0001 | trake | In the video '60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing human face, man, and clothing; (2) moment showing person, clothing, and human face; (3) moment showing television, footwear, and woman; (4) moment showing boat, clothing, and person. | L21_V001 / [2856, 11000, 26508, 35034] |
| trake-0002 | trake | In the video '60 Giây Sáng - Ngày 02082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing flag, plant, and window; (2) moment showing fruit, tree, and tomato; (3) moment showing person and clothing; (4) moment showing clothing, man, and human face. | L21_V002 / [2331, 8373, 22127, 28860] |
| trake-0003 | trake | In the video '60 Giây Sáng - Ngày 03082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) opening sampled moment associated with 60 Giây Official and HTV Tin tức; (2) moment showing plant, dinosaur, and fish; (3) moment showing man, clothing, and human face; (4) moment showing tower, clothing, and person. | L21_V003 / [2130, 8730, 20340, 27270] |
| trake-0004 | trake | In the video '60 Giây Sáng - Ngày 05082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing vehicle, toy, and person; (2) moment showing person, boat, and wheel; (3) moment showing tree; (4) moment showing man, clothing, and skyscraper. | L21_V005 / [2139, 7781, 20109, 26307] |
| trake-0005 | trake | In the video '60 Giây Sáng - Ngày 06082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing woman, clothing, and man; (2) moment showing person, curtain, and clothing; (3) moment showing helmet, clothing, and man; (4) closing sampled moment associated with 60 Giây Official and HTV Tin tức. | L21_V006 / [2601, 10358, 19206, 28446] |
| trake-0006 | trake | In the video '60 Giây Sáng - Ngày 07082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing man, clothing, and person; (2) moment showing person, helmet, and window; (3) moment showing vehicle and weapon; (4) moment showing person and toy. | L21_V007 / [1974, 8778, 15726, 23469] |
| trake-0007 | trake | In the video '60 Giây Sáng - Ngày 08082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing person, street light, and car; (2) moment showing skyscraper and tree; (3) moment showing clothing and guitar; (4) moment showing human face, person, and clothing. | L21_V008 / [2855, 12367, 21083, 31035] |
| trake-0008 | trake | In the video '60 Giây Sáng - Ngày 09082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing person, clothing, and fashion accessory; (2) moment showing clothing, man, and human face; (3) moment showing flag, drink, and lamp; (4) moment showing car, land vehicle, and coin. | L21_V009 / [2130, 10006, 18408, 26789] |
| trake-0009 | trake | In the video '60 Giây Sáng - Ngày 10082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing person, office building, and clothing; (2) moment showing boat and flag; (3) moment showing car, human face, and person; (4) moment showing dog, animal, and monkey. | L21_V010 / [2200, 10079, 18027, 26136] |
| trake-0010 | trake | In the video '60 Giây Sáng - Ngày 11082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing clothing, girl, and human face; (2) moment showing human face, boy, and clothing; (3) moment showing human face, clothing, and woman; (4) moment showing person, clothing, and human face. | L21_V011 / [2077, 8809, 16019, 22710] |
| trake-0011 | trake | In the video '60 Giây Sáng - Ngày 12082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing tree; (2) moment showing human face, poster, and woman; (3) moment showing man, human face, and skyscraper; (4) moment showing human face, poster, and woman. | L21_V012 / [2205, 8865, 17369, 25574] |
| trake-0012 | trake | In the video '60 Giây Sáng - Ngày 13082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing human face, clothing, and person; (2) moment showing human face, houseplant, and person; (3) moment showing man, clothing, and human face; (4) moment showing person, land vehicle, and wheel. | L21_V013 / [2328, 12080, 21858, 31581] |
| trake-0013 | trake | In the video '60 Giây Sáng - Ngày 14082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing human face, clothing, and skyscraper; (2) moment showing clothing, man, and mobile phone; (3) moment showing clothing, man, and skyscraper; (4) moment showing human face, clothing, and woman. | L21_V014 / [2502, 10998, 20589, 31230] |
| trake-0014 | trake | In the video '60 Giây Sáng - Ngày 15082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing truck, car, and wheel; (2) moment showing snack and food; (3) moment showing man, clothing, and person; (4) moment showing person and bicycle. | L21_V015 / [2733, 13314, 24387, 35963] |
| trake-0015 | trake | In the video '60 Giây Sáng - Ngày 16082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing tripod and billboard; (2) moment showing clothing, man, and skyscraper; (3) moment showing building and person; (4) moment showing building, boy, and person. | L21_V016 / [1926, 9867, 19668, 29348] |
| trake-0016 | trake | In the video '60 Giây Sáng - Ngày 17082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing chair, footwear, and clothing; (2) moment showing tree and plant; (3) moment showing man, footwear, and woman; (4) moment showing person. | L21_V017 / [1717, 8310, 14969, 21780] |
| trake-0017 | trake | In the video '60 Giây Sáng - Ngày 18082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing man, clothing, and human face; (2) moment showing land vehicle, wheel, and tire; (3) moment showing dress, woman, and human face; (4) moment showing man, person, and clothing. | L21_V018 / [2550, 9864, 18355, 26070] |
| trake-0018 | trake | In the video '60 Giây Sáng - Ngày 19082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing human face and woman; (2) moment showing man, shorts, and clothing; (3) moment showing man, human face, and glasses; (4) moment showing bicycle helmet, person, and man. | L21_V019 / [2610, 10637, 20888, 30378] |
| trake-0019 | trake | In the video '60 Giây Sáng - Ngày 21082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing flower, marine invertebrates, and food; (2) moment showing human hand and person; (3) moment showing human face, woman, and clothing; (4) moment showing flower and tree. | L21_V021 / [2406, 10284, 18689, 27879] |
| trake-0020 | trake | In the video '60 Giây Sáng - Ngày 22082024 - HTV Tin Tức Mới Nhất 2024', locate these sampled moments in temporal order: (1) moment showing tree; (2) moment showing clothing, girl, and tree; (3) moment showing tree, motorcycle, and monkey; (4) moment showing poster and chair. | L21_V022 / [1830, 10209, 18900, 27078] |

## Run against the current backend

```powershell
python tools/stress_test/run_backend_stress.py --limit 100
python tools/stress_test/evaluate_predictions.py `
  --queries data/stress_test_1200/queries/all_queries.jsonl `
  --predictions data/stress_test_1200/predictions.jsonl
```

Start with `--limit 100`. A full run can invoke thousands of retrieval calls and may incur Gemini/API usage when LLM re-ranking is enabled.
