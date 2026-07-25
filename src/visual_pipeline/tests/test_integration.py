"""
test_integration.py — Integration Test cho /internal/search/visual
===================================================================
Script này test THẬT endpoint đang chạy trên port 8001.
KHÔNG dùng mock — gọi thật vào server, nhận kết quả thật.

Yêu cầu trước khi chạy:
    1. Đã chạy extractor.py  (sinh file .npy)
    2. Đã chạy database.py   (nạp vector vào Qdrant)
    3. Đang chạy server.py   (port 8001)

Cách chạy:
    cd src/visual_pipeline
    python tests/test_integration.py

    # Hoặc truyền query tuỳ ý:
    python tests/test_integration.py "a person riding a bicycle"
"""

import json
import statistics
import sys
import time
import urllib.request
import urllib.error

# Fix encoding khi script được chạy từ PowerShell/CI dùng code page Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
# Dùng IPv4 loopback rõ ràng để tránh `localhost` resolve sang ::1 trong khi
# Uvicorn đang bind 0.0.0.0 (IPv4).
BASE_URL   = "http://127.0.0.1:8001"
ENDPOINT   = f"{BASE_URL}/internal/search/visual"
EVALUATION_TOP_K = 50
DISPLAY_TOP_K = 10

TEST_CASES = [
    {
        "name": "basketball court",
        "query": "basketball players on a court",
        "prompt_variants": [],
        "expected_videos": ["L25_V002"],
        "expected_frames": [55, 56, 57, 60, 61],
        "relevance_window": 5,
        "min_hit_at_5": 1.0,
        "min_recall_at_10": 1.0,
        "min_mrr": 0.20,
    },
    {
        "name": "television broadcast presenter",
        "query": "TV news",
        "prompt_variants": [
            "a television broadcast",
            "a news anchor presenting information on screen",
            "a presenter standing beside an information display",
        ],
        "expected_videos": ["L25_V004"],
        "expected_frames": [5, 175, 177, 305],
        "relevance_window": 5,
        "min_hit_at_5": 1.0,
        "min_recall_at_10": 1.0,
        "min_mrr": 0.20,
    },
]
DEFAULT_TOP_K = 5

# ─── Màu terminal ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

REQUIRED_FIELDS = {
    "frame_id",
    "score",
    "normalized_score",
    "video_name",
    "frame_index",
}


def post_search(
    prompt: str,
    top_k: int = EVALUATION_TOP_K,
    prompt_variants: list[str] | None = None,
) -> dict:
    """Gửi POST request đến /internal/search/visual."""
    payload = json.dumps({
        "visual_prompt": prompt,
        "prompt_variants": prompt_variants or [],
        "top_k": top_k,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def check_server_alive() -> bool:
    """Kiểm tra server có đang chạy không."""
    try:
        urllib.request.urlopen(f"{BASE_URL}/docs", timeout=3)
        return True
    except Exception:
        return False


def validate_response(response: dict, prompt: str) -> list[str]:
    """Kiểm tra format response. Trả về danh sách lỗi (rỗng = OK)."""
    errors = []
    if not isinstance(response, dict):
        errors.append(f"Response không phải dict: {type(response)}")
        return errors

    if response.get("status") != "success":
        errors.append(f"Response status không phải 'success': {response.get('status')}")

    results = response.get("data")
    if not isinstance(results, list):
        errors.append(f"Response data không phải list: {type(results)}")
        return errors

    for i, item in enumerate(results):
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            errors.append(f"Item #{i}: thiếu trường {missing}")
            continue

        if not isinstance(item["frame_id"],    str):
            errors.append(f"Item #{i}: frame_id phải là str")
        if not isinstance(item["score"],       float):
            errors.append(f"Item #{i}: score phải là float")
        if not isinstance(item["normalized_score"], float):
            errors.append(f"Item #{i}: normalized_score phải là float")
        if not isinstance(item["video_name"],  str):
            errors.append(f"Item #{i}: video_name phải là str")
        if not isinstance(item["frame_index"], int):
            errors.append(f"Item #{i}: frame_index phải là int")
        if not (-1.0 <= item["score"] <= 1.0):
            errors.append(f"Item #{i}: score {item['score']:.4f} nằm ngoài [-1,1]")
        if not (0.0 <= item["normalized_score"] <= 1.0):
            errors.append(
                f"Item #{i}: normalized_score "
                f"{item['normalized_score']:.4f} nằm ngoài [0,1]"
            )
        expected_normalized = (item["score"] + 1.0) / 2.0
        if abs(item["normalized_score"] - expected_normalized) > 1e-6:
            errors.append(
                f"Item #{i}: normalized_score không khớp với (score + 1) / 2"
            )
        if not item["frame_id"]:
            errors.append(f"Item #{i}: frame_id rỗng")
        if not item["video_name"]:
            errors.append(f"Item #{i}: video_name rỗng")

    scores = [item["score"] for item in results if isinstance(item.get("score"), float)]
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        errors.append("Kết quả không được sắp xếp theo score giảm dần")

    return errors


def relevant_segments(test_case: dict) -> list[tuple[str, tuple[int, ...]]]:
    """Gộp các ground-truth frame gần nhau thành đoạn relevance."""
    window = test_case.get("relevance_window", 0)
    frames = sorted(set(test_case["expected_frames"]))
    frame_groups = []

    for frame_index in frames:
        if not frame_groups or frame_index - frame_groups[-1][-1] > window:
            frame_groups.append([frame_index])
        else:
            frame_groups[-1].append(frame_index)

    return [
        (video_name, tuple(frame_group))
        for video_name in test_case["expected_videos"]
        for frame_group in frame_groups
    ]


def matches_segment(
    item: tuple[str, int],
    segment: tuple[str, tuple[int, ...]],
    relevance_window: int,
) -> bool:
    """Một kết quả đúng nếu cùng video và nằm gần ground-truth segment."""
    video_name, frame_index = item
    expected_video, expected_frames = segment
    return (
        video_name == expected_video
        and any(
            abs(frame_index - expected_frame) <= relevance_window
            for expected_frame in expected_frames
        )
    )


def calculate_retrieval_metrics(results: list[dict], test_case: dict) -> dict:
    """Tính Hit@5, Recall@10, MRR và khoảng cách top 1 - median."""
    segments = relevant_segments(test_case)
    relevance_window = test_case.get("relevance_window", 0)
    ranked_items = [
        (item["video_name"], item["frame_index"])
        for item in results
    ]

    matched_at_5 = {
        segment_index
        for segment_index, segment in enumerate(segments)
        if any(
            matches_segment(item, segment, relevance_window)
            for item in ranked_items[:5]
        )
    }
    matched_at_10 = {
        segment_index
        for segment_index, segment in enumerate(segments)
        if any(
            matches_segment(item, segment, relevance_window)
            for item in ranked_items[:10]
        )
    }
    hit_at_5 = float(bool(matched_at_5))
    recall_at_10 = len(matched_at_10) / len(segments) if segments else 0.0

    first_relevant_rank = next(
        (
            rank
            for rank, item in enumerate(ranked_items, start=1)
            if any(
                matches_segment(item, segment, relevance_window)
                for segment in segments
            )
        ),
        None,
    )
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    scores = [item["score"] for item in results]
    median_score = statistics.median(scores) if scores else 0.0
    top_1_score = scores[0] if scores else 0.0

    return {
        "hit_at_5": hit_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "first_relevant_rank": first_relevant_rank,
        "top_1_score": top_1_score,
        "median_score": median_score,
        "top_1_median_gap": top_1_score - median_score,
    }


def validate_retrieval_metrics(metrics: dict, test_case: dict) -> list[str]:
    """Kiểm tra các ngưỡng relevance đã khai báo trong test case."""
    errors = []
    thresholds = {
        "hit_at_5": test_case["min_hit_at_5"],
        "recall_at_10": test_case["min_recall_at_10"],
        "mrr": test_case["min_mrr"],
    }
    for metric_name, minimum in thresholds.items():
        actual = metrics[metric_name]
        if actual < minimum:
            errors.append(
                f"{metric_name}={actual:.4f} thấp hơn ngưỡng {minimum:.4f}"
            )
    return errors


def print_results(response: dict, prompt: str, elapsed_ms: float):
    """In kết quả search theo dạng JSON chuẩn cho truy vấn ad-hoc."""
    print(f"\n  {CYAN}⏱  Latency: {elapsed_ms:.1f} ms{RESET}")
    json_str = json.dumps(response, indent=2, ensure_ascii=False)
    for line in json_str.split("\n"):
        print(f"  {line}")


def print_retrieval_results(results: list[dict], limit: int = DISPLAY_TOP_K):
    """In bảng xếp hạng gọn để kiểm tra relevance bằng mắt."""
    print(f"\n  {'Rank':<6}{'Video':<12}{'Frame':<9}{'Score':<10}")
    for rank, item in enumerate(results[:limit], start=1):
        print(
            f"  {rank:<6}{item['video_name']:<12}"
            f"{item['frame_index']:<9}{item['score']:.6f}"
        )


def print_metrics(metrics: dict):
    """In các retrieval metric của một test case."""
    first_rank = metrics["first_relevant_rank"]
    first_rank_text = str(first_rank) if first_rank is not None else "N/A"
    print("\n  Metrics:")
    print(f"    Hit@5             : {metrics['hit_at_5']:.4f}")
    print(f"    Recall@10         : {metrics['recall_at_10']:.4f}")
    print(f"    MRR               : {metrics['mrr']:.4f}")
    print(f"    First relevant    : {first_rank_text}")
    print(f"    Top-1 score       : {metrics['top_1_score']:.6f}")
    print(f"    Median score      : {metrics['median_score']:.6f}")
    print(f"    Top-1/median gap  : {metrics['top_1_median_gap']:.6f}")


def run_format_test(prompt: str, top_k: int = DEFAULT_TOP_K) -> bool:
    """Chạy kiểm tra format cho một truy vấn ad-hoc."""
    print(f"\n{'─'*60}")
    print(f"{BOLD}🔍 Query:{RESET} \"{prompt}\"  (top_k={top_k})")

    # Gọi API
    t0 = time.perf_counter()
    try:
        response = post_search(prompt, top_k)
    except urllib.error.URLError as e:
        print(f"  {RED}❌ Không kết nối được server: {e}{RESET}")
        return False
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Validate format
    errors = validate_response(response, prompt)
    if errors:
        print(f"  {RED}❌ Lỗi format:{RESET}")
        for err in errors:
            print(f"     • {err}")
        return False

    results = response.get("data", [])

    # Validate HTTP + count
    if len(results) == 0 and top_k > 0:
        print(f"  {YELLOW}⚠  Response rỗng (không có kết quả nào){RESET}")

    # In kết quả
    print_results(response, prompt, elapsed_ms)

    # Kiểm tra số lượng kết quả
    if len(results) != top_k:
        print(f"  {YELLOW}⚠  top_k={top_k} nhưng nhận {len(results)} kết quả{RESET}")
    else:
        print(f"  {GREEN}✅ PASS — {len(results)} kết quả, format đúng{RESET}")

    return True


def run_retrieval_test(test_case: dict) -> bool:
    """Chạy một ground-truth retrieval test qua API thật."""
    prompt = test_case["query"]
    prompt_variants = test_case.get("prompt_variants", [])
    print(f"\n{'─'*60}")
    print(f"{BOLD}🔍 Case:{RESET} {test_case['name']}")
    print(f"  Query: \"{prompt}\"  (top_k={EVALUATION_TOP_K})")
    if prompt_variants:
        print(f"  Variants: {prompt_variants}")

    t0 = time.perf_counter()
    try:
        response = post_search(prompt, EVALUATION_TOP_K, prompt_variants)
    except urllib.error.URLError as e:
        print(f"  {RED}❌ Không kết nối được server: {e}{RESET}")
        return False
    elapsed_ms = (time.perf_counter() - t0) * 1000

    response_errors = validate_response(response, prompt)
    if response_errors:
        print(f"  {RED}❌ Lỗi response:{RESET}")
        for error in response_errors:
            print(f"     • {error}")
        return False

    results = response["data"]
    if len(results) < 10:
        print(f"  {RED}❌ Cần ít nhất 10 kết quả để tính Recall@10.{RESET}")
        return False

    metrics = calculate_retrieval_metrics(results, test_case)
    metric_errors = validate_retrieval_metrics(metrics, test_case)

    print(f"  ⏱  Latency: {elapsed_ms:.1f} ms")
    print_retrieval_results(results)
    print_metrics(metrics)

    if metric_errors:
        print(f"\n  {RED}❌ FAIL — relevance chưa đạt:{RESET}")
        for error in metric_errors:
            print(f"     • {error}")
        return False

    print(f"\n  {GREEN}✅ PASS — response và relevance đều đạt{RESET}")
    return True


def main():
    print(f"\n{'='*60}")
    print(f"{BOLD}  RETRIEVAL EVALUATION — /internal/search/visual{RESET}")
    print(f"  Server: {ENDPOINT}")
    print(f"{'='*60}")

    # Kiểm tra server còn sống không
    print("\n🔌 Kiểm tra kết nối server...", end=" ")
    if not check_server_alive():
        print(f"{RED}FAILED{RESET}")
        print(f"\n{RED}❌ Server chưa chạy hoặc không phải port 8001!{RESET}")
        print("   Hãy chạy:  python server.py")
        sys.exit(1)
    print(f"{GREEN}OK{RESET}")

    # Có đối số CLI: giữ chế độ kiểm tra format cho truy vấn ad-hoc.
    if len(sys.argv) > 1:
        queries = sys.argv[1:]
        passed = sum(run_format_test(query) for query in queries)
        total = len(queries)
        print(f"\n{'='*60}")
        status_color = GREEN if passed == total else RED
        print(
            f"{BOLD}  Kết quả: "
            f"{status_color}{passed}/{total} queries PASSED{RESET}"
        )
        print(f"{'='*60}\n")
        sys.exit(0 if passed == total else 1)

    # Không có đối số: chạy benchmark có ground truth.
    passed = 0
    total = len(TEST_CASES)

    for test_case in TEST_CASES:
        ok = run_retrieval_test(test_case)
        if ok:
            passed += 1

    # Tổng kết
    print(f"\n{'='*60}")
    status_color = GREEN if passed == total else RED
    print(
        f"{BOLD}  Kết quả retrieval: "
        f"{status_color}{passed}/{total} cases PASSED{RESET}"
    )
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
