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

import sys
import json
import time
import urllib.request
import urllib.error

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8001"
ENDPOINT   = f"{BASE_URL}/internal/search/visual"
DEFAULT_QUERIES = [
    "a red car on the road",
    "people walking on the street",
    "a crowd at a sports event",
    "sunset landscape outdoor",
    "indoor scene with furniture",
]
DEFAULT_TOP_K = 5

# ─── Màu terminal ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

REQUIRED_FIELDS = {"frame_id", "score", "video_name", "frame_index"}


def post_search(prompt: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Gửi POST request đến /internal/search/visual."""
    payload = json.dumps({"visual_prompt": prompt, "top_k": top_k}).encode()
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
        if not isinstance(item["video_name"],  str):
            errors.append(f"Item #{i}: video_name phải là str")
        if not isinstance(item["frame_index"], int):
            errors.append(f"Item #{i}: frame_index phải là int")
        if not (0.0 <= item["score"] <= 1.0):
            errors.append(f"Item #{i}: score {item['score']:.4f} nằm ngoài [0,1]")
        if not item["frame_id"]:
            errors.append(f"Item #{i}: frame_id rỗng")
        if not item["video_name"]:
            errors.append(f"Item #{i}: video_name rỗng")

    return errors


def print_results(response: dict, prompt: str, elapsed_ms: float):
    """In kết quả search theo dạng JSON chuẩn."""
    print(f"\n  {CYAN}⏱  Latency: {elapsed_ms:.1f} ms{RESET}")
    json_str = json.dumps(response, indent=2, ensure_ascii=False)
    for line in json_str.split("\n"):
        print(f"  {line}")


def run_test(prompt: str, top_k: int = DEFAULT_TOP_K) -> bool:
    """Chạy một test case. Trả về True nếu pass."""
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


def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES

    print(f"\n{'='*60}")
    print(f"{BOLD}  INTEGRATION TEST — /internal/search/visual{RESET}")
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

    # Chạy các test case
    passed = 0
    total  = len(queries)

    for prompt in queries:
        ok = run_test(prompt, top_k=DEFAULT_TOP_K)
        if ok:
            passed += 1

    # Tổng kết
    print(f"\n{'='*60}")
    status_color = GREEN if passed == total else RED
    print(f"{BOLD}  Kết quả: {status_color}{passed}/{total} queries PASSED{RESET}")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
