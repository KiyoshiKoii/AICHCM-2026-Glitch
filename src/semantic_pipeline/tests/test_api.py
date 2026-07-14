"""Task 5 - API Test cho endpoint /internal/search/text.

Đảm bảo JSON trả về đúng format [frame_id, score, video_name, frame_index]
với HTTP 200, tuân thủ API Contract v1 mục 5.
"""

ENDPOINT = "/internal/search/text"
CONTRACT_FIELDS = {"frame_id", "score", "video_name", "frame_index"}


class TestContract:
    """Response phải khớp API Contract đã chốt với Dev 3."""

    def test_returns_200(self, client):
        r = client.post(ENDPOINT, json={"keywords": ["booking", "entity"], "top_k": 5})
        assert r.status_code == 200

    def test_response_wrapper(self, client):
        body = client.post(ENDPOINT, json={"keywords": ["booking"]}).json()
        assert set(body.keys()) == {"status", "data"}
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_item_has_exactly_contract_fields(self, client):
        body = client.post(ENDPOINT, json={"keywords": ["booking", "entity"]}).json()
        assert body["data"], "cần ít nhất 1 kết quả để kiểm tra field"
        for item in body["data"]:
            assert set(item.keys()) == CONTRACT_FIELDS

    def test_field_types(self, client):
        item = client.post(ENDPOINT, json={"keywords": ["booking"]}).json()["data"][0]
        assert isinstance(item["frame_id"], str)
        assert isinstance(item["score"], float)
        assert isinstance(item["video_name"], str)
        assert isinstance(item["frame_index"], int)  # int, không phải str

    def test_scores_are_raw_not_normalised(self, client):
        """Luật task.md: KHÔNG chuẩn hóa điểm. BM25 thô có thể > 1."""
        body = client.post(ENDPOINT, json={"keywords": ["booking", "entity", "record"]}).json()
        assert body["data"][0]["score"] > 1.0

    def test_sorted_by_score_desc(self, client):
        data = client.post(ENDPOINT, json={"keywords": ["triangle", "man"], "top_k": 10}).json()["data"]
        scores = [item["score"] for item in data]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, client):
        data = client.post(ENDPOINT, json={"keywords": ["the", "a", "image"], "top_k": 2}).json()["data"]
        assert len(data) <= 2


class TestEdgeCases:
    """Ca biên — phân biệt rõ 'không tìm thấy' với 'client gửi sai'."""

    def test_no_match_returns_empty_list_not_error(self, client):
        """Từ khóa hợp lệ nhưng không frame nào khớp -> 200 + rỗng, KHÔNG phải 404."""
        r = client.post(ENDPOINT, json={"keywords": ["zzzznonexistentkeyword"]})
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_empty_keywords_rejected(self, client):
        """keywords rỗng gần như chắc chắn là bug bên Dev 3 -> 422 để họ thấy ngay."""
        assert client.post(ENDPOINT, json={"keywords": []}).status_code == 422

    def test_missing_keywords_rejected(self, client):
        assert client.post(ENDPOINT, json={"top_k": 5}).status_code == 422

    def test_wrong_keywords_type_rejected(self, client):
        assert client.post(ENDPOINT, json={"keywords": "not-a-list"}).status_code == 422

    def test_top_k_out_of_range_rejected(self, client):
        assert client.post(ENDPOINT, json={"keywords": ["x"], "top_k": 0}).status_code == 422
        assert client.post(ENDPOINT, json={"keywords": ["x"], "top_k": 999999}).status_code == 422

    def test_top_k_defaults_to_200(self, client):
        assert client.post(ENDPOINT, json={"keywords": ["triangle"]}).status_code == 200

    def test_top_k_larger_than_corpus_does_not_crash(self, client):
        r = client.post(ENDPOINT, json={"keywords": ["booking"], "top_k": 1000})
        assert r.status_code == 200


class TestInfrastructure:
    def test_health_endpoint(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["documents"] > 0

    def test_swagger_available(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_documents_response_schema(self, client):
        """Dev 3 phải xem được hình dạng response trên Swagger."""
        schema = client.get("/openapi.json").json()
        assert "SearchResponse" in schema["components"]["schemas"]
