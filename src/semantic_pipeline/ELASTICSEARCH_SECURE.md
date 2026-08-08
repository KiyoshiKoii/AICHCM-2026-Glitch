# Elasticsearch có authentication + TLS (local-secure/staging)

File `docker-compose.elasticsearch.secure.yml` là profile bảo mật tách biệt với
`docker-compose.elasticsearch.yml` đang dùng cho local development:

- project Compose riêng: `semantic_pipeline_secure`;
- port mặc định riêng: `127.0.0.1:19200`;
- data, certificate và CA dùng các named volume riêng;
- bật security, Basic Authentication, HTTPS và transport TLS;
- password được mount bằng Docker secret file, không nằm trong Compose hoặc Git;
- certificate có SAN cho `localhost`, `127.0.0.1` và hostname nội bộ
  `elasticsearch`;
- CA private key nằm trong volume chỉ service bootstrap certificate được mount.

Profile này giúp kiểm thử tích hợp có bảo mật và staging một node. Nó không biến
Docker Compose trên laptop thành một deployment production/HA.

## 1. Tạo secret bên ngoài repository

Chạy PowerShell từ repository root. Đoạn dưới sinh password ngẫu nhiên 32 byte
và lưu dưới `%LOCALAPPDATA%`, không tạo secret trong Git working tree.

```powershell
$secureDir = Join-Path $env:LOCALAPPDATA "AICHCM-2026-Glitch\elasticsearch-secure"
New-Item -ItemType Directory -Force -Path $secureDir | Out-Null

$passwordBytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($passwordBytes)
$rng.Dispose()
$elasticPassword = [Convert]::ToBase64String($passwordBytes)

$passwordFile = Join-Path $secureDir "elastic_password.txt"
[System.IO.File]::WriteAllText(
  $passwordFile,
  $elasticPassword,
  [System.Text.UTF8Encoding]::new($false)
)

$composeEnv = Join-Path $secureDir "compose.env"
$passwordPathForCompose = $passwordFile.Replace("\", "/")
$envLines = @(
  "ELASTIC_PASSWORD_SECRET_FILE=$passwordPathForCompose",
  "ELASTICSEARCH_SECURE_PORT=19200",
  "ELASTICSEARCH_CERT_DAYS=365"
)
[System.IO.File]::WriteAllLines(
  $composeEnv,
  $envLines,
  [System.Text.UTF8Encoding]::new($false)
)
```

`src/semantic_pipeline/.env.elasticsearch.secure.example` chỉ là template tên
biến; nó không chứa password. Hạn chế quyền đọc `elastic_password.txt` và
`compose.env` cho tài khoản người dùng đang chạy Docker Desktop.

## 2. Validate mà chưa khởi động container

```powershell
docker compose `
  --env-file $composeEnv `
  -f src/semantic_pipeline/docker-compose.elasticsearch.secure.yml `
  config --quiet
```

Lệnh thành công không in gì và không tạo container. Không chia sẻ output đầy đủ
của `docker compose config` trong log công khai vì một số loại Compose secret
khác có thể bị render khi cấu hình được mở rộng.

## 3. Khởi động secure stack

Node no-auth hiện tại ở port `9200` có thể tiếp tục chạy; secure node dùng port
`19200`.

```powershell
docker compose `
  --env-file $composeEnv `
  -f src/semantic_pipeline/docker-compose.elasticsearch.secure.yml `
  up -d

docker compose `
  --env-file $composeEnv `
  -f src/semantic_pipeline/docker-compose.elasticsearch.secure.yml `
  ps -a
```

Kết quả mong đợi: `cert_setup` đã exit code `0`, còn `elasticsearch` chuyển sang
`healthy`. Bootstrap certificate dùng `elasticsearch-certutil` có sẵn trong
đúng image Elastic 9.4.2, không tải binary không rõ nguồn.

Nếu named data volume đã được khởi tạo trước đó, thay nội dung password secret
file **không tự đổi** password của user `elastic`. Khi đó hãy reset password bằng
Security API hoặc `elasticsearch-reset-password`; không xóa data volume chỉ để
đổi password.

## 4. Xuất public CA và kiểm tra HTTPS

Python client trên host chỉ cần public CA; không xuất CA private key hoặc node
private key khỏi Docker volume.

```powershell
$caFile = Join-Path $secureDir "ca.crt"
docker compose `
  --env-file $composeEnv `
  -f src/semantic_pipeline/docker-compose.elasticsearch.secure.yml `
  cp cert_setup:/usr/share/elasticsearch/config/certs/ca/ca.crt $caFile

$env:ELASTICSEARCH_URL = "https://127.0.0.1:19200"
$env:ELASTICSEARCH_USERNAME = "elastic"
$env:ELASTICSEARCH_PASSWORD = [System.IO.File]::ReadAllText($passwordFile)
$env:ELASTICSEARCH_CA_CERT = $caFile

python src/semantic_pipeline/elasticsearch_backend.py health
```

Client dùng `ca_certs` để xác minh certificate và `basic_auth` để xác thực. Code
không có đường tắt `verify_certs=False`, đồng thời từ chối gửi credential qua
HTTP thuần.

## 5. Bootstrap index secure riêng

Secure cluster là cluster mới nên cần ingest metadata và tạo alias:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v6 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json
```

Sau đó chạy benchmark hoặc acceptance như bình thường; các biến môi trường ở
mục 4 khiến tất cả client Elasticsearch kết nối bằng HTTPS.

## 6. Tạo user chỉ-đọc cho API

Không chạy web API lâu dài bằng superuser `elastic`. Dùng superuser một lần để
tạo role và user riêng. Trước tiên sinh password ứng dụng độc lập:

```powershell
$appPasswordBytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($appPasswordBytes)
$rng.Dispose()
$appPassword = [Convert]::ToBase64String($appPasswordBytes)
$appPasswordFile = Join-Path $secureDir "semantic_search_password.txt"
[System.IO.File]::WriteAllText(
  $appPasswordFile,
  $appPassword,
  [System.Text.UTF8Encoding]::new($false)
)
$env:SEMANTIC_APP_PASSWORD_FILE = $appPasswordFile

@'
import os
from pathlib import Path

from src.semantic_pipeline.elasticsearch_backend import create_client

password = Path(os.environ["SEMANTIC_APP_PASSWORD_FILE"]).read_text(
    encoding="utf-8"
).rstrip("\r\n")
client = create_client()
client.security.put_role(
    name="semantic_search_app",
    cluster=["monitor"],
    indices=[
        {
            "names": ["semantic_frames", "semantic_frames_v*"],
            "privileges": ["read", "view_index_metadata"],
            "allow_restricted_indices": False,
        }
    ],
)
client.security.put_user(
    username="semantic_search",
    password=password,
    roles=["semantic_search_app"],
    full_name="Semantic Search API",
    enabled=True,
)
client.close()
print("Created/updated least-privilege user: semantic_search")
'@ | python -

Remove-Item Env:SEMANTIC_APP_PASSWORD_FILE
```

Script dùng chính Python client đã xác minh CA để gọi Security API; không cần
`-SkipCertificateCheck`. Sau đó cấu hình API bằng user mới:

```powershell
$env:SEMANTIC_SEARCH_BACKEND = "elasticsearch"
$env:ELASTICSEARCH_URL = "https://127.0.0.1:19200"
$env:ELASTICSEARCH_USERNAME = "semantic_search"
$env:ELASTICSEARCH_PASSWORD = [System.IO.File]::ReadAllText($appPasswordFile)
$env:ELASTICSEARCH_CA_CERT = $caFile
$env:ELASTICSEARCH_INDEX = "semantic_frames"
python src/semantic_pipeline/server.py
```

Không commit password user ứng dụng. Với staging thật, ưu tiên secret manager của
nền tảng thay vì biến môi trường đặt thủ công.

## 7. Test

Unit test client và validation Compose không khởi động stack:

```powershell
python -m pytest -q `
  src/semantic_pipeline/tests/test_elasticsearch_backend.py `
  src/semantic_pipeline/tests/test_elasticsearch_secure_config.py
```

Live read-only integration sau khi bootstrap secure index:

```powershell
$env:RUN_ELASTICSEARCH_INTEGRATION = "1"
python -m pytest -q `
  src/semantic_pipeline/tests/test_elasticsearch_backend.py
Remove-Item Env:RUN_ELASTICSEARCH_INTEGRATION
```

Xóa các biến chứa credential khỏi shell sau khi dùng:

```powershell
Remove-Item Env:ELASTICSEARCH_USERNAME -ErrorAction SilentlyContinue
Remove-Item Env:ELASTICSEARCH_PASSWORD -ErrorAction SilentlyContinue
Remove-Item Env:ELASTICSEARCH_CA_CERT -ErrorAction SilentlyContinue
Remove-Item Env:ELASTICSEARCH_URL -ErrorAction SilentlyContinue
```

## 8. Dừng stack

```powershell
docker compose `
  --env-file $composeEnv `
  -f src/semantic_pipeline/docker-compose.elasticsearch.secure.yml `
  down
```

Lệnh trên giữ lại CA, certificate và data volumes. Không thêm `-v` nếu chưa có
backup: tùy chọn đó xóa cả index và certificate material của secure cluster.

## Giới hạn còn lại trước production

- Đây vẫn là single-node: không HA, không replica/failover và chưa kiểm thử
  disaster recovery.
- Chưa có snapshot repository, restore drill, monitoring/alerting, audit-log
  retention hoặc certificate-rotation automation.
- Port chỉ bind loopback. Staging nhiều máy cần firewall/private network hoặc
  ingress được quản lý; không đổi thành `0.0.0.0` một cách trực tiếp.
- CA tự ký phù hợp local-secure. Staging/production nên dùng PKI nội bộ hoặc CA
  được tổ chức quản lý, có quy trình cấp và thu hồi certificate.
- Built-in `elastic` là bootstrap superuser, không phải credential của ứng dụng.
- Docker Compose và `mem_limit` không thay thế capacity planning, shard sizing,
  rolling upgrade, backup hay orchestrator production.

Tham khảo chính thức:

- [Elastic: Set up HTTPS](https://www.elastic.co/docs/deploy-manage/security/set-up-basic-security-plus-https/)
- [Elastic: Set up transport TLS](https://www.elastic.co/docs/deploy-manage/security/set-up-basic-security)
- [Elastic Python client: Connecting](https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting)
- [Elastic Python client: TLS configuration](https://www.elastic.co/docs/reference/elasticsearch/clients/python/configuration)
