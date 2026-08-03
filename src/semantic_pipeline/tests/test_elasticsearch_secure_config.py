"""Static and Docker-Compose validation for the isolated secure ES profile."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SEMANTIC_DIR = Path(__file__).resolve().parent.parent
SECURE_COMPOSE = SEMANTIC_DIR / "docker-compose.elasticsearch.secure.yml"
LOCAL_COMPOSE = SEMANTIC_DIR / "docker-compose.elasticsearch.yml"
ENV_TEMPLATE = SEMANTIC_DIR / ".env.elasticsearch.secure.example"


def test_secure_profile_is_isolated_and_has_no_literal_secret():
    secure = SECURE_COMPOSE.read_text(encoding="utf-8")
    local = LOCAL_COMPOSE.read_text(encoding="utf-8")
    env_template = ENV_TEMPLATE.read_text(encoding="utf-8")

    assert "name: semantic_pipeline_secure" in secure
    assert '127.0.0.1:${ELASTICSEARCH_SECURE_PORT:-19200}:9200' in secure
    assert "secure_elasticsearch_data" in secure
    assert "semantic_elasticsearch_data" not in secure

    assert 'xpack.security.enabled: "true"' in secure
    assert 'xpack.security.http.ssl.enabled: "true"' in secure
    assert 'xpack.security.transport.ssl.enabled: "true"' in secure
    assert "ELASTIC_PASSWORD_FILE: /run/secrets/elastic_password" in secure
    assert not re.search(r"(?m)^\s*ELASTIC_PASSWORD\s*[:=]", secure)
    assert not re.search(r"(?m)^ELASTIC_PASSWORD=", env_template)

    # The original loopback/no-auth developer profile remains unchanged.
    assert "xpack.security.enabled=false" in local
    assert '"127.0.0.1:9200:9200"' in local


def test_certificate_bootstrap_covers_host_and_container_names():
    secure = SECURE_COMPOSE.read_text(encoding="utf-8")

    assert "bin/elasticsearch-certutil ca" in secure
    assert "bin/elasticsearch-certutil cert" in secure
    assert "      - elasticsearch" in secure
    assert "      - localhost" in secure
    assert "      - 127.0.0.1" in secure
    assert "secure_elasticsearch_ca:/usr/share/elasticsearch/config/cert-authority" in secure

    elasticsearch_section = secure.split("  elasticsearch:\n", maxsplit=1)[1]
    assert "/usr/share/elasticsearch/config/cert-authority" not in elasticsearch_section


def _load_compose_config(tmp_path: Path) -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable; static secure-config tests still ran")

    password_file = tmp_path / "dummy-password.txt"
    password_file.write_text("dummy-config-validation-password", encoding="utf-8")
    env = os.environ.copy()
    env["ELASTIC_PASSWORD_SECRET_FILE"] = str(password_file.resolve())
    env["ELASTICSEARCH_SECURE_PORT"] = "19200"

    result = subprocess.run(
        [
            docker,
            "compose",
            "-f",
            str(SECURE_COMPOSE),
            "config",
            "--format",
            "json",
        ],
        cwd=SEMANTIC_DIR.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if result.returncode != 0 and "compose" in result.stderr.lower():
        pytest.skip(f"Docker Compose is unavailable: {result.stderr.strip()}")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_docker_compose_resolves_secure_profile_without_starting_it(tmp_path):
    config = _load_compose_config(tmp_path)
    assert config["name"] == "semantic_pipeline_secure"

    setup = config["services"]["cert_setup"]
    node = config["services"]["elasticsearch"]
    node_env = node["environment"]

    assert setup["network_mode"] == "none"
    assert node["depends_on"]["cert_setup"]["condition"] == (
        "service_completed_successfully"
    )
    assert node_env["xpack.security.enabled"] == "true"
    assert node_env["xpack.security.http.ssl.enabled"] == "true"
    assert node_env["xpack.security.transport.ssl.enabled"] == "true"
    assert node_env["ELASTIC_PASSWORD_FILE"] == "/run/secrets/elastic_password"
    assert "ELASTIC_PASSWORD" not in node_env

    assert node["ports"] == [
        {
            "mode": "ingress",
            "host_ip": "127.0.0.1",
            "target": 9200,
            "published": "19200",
            "protocol": "tcp",
        }
    ]
    node_mount_targets = {volume["target"] for volume in node["volumes"]}
    assert "/usr/share/elasticsearch/config/cert-authority" not in node_mount_targets
    cert_mount = next(
        volume
        for volume in node["volumes"]
        if volume["target"] == "/usr/share/elasticsearch/config/certs"
    )
    assert cert_mount["read_only"] is True
