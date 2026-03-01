import os
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit


def _first_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def get_httpx_proxies(scope: Optional[str] = None, include_global: bool = True) -> Optional[Dict[str, str]]:
    prefix = f"{scope.upper()}_" if scope else ""
    http_keys = [f"{prefix}HTTP_PROXY", f"{prefix}PROXY"] if prefix else []
    https_keys = [f"{prefix}HTTPS_PROXY", f"{prefix}PROXY"] if prefix else []
    if include_global:
        http_keys.extend(["HTTP_PROXY", "ALL_PROXY"])
        https_keys.extend(["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"])

    http_proxy = _first_env(*http_keys)
    https_proxy = _first_env(*https_keys)

    proxies: Dict[str, str] = {}
    if http_proxy:
        proxies["http://"] = http_proxy
    if https_proxy:
        proxies["https://"] = https_proxy

    return proxies or None


def get_playwright_proxy(scope: Optional[str] = None, include_global: bool = True) -> Optional[Dict[str, str]]:
    prefix = f"{scope.upper()}_" if scope else ""
    keys = []
    if prefix:
        keys.extend([f"{prefix}HTTPS_PROXY", f"{prefix}HTTP_PROXY", f"{prefix}PROXY"])
    if include_global:
        keys.extend(["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"])

    proxy_url = _first_env(*keys)
    if not proxy_url:
        return None

    parsed = urlsplit(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return None

    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server = f"{server}:{parsed.port}"

    proxy: Dict[str, str] = {"server": server}
    if parsed.username:
        proxy["username"] = unquote(parsed.username)
    if parsed.password:
        proxy["password"] = unquote(parsed.password)
    return proxy
