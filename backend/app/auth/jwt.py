"""Supabase Auth access-token verification (Phase 7).

Follows Supabase's guidance for asymmetric JWT signing keys: fetch the project's public keys
from ``<SUPABASE_URL>/auth/v1/.well-known/jwks.json`` (cached), verify the signature (RS256 /
ES256), ``iss = <SUPABASE_URL>/auth/v1``, ``aud = authenticated``, ``exp`` and require ``sub``.
Only ``role = authenticated`` end-user tokens are accepted (``anon`` / ``service_role`` are
refused). Legacy HS256 shared-secret projects work only when ``SUPABASE_JWT_SECRET`` is set
explicitly. Construction performs no network I/O; keys are fetched on first verification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jwt

from app.auth.errors import AuthNotConfiguredError, InvalidTokenError

ASYMMETRIC = ("RS256", "ES256")
LEEWAY_SECONDS = 30


class JwtVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str | None,
        hs256_secret: str | None = None,
        cache_seconds: int = 300,
        key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._secret = hs256_secret
        self._jwks_url = jwks_url
        self._cache_seconds = cache_seconds
        self._client: jwt.PyJWKClient | None = None
        self._key_resolver = key_resolver  # tests: token -> signing key

    @classmethod
    def from_settings(cls, settings: Any) -> JwtVerifier:
        if not settings.supabase_url:
            raise AuthNotConfiguredError()
        base = settings.supabase_url.rstrip("/")
        secret = (
            settings.supabase_jwt_secret.get_secret_value()
            if settings.supabase_jwt_secret
            else None
        )
        return cls(
            issuer=f"{base}/auth/v1",
            audience=settings.supabase_jwt_audience,
            jwks_url=f"{base}/auth/v1/.well-known/jwks.json",
            hs256_secret=secret,
            cache_seconds=settings.auth_jwks_cache_seconds,
        )

    def _signing_key(self, token: str, alg: str) -> Any:
        if alg == "HS256":
            if not self._secret:
                raise InvalidTokenError()
            return self._secret
        if self._key_resolver is not None:
            return self._key_resolver(token)
        if self._client is None:
            self._client = jwt.PyJWKClient(
                self._jwks_url or "", cache_keys=True, lifespan=self._cache_seconds, timeout=5
            )
        return self._client.get_signing_key_from_jwt(token).key

    def verify(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
            alg = header.get("alg")
            allowed = (*ASYMMETRIC, "HS256") if self._secret else ASYMMETRIC
            if alg not in allowed:  # never "none", never an unexpected algorithm
                raise InvalidTokenError()
            claims = jwt.decode(
                token,
                self._signing_key(token, alg),
                algorithms=[alg],
                audience=self._audience,
                issuer=self._issuer,
                leeway=LEEWAY_SECONDS,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except InvalidTokenError:
            raise
        except (jwt.PyJWTError, ValueError, TypeError):
            raise InvalidTokenError() from None
        if claims.get("role") != "authenticated" or not isinstance(claims.get("sub"), str):
            raise InvalidTokenError()
        if not 1 <= len(claims["sub"]) <= 128:
            raise InvalidTokenError()
        return claims
