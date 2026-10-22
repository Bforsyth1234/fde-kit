"""Encryption for everything a durable run writes outside the worker process.

A durable run leaves tenant content in two places the application doesn't control:

- Temporal's event history. Every workflow input, activity input and activity result is a
  payload in the history: the prompt, every retrieved passage, every tool argument, the answer.
  The server stores them, the UI on :8233 renders them, and any client with the address can
  `fetch_history()` them. Base64 is the only thing between a reader and the text.
- The tool journal. Its `content` column is the tool's output, in plaintext, with no RLS and no
  expiry, so a worker that retries can serve it.

Both get the same fix: AES-256-GCM under a key the server never sees.

- `PayloadCipher` is the primitive: `encrypt(bytes) -> bytes` with a 12-byte random nonce and
  the key id as associated data, `decrypt` the reverse. Output layout: `key_id (8 bytes) ||
  nonce (12) || ciphertext+tag`. The key id lets a deployment rotate keys: encrypt with the
  current one, decrypt with any key still in `FDE_PAYLOAD_KEYS`.
- `EncryptionCodec` is Temporal's `PayloadCodec` over it. The worker and every client connect
  with it, so the server only ever holds ciphertext; it is applied after serialisation and
  before transport, which is the point where Temporal lets you do this.
- The journals call `encrypt_text` / `decrypt_text` on `content`.

Keys: `FDE_PAYLOAD_KEY` is one base64 key (32 bytes), or `FDE_PAYLOAD_KEYS` is `id=base64`
pairs separated by commas, first one current. Without a key, nothing is encrypted and a
tenant-bound run is refused (`start_run(tenant_id=...)` and the worker both check), because a
run that names a tenant is a run whose content must not end up in a shared history.

What this doesn't do: hide metadata. Workflow ids, activity names, timings, attempt counts,
and the `tenant.id` search attribute stay visible, which is what operators need. The Temporal
UI shows `[encrypted]` for payloads unless a codec server is configured for it (see README).
"""

from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from temporalio.api.common.v1 import Payload
from temporalio.converter import DataConverter, PayloadCodec

ENCODING = b"binary/encrypted"
KEY_ID_BYTES = 8
NONCE_BYTES = 12


class PayloadKeyError(RuntimeError):
    pass


@dataclass
class PayloadCipher:
    keys: dict[str, bytes]
    current: str

    @classmethod
    def from_settings(cls, single: str | None, multi: str | None) -> PayloadCipher | None:
        """Build from `FDE_PAYLOAD_KEY` / `FDE_PAYLOAD_KEYS`. None when neither is set."""
        keys: dict[str, bytes] = {}
        current = ""
        if multi:
            for item in multi.split(","):
                kid, _, b64 = item.strip().partition("=")
                if not kid or not b64:
                    raise PayloadKeyError("FDE_PAYLOAD_KEYS entries are id=base64key")
                keys[kid] = _decode_key(b64)
                current = current or kid
        elif single:
            keys["k1"] = _decode_key(single)
            current = "k1"
        else:
            return None
        return cls(keys=keys, current=current)

    @classmethod
    def generate(cls) -> PayloadCipher:
        return cls(keys={"k1": secrets.token_bytes(32)}, current="k1")

    def export_current(self) -> str:
        return base64.b64encode(self.keys[self.current]).decode()

    def encrypt(self, data: bytes) -> bytes:
        kid = self.current.encode().ljust(KEY_ID_BYTES, b"\0")[:KEY_ID_BYTES]
        nonce = os.urandom(NONCE_BYTES)
        ct = AESGCM(self.keys[self.current]).encrypt(nonce, data, kid)
        return kid + nonce + ct

    def decrypt(self, blob: bytes) -> bytes:
        if len(blob) < KEY_ID_BYTES + NONCE_BYTES + 16:
            raise PayloadKeyError("ciphertext too short")
        kid_raw = blob[:KEY_ID_BYTES]
        kid = kid_raw.rstrip(b"\0").decode(errors="replace")
        key = self.keys.get(kid)
        if key is None:
            raise PayloadKeyError(f"no key {kid!r} in FDE_PAYLOAD_KEYS; cannot decrypt")
        nonce = blob[KEY_ID_BYTES : KEY_ID_BYTES + NONCE_BYTES]
        try:
            return AESGCM(key).decrypt(nonce, blob[KEY_ID_BYTES + NONCE_BYTES :], kid_raw)
        except InvalidTag as exc:
            raise PayloadKeyError("payload failed authentication (wrong key or tampered)") from exc

    # Text helpers for the journal's content column: base64 so the column stays TEXT.

    def encrypt_text(self, text: str) -> str:
        return "enc:" + base64.b64encode(self.encrypt(text.encode())).decode()

    def decrypt_text(self, stored: str) -> str:
        if not stored.startswith("enc:"):
            raise PayloadKeyError("journal row is not encrypted but a payload key is configured")
        return self.decrypt(base64.b64decode(stored[4:])).decode()


def _decode_key(b64: str) -> bytes:
    try:
        raw = base64.b64decode(b64.strip(), validate=True)
    except Exception as exc:
        raise PayloadKeyError("payload key must be base64") from exc
    if len(raw) != 32:
        raise PayloadKeyError(f"payload key must decode to 32 bytes, got {len(raw)}")
    return raw


@dataclass
class EncryptionCodec(PayloadCodec):
    cipher: PayloadCipher
    encoded: int = field(default=0, init=False)

    async def encode(self, payloads):  # type: ignore[override]
        out = []
        for p in payloads:
            out.append(
                Payload(
                    metadata={"encoding": ENCODING},
                    data=self.cipher.encrypt(p.SerializeToString()),
                )
            )
            self.encoded += 1
        return out

    async def decode(self, payloads):  # type: ignore[override]
        out = []
        for p in payloads:
            if p.metadata.get("encoding") != ENCODING:
                out.append(p)  # a pre-encryption history, or a server-generated payload
                continue
            inner = Payload()
            inner.ParseFromString(self.cipher.decrypt(p.data))
            out.append(inner)
        return out


def data_converter(cipher: PayloadCipher | None) -> DataConverter:
    """Temporal's default converter, with encryption added when a key is configured."""
    base = DataConverter.default
    if cipher is None:
        return base
    return DataConverter(
        payload_converter_class=base.payload_converter_class,
        payload_codec=EncryptionCodec(cipher),
        failure_converter_class=base.failure_converter_class,
    )
