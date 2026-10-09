"""Seekable file-like object over HTTP Range requests, so that ``zipfile`` can list
and extract single members of a huge remote archive (ZIP64 is handled by zipfile).

Usage:
    with RemoteFile(url) as rf, zipfile.ZipFile(rf) as z:
        names = z.namelist(); data = z.read(names[0])
    print(rf.bytes_transferred)

The (redirecting) URL is resolved lazily; the signed redirect target is reused until
it fails (403/expired) and is then re-resolved from the original URL.
CLI:  python remote_zip.py list <url>
      python remote_zip.py fetch <url> <outdir> <member> [<member> ...] [--manifest f.json]
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

UA = "fly-eye-multitask-research"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, retries: int = 6, log=None):
        self.url, self.block, self.retries = url, block, retries
        self._signed = None
        self._pos = 0
        self.bytes_transferred = 0
        self.n_requests = 0
        self._log = log
        self._cache_start, self._cache = 0, b""
        self.size = self._probe()

    # -- http ------------------------------------------------------------------
    def _open(self, url, headers):
        req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
        return urllib.request.urlopen(req, timeout=120)

    def _resolve(self):
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        try:
            r = opener.open(req, timeout=60)
            self._signed = self.url
            r.close()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                self._signed = e.headers["Location"]
            else:
                raise
        return self._signed

    def _get(self, start, end):  # inclusive
        err = None
        for attempt in range(self.retries):
            try:
                if self._signed is None:
                    self._resolve()
                with self._open(self._signed, {"Range": f"bytes={start}-{end}"}) as r:
                    if r.status != 206:
                        raise IOError(f"expected 206, got {r.status}")
                    cr = r.headers.get("Content-Range", "")
                    data = r.read()
                self.n_requests += 1
                self.bytes_transferred += len(data)
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end-start+1}")
                return data, cr
            except Exception as e:  # noqa
                err = e
                self._signed = None  # force re-resolve (expired signature etc.)
                time.sleep(min(2 ** attempt, 30))
        raise IOError(f"range {start}-{end} failed: {err}")

    def _probe(self):
        data, cr = self._get(0, 0)
        return int(cr.split("/")[-1])

    # -- file API --------------------------------------------------------------
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self._pos

    def seek(self, off, whence=0):
        self._pos = {0: off, 1: self._pos + off, 2: self.size + off}[whence]
        return self._pos

    def read(self, n=-1):
        if n < 0 or self._pos + n > self.size:
            n = self.size - self._pos
        if n <= 0:
            return b""
        s, e = self._pos, self._pos + n
        cs, c = self._cache_start, self._cache
        if cs <= s and e <= cs + len(c):
            out = c[s - cs:e - cs]
        elif n >= self.block:
            out = b"".join(self._get(a, min(a + (8 << 20), e) - 1)[0] for a in range(s, e, 8 << 20))
        else:
            fs = s
            fe = min(self.size, max(e, s + self.block))
            self._cache, self._cache_start = self._get(fs, fe - 1)[0], fs
            out = self._cache[:n]
        self._pos = e
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def fetch_members(url, members, outdir, log=print):
    """Extract members to outdir; return manifest records."""
    outdir = Path(outdir); recs = []
    with RemoteFile(url) as rf, zipfile.ZipFile(rf) as z:
        log(f"directory read: {rf.bytes_transferred} bytes transferred so far, {len(z.namelist())} entries")
        for m in members:
            zi = z.getinfo(m)
            before = rf.bytes_transferred
            data = z.read(zi)  # zipfile verifies CRC32
            dest = outdir / m
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            recs.append(dict(member=m, uncompressed_size=zi.file_size, compressed_size=zi.compress_size,
                             crc32=f"{zi.CRC:08x}", sha256=hashlib.sha256(data).hexdigest(),
                             local_path=str(dest), bytes_transferred=rf.bytes_transferred - before,
                             downloaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        total = rf.bytes_transferred
    log(f"fetched {len(members)} members, total transferred (incl. directory) {total} bytes")
    return recs, total


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        with RemoteFile(sys.argv[2]) as rf, zipfile.ZipFile(rf) as z:
            for i in z.infolist():
                print(i.file_size, i.filename)
            print("transferred", rf.bytes_transferred, file=sys.stderr)
    elif cmd == "fetch":
        recs, tot = fetch_members(sys.argv[2], sys.argv[4:], sys.argv[3])
        print(json.dumps(dict(total=tot, records=recs), indent=1))
