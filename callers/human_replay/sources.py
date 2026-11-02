"""Where the replayed human speech comes from, its license, and how it is downloaded.

Two sources (checked 2026-11-02):

* LibriSpeech test-clean: read audiobook speech, 40 speakers (20 F / 20 M), gender in SPEAKERS.TXT.
  CC BY 4.0. 346 MB tar.gz. https://www.openslr.org/12/
* EdAcc (Edinburgh International Accents of English Corpus): relaxed two-person conversations
  over video calls, ~100 speakers, 40+ accents, gender + accent per speaker. CC BY-SA 4.0
  (https://doi.org/10.7488/ds/7914, "released on our website under CC-BY-SA"). We read the
  Hugging Face copy (https://huggingface.co/datasets/edinburghcstr/edacc, parquet, 32 kHz WAV
  per segment) with HTTP range requests: only the parquet footers (~2 MB in all) and the row
  groups of the conversations we pick (~25-50 MB each), never the 7 GB dataset.

Not used: Common Voice (CC0, but since Oct 2025 only through Mozilla Data Collective with an
account: https://commonvoice.mozilla.org/en/faq), VCTK (CC BY 4.0, but one ~11 GB zip and
2-4 s sentences), FLEURS (CC BY 4.0, no speaker ids), VoxPopuli en (CC0, but one 936 MB test
parquet), People's Speech (no speaker metadata).

Everything lands under data/human_replay/ (gitignored). Downloaded files are data only: tar
members are extracted with tarfile's "data" filter, parquet is parsed by pyarrow, audio by
libsndfile. Nothing downloaded is executed.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "human_replay"
RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    license: str
    license_url: str
    style: str               # "read" or "conversational"
    homepage: str
    attribution: str         # paste-ready line for the repo README


SOURCES: dict[str, Source] = {
    "librispeech": Source(
        key="librispeech",
        name="LibriSpeech ASR corpus, test-clean",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        style="read",
        homepage="https://www.openslr.org/12/",
        attribution=(
            "LibriSpeech (test-clean) by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, "
            "\"Librispeech: an ASR corpus based on public domain audio books\", ICASSP 2015, "
            "https://www.openslr.org/12/, licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "The audio comes from LibriVox volunteer recordings. Changes: we cut 6-12 s clips, resampled them "
            "and normalized their level, and played them over phone calls."
        ),
    ),
    "edacc": Source(
        key="edacc",
        name="EdAcc: The Edinburgh International Accents of English Corpus",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        style="conversational",
        homepage="https://groups.inf.ed.ac.uk/edacc/",
        attribution=(
            "EdAcc: The Edinburgh International Accents of English Corpus by Ramon Sanabria, Nina Markl, "
            "Andrea Carmantini, Ondrej Klejch, Peter Bell and Nikolay Bogoychev, University of Edinburgh "
            "(https://doi.org/10.7488/ds/7914; ICASSP 2023, https://arxiv.org/abs/2303.18110), licensed "
            "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), read from the Hugging Face copy "
            "edinburghcstr/edacc. Changes: we cut segments to at most 12 s, resampled them and normalized their "
            "level, and played them over phone calls. Any clips we share are under CC BY-SA 4.0."
        ),
    ),
}
ALLOWED_LICENSES = {s.license for s in SOURCES.values()}


# --------------------------------------------------------------------------------------------
# HTTP helpers

def _client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0),
                        headers={"User-Agent": "emberz-human-replay/1.0"})


class HttpRangeFile:
    """Seekable read-only file over HTTP Range requests, so pyarrow reads only the bytes it needs
    (the parquet footer, then the column chunks of the row groups we ask for)."""

    def __init__(self, url: str, client: Optional[httpx.Client] = None):
        self.client = client or _client()
        r = self.client.head(url)
        r.raise_for_status()
        self.url = str(r.url)
        self.size = int(r.headers["content-length"])
        self.pos = 0
        self.fetched = 0

    # file protocol used by pyarrow.PythonFile
    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        r = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        r.raise_for_status()
        data = r.content
        if r.status_code != 206 and len(data) != n:
            raise IOError(f"server ignored the Range header for {self.url}")
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------------
# LibriSpeech

LIBRI_MIRRORS = [
    "https://www.openslr.org/resources/12/{f}",
    "https://openslr.elda.org/resources/12/{f}",
    "https://openslr.trmal.net/resources/12/{f}",
]
LIBRI_SUBSET_MB = {"test-clean": 346, "test-other": 328, "dev-clean": 337, "dev-other": 314}


def libri_dir() -> Path:
    return RAW_DIR / "librispeech"


def download_librispeech(subset: str = "test-clean", keep_archive: bool = False, log=print) -> Path:
    """Stream the tar.gz, check its md5 against openslr's md5sum.txt, extract (data filter only)."""
    out = libri_dir()
    marker = out / f".{subset}.done"
    if marker.exists():
        log(f"librispeech {subset}: already extracted in {out}")
        return out
    fname = f"{subset}.tar.gz"
    dl = RAW_DIR / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    archive = dl / fname
    with _client() as c:
        md5s: dict[str, str] = {}
        for m in LIBRI_MIRRORS:
            try:
                r = c.get(m.format(f="md5sum.txt"))
                if r.status_code == 200:
                    md5s = {ln.split()[1]: ln.split()[0] for ln in r.text.splitlines() if len(ln.split()) == 2}
                    break
            except httpx.HTTPError:
                continue
        want = md5s.get(fname)
        if not archive.exists() or (want and _md5(archive) != want):
            last_err: Exception | None = None
            for m in LIBRI_MIRRORS:
                url = m.format(f=fname)
                try:
                    log(f"downloading {url} (~{LIBRI_SUBSET_MB.get(subset, '?')} MB)")
                    tmp = archive.with_suffix(".part")
                    with c.stream("GET", url) as r:
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
                    tmp.rename(archive)
                    last_err = None
                    break
                except httpx.HTTPError as e:
                    last_err = e
                    log(f"  mirror failed: {e}")
            if last_err:
                raise SystemExit(f"could not download {fname}: {last_err}")
    if want:
        got = _md5(archive)
        if got != want:
            raise SystemExit(f"{fname}: md5 {got} != {want} (openslr md5sum.txt)")
        log(f"md5 ok ({want})")
    else:
        log("warning: no md5 from openslr; archive not verified")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and (m.name.endswith((".flac", ".trans.txt")) or m.name.endswith("SPEAKERS.TXT"))]
        tf.extractall(out, members=members, filter="data")
    marker.write_text(json.dumps({"subset": subset, "md5": want}), encoding="utf-8")
    if not keep_archive:
        archive.unlink()
    log(f"librispeech {subset}: extracted to {out}")
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_libri_speakers(root: Path) -> dict[str, dict[str, str]]:
    """SPEAKERS.TXT: ';'-comment lines, then 'ID | SEX | SUBSET | MINUTES | NAME'."""
    out: dict[str, dict[str, str]] = {}
    for p in root.rglob("SPEAKERS.TXT"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(";") or "|" not in line:
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) >= 3 and parts[0].isdigit():
                out[parts[0]] = {"sex": parts[1].upper(), "subset": parts[2]}
    return out


def libri_segments(root: Path, subsets: tuple[str, ...] = ("test-clean",)) -> Iterator[dict[str, Any]]:
    """One dict per utterance: speaker, gender, path, duration, text."""
    import soundfile as sf

    spk = read_libri_speakers(root)
    for subset in subsets:
        base = root / "LibriSpeech" / subset
        if not base.exists():
            continue
        for trans in sorted(base.rglob("*.trans.txt")):
            texts = dict(line.split(" ", 1) for line in trans.read_text(encoding="utf-8").splitlines() if " " in line)
            for flac in sorted(trans.parent.glob("*.flac")):
                utt = flac.stem
                speaker = utt.split("-")[0]
                sex = spk.get(speaker, {}).get("sex", "")
                yield {
                    "source": "librispeech",
                    "speaker": f"librispeech:{speaker}",
                    "gender": {"F": "female", "M": "male"}.get(sex, "unknown"),
                    "accent": "",
                    "path": str(flac),
                    "duration": round(sf.info(str(flac)).duration, 3),
                    "text": texts.get(utt, "").strip(),
                    "source_file": f"{subset}/{utt}.flac",
                }


# --------------------------------------------------------------------------------------------
# EdAcc (Hugging Face parquet, range reads)

EDACC_REPO = "edinburghcstr/edacc"
EDACC_TREE = f"https://huggingface.co/api/datasets/{EDACC_REPO}/tree/main/data"
EDACC_FILE = f"https://huggingface.co/datasets/{EDACC_REPO}/resolve/main/" + "{path}"


def edacc_dir() -> Path:
    return RAW_DIR / "edacc"


def _need_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        raise SystemExit("EdAcc needs pyarrow (pip install pyarrow, in your own venv; the shared .venv lacks it). "
                         "Or build from LibriSpeech only: --sources librispeech")
    import pyarrow.parquet as pq
    return pq


def survey_edacc(log=print) -> list[dict[str, Any]]:
    """Read only the parquet footers + the small metadata columns (~2 MB in all). One row per
    (shard, row group): its byte size and the rows per speaker with their gender/accent/l1."""
    pq = _need_pyarrow()
    cache = edacc_dir() / "survey.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    with _client() as c:
        files = sorted(f["path"] for f in c.get(EDACC_TREE).json() if f["path"].endswith(".parquet"))
        total = 0
        for path in files:
            hf = HttpRangeFile(EDACC_FILE.format(path=path), c)
            pf = pq.ParquetFile(hf)
            cols = ["speaker", "gender", "accent", "l1"]
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=cols).to_pylist()
                md = pf.metadata.row_group(rg)
                size = sum(md.column(i).total_compressed_size for i in range(md.num_columns))
                speakers: dict[str, dict[str, Any]] = {}
                for r in t:
                    s = speakers.setdefault(r["speaker"], {"gender": r["gender"], "accent": r["accent"], "l1": r["l1"], "rows": 0})
                    s["rows"] += 1
                out.append({"path": path, "row_group": rg, "bytes": size, "speakers": speakers})
            total += hf.fetched
        log(f"edacc survey: {len(files)} shards, {len(out)} row groups, {total / 1e6:.1f} MB of metadata read")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def conversation_of(speaker: str) -> str:
    """'EDACC-C35-A' -> 'EDACC-C35'."""
    return speaker.rsplit("-", 1)[0]


def plan_edacc_row_groups(survey: list[dict[str, Any]], max_bytes: int, min_rows_per_speaker: int = 25,
                          max_rg_bytes: int = 45_000_000) -> list[dict[str, Any]]:
    """Pick at most one row group per conversation, covering both speakers, within max_bytes.
    Row groups over max_rg_bytes are skipped (cost per speaker). Order: new accents first
    (accent diversity), then the gender we have fewer of, then smaller row groups."""
    best: dict[str, dict[str, Any]] = {}
    for rg in survey:
        convs: dict[str, list[str]] = {}
        for spk, meta in rg["speakers"].items():
            if meta["rows"] >= min_rows_per_speaker:
                convs.setdefault(conversation_of(spk), []).append(spk)
        for conv, spks in convs.items():
            if len(spks) < 2:
                continue
            if rg["bytes"] > max_rg_bytes:
                continue
            cur = best.get(conv)
            if cur is None or rg["bytes"] < cur["bytes"]:
                best[conv] = {**rg, "conversation": conv}
    seen_accents: set[str] = set()
    gender_count = {"female": 0, "male": 0}
    chosen: list[dict[str, Any]] = []
    used = 0
    pool = sorted(best.values(), key=lambda r: (r["bytes"], r["conversation"]))
    while pool:
        def score(r: dict[str, Any]) -> tuple:
            spk = [m for s, m in r["speakers"].items() if conversation_of(s) == r["conversation"]]
            new_acc = len({m["accent"] for m in spk} - seen_accents)
            g = [m["gender"] for m in spk]
            short = min(gender_count, key=gender_count.get)
            return (-new_acc, -g.count(short), r["bytes"], r["conversation"])
        pool.sort(key=score)
        r = pool.pop(0)
        if used + r["bytes"] > max_bytes:
            continue
        used += r["bytes"]
        chosen.append(r)
        for s, m in r["speakers"].items():
            if conversation_of(s) == r["conversation"]:
                seen_accents.add(m["accent"])
                if m["gender"] in gender_count:
                    gender_count[m["gender"]] += 1
    return chosen


def download_edacc(max_mb: float = 550, log=print) -> Path:
    """Fetch the planned row groups and write each segment of the chosen conversation as a WAV,
    plus segments.jsonl with the metadata."""
    pq = _need_pyarrow()
    import soundfile as sf

    out = edacc_dir()
    seg_file = out / "segments.jsonl"
    done_file = out / "fetched.json"
    done = json.loads(done_file.read_text()) if done_file.exists() else []
    plan = plan_edacc_row_groups(survey_edacc(log), int(max_mb * 1e6))
    log(f"edacc plan: {len(plan)} conversations, {sum(r['bytes'] for r in plan) / 1e6:.0f} MB")
    with _client() as c:
        for r in plan:
            key = f"{r['path']}#{r['row_group']}#{r['conversation']}"
            if key in done:
                continue
            hf = HttpRangeFile(EDACC_FILE.format(path=r["path"]), c)
            pf = pq.ParquetFile(hf)
            rows = pf.read_row_group(r["row_group"]).to_pylist()
            conv_dir = out / r["conversation"]
            conv_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(seg_file, "a", encoding="utf-8") as segf:
                for i, row in enumerate(rows):
                    if conversation_of(row["speaker"]) != r["conversation"]:
                        continue
                    data = row["audio"]["bytes"]
                    info = sf.info(io.BytesIO(data))
                    name = Path(row["audio"].get("path") or f"{row['speaker']}-{r['row_group']}-{i}.wav").name
                    name = "".join(ch for ch in name if ch.isalnum() or ch in "-_.") or f"seg{i}.wav"
                    dest = conv_dir / name
                    dest.write_bytes(data)
                    segf.write(json.dumps({
                        "source": "edacc",
                        "speaker": f"edacc:{row['speaker']}",
                        "gender": row["gender"] if row["gender"] in ("female", "male") else "unknown",
                        "accent": row["accent"],
                        "l1": row["l1"],
                        "path": str(dest),
                        "duration": round(info.duration, 3),
                        "text": row["text"],
                        "source_file": f"{r['path']}#rg{r['row_group']}/{name}",
                    }) + "\n")
                    n += 1
            done.append(key)
            done_file.write_text(json.dumps(done), encoding="utf-8")
            log(f"  {r['conversation']}: {n} segments ({hf.fetched / 1e6:.1f} MB)")
    return out


def edacc_segments(root: Optional[Path] = None) -> Iterator[dict[str, Any]]:
    seg_file = (root or edacc_dir()) / "segments.jsonl"
    if not seg_file.exists():
        return
    seen: set[str] = set()
    for line in seg_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["path"] not in seen and Path(row["path"]).exists():
                seen.add(row["path"])
                yield row


def disk_usage_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6 if path.exists() else 0.0


def remove_raw(source: str) -> None:
    d = {"librispeech": libri_dir(), "edacc": edacc_dir()}[source]
    if d.exists():
        shutil.rmtree(d)
