#!/usr/bin/env python3
"""Fetch the RMBG-2.0 checkpoint that rmbg-rs reads with --weights.

The checkpoint is not in this repository or its releases: RMBG-2.0 is licensed
by BRIA for non-commercial use only, so it is taken from the upstream model card
instead of being redistributed.  Read https://huggingface.co/briaai/RMBG-2.0
before you use it.

briaai/RMBG-2.0 is a *gated* model.  A Hugging Face account is required and the
account must have accepted BRIA's terms on the model page; downloads are then
made with an access token from https://huggingface.co/settings/tokens .

Usage, from a checkout or straight off GitHub:

    ./get-model.py                                  # writes ./rmbg-2.0.safetensors
    ./get-model.py /models/rmbg.safetensors         # writes somewhere else instead
    python3 get-model.py --token hf_xxx             # the token can be passed explicitly

The token is looked up in this order:

    1. --token on the command line
    2. the HF_TOKEN environment variable
    3. the HUGGING_FACE_HUB_TOKEN environment variable
    4. ~/.cache/huggingface/token
    5. an interactive prompt (nothing is echoed)

This file is the whole tool: standard library only, no curl, no wget, no
`huggingface_hub`, and no second file to download.  It only renames the finished
file into place once its size and SHA-256 match the checkpoint this engine was
validated on, and it leaves nothing behind if it fails or is interrupted: the
half-written download is deleted on the way out.
"""

import argparse
import hashlib
import getpass
import os
import pathlib
import shutil
import sys
import urllib.error
import urllib.request

REPO = "briaai/RMBG-2.0"
REVISION = "main"
FILENAME = "model.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"

# The checkpoint this engine was validated against.  A re-uploaded file fails
# the check instead of being used silently, which is the point.
SIZE = 884_878_856
SHA256 = "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7"

DEFAULT_OUT = "rmbg-2.0.safetensors"
CHUNK = 1 << 20  # 1 MiB
TOKEN_FILE = pathlib.Path.home() / ".cache" / "huggingface" / "token"


def find_token(cli_token):
    """Return (token, source), prompting if nothing is available."""
    if cli_token:
        return cli_token, "--token"
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip(), var
    try:
        value = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    else:
        if value:
            return value, str(TOKEN_FILE)

    if not sys.stdin.isatty():
        die(
            "no Hugging Face access token found.\n"
            "  briaai/RMBG-2.0 is gated.  Accept the licence on its model page\n"
            "\n"
            "      https://huggingface.co/briaai/RMBG-2.0\n"
            "\n"
            "  with your Hugging Face account, then create a read token at\n"
            "  https://huggingface.co/settings/tokens and pass it with --token,\n"
            f"  set HF_TOKEN, or write it to {TOKEN_FILE}."
        )
    print("No Hugging Face access token found.", file=sys.stderr)
    print(
        "Accept the licence at https://huggingface.co/briaai/RMBG-2.0 with your\n"
        "Hugging Face account, then create a read token at\n"
        "https://huggingface.co/settings/tokens .",
        file=sys.stderr,
    )
    token = getpass.getpass("Access token (input hidden): ").strip()
    if not token:
        die("no token given")
    return token, "prompt"


def die(message, code=1):
    print(f"get-model: error: {message}", file=sys.stderr)
    raise SystemExit(code)


def open_url(token):
    """Open the checkpoint URL, authenticated."""
    request = urllib.request.Request(URL)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "rmbg-rs-get-model/1")
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            body = error.read().decode("utf-8", "replace").strip()
            gated = "gated" in body.lower() or "access to model" in body.lower()
            die(
                f"the server refused the request (HTTP {error.code}).\n"
                + (
                    "  This is what a token from an account that has not accepted the\n"
                    "  model's terms looks like.  Open\n"
                    "\n"
                    "      https://huggingface.co/briaai/RMBG-2.0\n"
                    "\n"
                    "  log in, accept the licence on that page, and run this again with a\n"
                    "  token from https://huggingface.co/settings/tokens .\n"
                    if gated or error.code == 401
                    else ""
                )
                + f"  The server said: {body[:200] or error.reason}"
            )
        if error.code == 404:
            die(
                f"HTTP 404: {URL} has moved.  Check the model card at\n"
                f"  https://huggingface.co/{REPO}"
            )
        die(f"HTTP {error.code} from {URL}")
    except urllib.error.URLError as error:
        die(f"could not reach huggingface.co: {error.reason}")


def verify(path):
    """Return None if path matches SIZE and SHA256, else a description."""
    actual_size = path.stat().st_size
    if actual_size != SIZE:
        return f"size is {actual_size:,} bytes, expected {SIZE:,}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash != SHA256:
        return f"sha256 is {actual_hash}, expected {SHA256}"
    return None


def human(count):
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def download(out, token):
    """Download the checkpoint to a temporary sibling, verify it, rename it in.

    Nothing is left behind on any exit path: the temporary file is removed if
    the download fails, is interrupted or fails verification.  Nothing ever
    appears at *out* unless the size and SHA-256 both match, because the rename
    is the last step.
    """
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)

    response = open_url(token)
    print(f"get-model: downloading {human(SIZE)} from {URL}", file=sys.stderr)

    done = 0
    try:
        with part.open("wb") as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                percent = 100.0 * done / SIZE if SIZE else 100.0
                print(
                    f"\r  {percent:5.1f}%  {human(done)} of {human(SIZE)}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        print(file=sys.stderr)

        problem = verify(part)
        if problem:
            die(
                f"the downloaded file does not match the checkpoint this engine was\n"
                f"  validated against: {problem}\n"
                f"  The temporary file has been removed; nothing was written to {out}.\n"
                f"  If the upstream checkpoint was re-uploaded, this script and the\n"
                f"  engine both need updating."
            )
    except BaseException as error:
        # BaseException, not Exception: an aborted run (Ctrl-C) must not leave a
        # partial file behind any more than a failed one does.
        part.unlink(missing_ok=True)
        if isinstance(error, KeyboardInterrupt):
            print("\nget-model: interrupted; nothing was written", file=sys.stderr)
        raise
    finally:
        response.close()

    os.replace(part, out)
    print(f"get-model: wrote {out} ({SIZE:,} bytes, sha256 {SHA256[:16]}...)", file=sys.stderr)


def main(argv):
    parser = argparse.ArgumentParser(
        prog="get-model",
        description="Download the RMBG-2.0 checkpoint that rmbg-rs reads with --weights.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=os.environ.get("RMBG_MODEL_OUT", DEFAULT_OUT),
        help=f"where to write the checkpoint (default: {DEFAULT_OUT}, or $RMBG_MODEL_OUT)",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token (default: $HF_TOKEN, "
        f"$HUGGING_FACE_HUB_TOKEN, {TOKEN_FILE}, or an interactive prompt)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the file is already correct")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.output).expanduser()
    if out.is_dir():
        out = out / DEFAULT_OUT

    print(
        "RMBG-2.0 is licensed by BRIA for non-commercial use only.\n"
        "Read https://huggingface.co/briaai/RMBG-2.0 before you use it.",
        file=sys.stderr,
    )

    if out.exists() and not args.force:
        problem = verify(out)
        if problem is None:
            print(f"get-model: {out} is already present and verified", file=sys.stderr)
            return 0
        print(f"get-model: {out} exists but {problem}; downloading again", file=sys.stderr)

    if not out.parent.exists():
        die(f"no such directory: {out.parent}")
    free = shutil.disk_usage(out.parent).free
    if free < SIZE:
        die(f"not enough free space for the checkpoint: {human(free)} available, {human(SIZE)} needed")

    token, source = find_token(args.token)
    print(f"get-model: using token from {source}", file=sys.stderr)
    download(out, token)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
