"""Package the fixed standalone Q1 TCN source, never logs or checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


VERSION = "q1_tcn_v7_joint_v1"
ROOT = Path(__file__).resolve().parents[1]


def package(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Never overwrite a fixed release: " + str(output))
    paths = {ROOT/"setup.py", ROOT/"README.md", ROOT/"LICENSE", ROOT/".gitignore"}
    paths.update((ROOT/"wheel_legged_gym").rglob("*.py"))
    paths.update((ROOT/"tests").rglob("*.py"))
    for extension in ("*.py", "*.sh", "*.json"):
        paths.update((ROOT/"tools").rglob(extension))
    # Retain the other robots' assets as well; unpacking this source release
    # must not remove resources needed by the repository's original tasks.
    paths.update(p for p in (ROOT/"resources").rglob("*") if p.is_file())
    paths.update(p for p in (ROOT/"licenses").rglob("*") if p.is_file())
    paths.add(ROOT/"docs/q1_tcn_joint.md")
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)}
    content = json.dumps(dict(profile_version=VERSION, files=manifest), indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path in sorted(paths):
                archive.add(path, arcname=VERSION+"/"+path.relative_to(ROOT).as_posix(), recursive=False)
            info = tarfile.TarInfo(VERSION+"/release_manifest.json")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(digest+"  "+output.name+"\n", encoding="utf-8")
    print(json.dumps(dict(path=str(output), sha256=digest, files=len(manifest)), indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"releases"/(VERSION+".tar.gz"))
    package(parser.parse_args().output)
