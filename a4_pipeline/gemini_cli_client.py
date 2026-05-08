from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_NODE = Path.home() / ".nvm" / "versions" / "node" / "v24.15.0" / "bin" / "node"
DEFAULT_GEMINI_JS = (
    Path.home()
    / ".npm"
    / "_npx"
    / "d07ada7b4a99c96e"
    / "node_modules"
    / "@google"
    / "gemini-cli"
    / "bundle"
    / "gemini.js"
)

CLI_MODELS = [
    "auto-gemini-2.5",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "auto-gemini-3",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"JSON object not found in Gemini CLI output: {text[:300]}")
    return json.loads(text[start : end + 1])


def gemini_command() -> List[str]:
    override = os.environ.get("GEMINI_CLI_CMD", "").strip()
    if override:
        return override.split()
    if DEFAULT_GEMINI_JS.exists() and DEFAULT_NODE.exists():
        return [str(DEFAULT_NODE), str(DEFAULT_GEMINI_JS)]
    return ["gemini"]


def generate_with_gemini_cli(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    timeout: int = 240,
    output_format: str = "text",
) -> str:
    cmd = gemini_command() + [
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        output_format,
        "--skip-trust",
    ]
    env = os.environ.copy()
    node_bin = str(DEFAULT_NODE.parent)
    env["PATH"] = f"{node_bin}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")
    proc = subprocess.run(
        cmd,
        cwd="/Volumes/외장 2TB/cpu2026/common/code",
        text=True,
        input="",
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    combined = "\n".join(x for x in [stdout, stderr] if x)
    if "Opening authentication page" in combined or "Do you want to continue" in combined:
        raise RuntimeError(
            "Gemini CLI requires interactive authentication in this process. "
            "Run the same command once in your normal Terminal, or install/use a global gemini command."
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Gemini CLI failed with code {proc.returncode}: {combined[:1000]}")
    return stdout


def generate_json_with_gemini_cli(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    timeout: int = 240,
) -> Dict[str, Any]:
    text = generate_with_gemini_cli(prompt, model=model, timeout=timeout, output_format="text")
    return extract_json_object(text)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Small Gemini CLI wrapper test.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--prompt", default='Return only JSON: {"ok": true}')
    args = parser.parse_args()

    if args.list_models:
        print(json.dumps({"models": CLI_MODELS}, ensure_ascii=False, indent=2))
        return
    result = generate_with_gemini_cli(args.prompt, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
