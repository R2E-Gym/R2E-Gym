import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="local-transformers")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base = str(args.base_url).rstrip("/")
    url = base + "/v1/chat/completions"
    payload = {
        "model": str(args.model),
        "messages": [{"role": "user", "content": str(args.prompt)}],
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    obj = json.loads(raw)
    print(obj["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
