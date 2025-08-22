#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, logging, argparse, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests, yaml

# ---------- Config ----------
OUTPUT_DIR = Path("content/posts")
KEYWORDS_FILE = Path(os.getenv("KEYWORDS_FILE", "keywords.txt"))
PRODUCTS_FILE = Path(os.getenv("AFFILIATE_PRODUCTS", "affiliate_products.csv"))  # optional
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
AMAZON_TAG = os.getenv("AMAZON_TAG", "").strip()  # e.g., "yourtag-20"
AMAZON_TLD = os.getenv("AMAZON_TLD", "com").strip()  # "com", "co.uk", "ca", etc.

# Azure optional
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("auto-gen")

# ---------- Utils ----------
def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", "-".join(s.lower().split())).strip("-")

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def read_keywords(limit: Optional[int]) -> List[str]:
    if not KEYWORDS_FILE.exists():
        raise SystemExit(f"Missing {KEYWORDS_FILE}")
    kws = [k.strip() for k in KEYWORDS_FILE.read_text(encoding="utf-8").splitlines() if k.strip()]
    return kws[:limit] if limit else kws

def load_products(path: Path) -> List[Dict[str, str]]:
    """
    affiliate_products.csv (optional), headers (case-insensitive):
    name,asin,country (optional TLD like com, co.uk, ca)
    """
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 and "name" in line.lower() and "asin" in line.lower():
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        d = {"name": parts[0]}
        if len(parts) > 1: d["asin"] = parts[1]
        if len(parts) > 2: d["country"] = parts[2]
        rows.append(d)
    return rows

def amazon_search_url(query: str, tag: str, tld: str) -> str:
    from urllib.parse import quote_plus
    base = f"https://www.amazon.{tld}/s?k={quote_plus(query)}"
    return f"{base}&tag={tag}" if tag else base

def amazon_asin_url(asin: str, tag: str, tld: str) -> str:
    url = f"https://www.amazon.{tld}/dp/{asin}/"
    return f"{url}?tag={tag}" if tag else url

def pick_products_for_keyword(keyword: str, products: List[Dict[str, str]], tld: str) -> List[Dict[str, str]]:
    """
    naive match: any product whose name contains any keyword token; else return empty -> use search fallback
    """
    toks = [t for t in re.split(r"[\s/-]+", keyword.lower()) if t]
    picks = []
    for p in products:
        name = p.get("name","")
        if any(t in name.lower() for t in toks):
            picks.append(p)
    return picks[:5]  # cap top 5

# ---------- LLM ----------
def build_chat(messages: List[Dict[str, str]]) -> (str, Dict[str,str], Dict):
    if AZURE_ENDPOINT:
        url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}
        payload = {"messages": messages, "temperature": 0.2, "max_tokens": 2200}
    else:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise SystemExit("Missing OPENAI_API_KEY")
        url = f"{OPENAI_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 2200}
    return url, headers, payload

def llm_article(keyword: str) -> Dict[str, str]:
    system = "You are a helpful SEO content assistant."
    user = f"""
Return STRICT JSON for an SEO article about: "{keyword}"

JSON keys:
- title (<=70 chars), meta (<=160 chars), slug, content_markdown (1100-1500 words).
- Use H2/H3 headings, bullets, a 'Top Picks' product section, a 'How to choose' section, and 3 FAQs.
- For each product in 'Top Picks', include a short 1-2 sentence benefit-forward blurb.
Output ONLY JSON. No backticks.
"""
    url, headers, payload = build_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        raise SystemExit(f"OpenAI error {r.status_code}: {r.text[:600]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()
    # Strict JSON parse with fallback
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            return json.loads(m.group(0))
        # last resort
        title = keyword.title()
        slug = slugify(keyword)
        meta = content[:157] + "..." if len(content) > 160 else content
        return {"title": title, "meta": meta, "slug": slug, "content_markdown": content}

# ---------- Post writing ----------
def inject_affiliate_links(markdown: str, keyword: str, products: List[Dict[str,str]]) -> str:
    """
    Replace a 'Top Picks' section in-place:
    - If curated products available: make bullet list with ASIN links (preferred).
    - Else: create search links for the keyword.
    """
    lines = []
    # Try curated first
    curated = products
    if curated:
        lines.append("## Top Picks\n")
        for p in curated:
            name = p.get("name","").strip()
            asin = p.get("asin","").strip()
            tld = p.get("country", AMAZON_TLD) or AMAZON_TLD
            if asin:
                url = amazon_asin_url(asin, AMAZON_TAG, tld)
            else:
                url = amazon_search_url(name or keyword, AMAZON_TAG, tld)
            lines.append(f"- [{name}]({url}) — editor’s note: great value and eco-friendly materials.\n")
    else:
        # fallback: keyword search links
        lines.append("## Top Picks\n")
        for i in range(1, 6):
            url = amazon_search_url(f"{keyword} best option {i}", AMAZON_TAG, AMAZON_TLD)
            lines.append(f"- [See option {i}]({url}) — curated pick for {keyword}.\n")

    block = "\n".join(lines) + "\n"

    # Try to replace existing "Top Picks" section; else append near the top
    replaced = re.sub(r"(?is)##\s*Top Picks.*?(?=^##|\Z)", block, markdown, count=1, flags=re.M)
    if replaced != markdown:
        return replaced
    # Insert after first H2 if exists, else prepend
    m = re.search(r"(?m)^##\s+.+\n", markdown)
    if m:
        idx = m.end()
        return markdown[:idx] + "\n" + block + "\n" + markdown[idx:]
    return block + "\n" + markdown

def write_markdown(post: Dict[str, str]) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_iso = now_iso()
    front = {
        "title": post["title"],
        "date": date_iso,
        "description": post.get("meta",""),
        "slug": post.get("slug") or slugify(post["title"]),
        "tags": ["affiliate","guide","eco"],
        "draft": False,
    }
    fm = "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n"
    fname = f"{date_iso[:10]}-{front['slug']}.md"
    path = OUTPUT_DIR / fname
    path.write_text(fm + post["content_markdown"], encoding="utf-8")
    return str(path)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate posts with Amazon affiliate links.")
    ap.add_argument("--limit", type=int, default=1, help="How many keywords to process this run.")
    args = ap.parse_args()

    keywords = read_keywords(args.limit)
    curated = load_products(PRODUCTS_FILE)

    created = 0
    for kw in keywords:
        LOG.info(f"Generating for: {kw}")
        article = llm_article(kw)
        article.setdefault("title", kw.title())
        article.setdefault("slug", slugify(kw))
        article.setdefault("meta", "")

        # curate affiliates
        picks = pick_products_for_keyword(kw, curated, AMAZON_TLD)
        body = inject_affiliate_links(article["content_markdown"], kw, picks)
        path = write_markdown({
            "title": article["title"],
            "slug": slugify(article["slug"]),
            "meta": article["meta"],
            "content_markdown": body,
        })
        LOG.info(f"Wrote {path}")
        created += 1
        time.sleep(1)

    if created:
        try:
            subprocess.run(["git","config","user.name","github-actions"], check=True)
            subprocess.run(["git","config","user.email","actions@users.noreply.github.com"], check=True)
            subprocess.run(["git","add","content/posts"], check=True)
            subprocess.run(["git","commit","-m",f"Add {created} post(s) [auto]"], check=True)
            subprocess.run(["git","push","origin","main"], check=True)
        except Exception as e:
            LOG.warning(f"Git commit/push skipped or failed: {e}")

if __name__ == "__main__":
    main()
