"""
Scrape West Wing scripts from westwingwiki.com using Playwright.

Usage:
    1. Install dependencies:
       pip install playwright beautifulsoup4
       playwright install chromium

    2. Run:
       python scrape_scripts.py

    Scripts are saved to ./scripts/season_N/ as text files for each season.
"""

import asyncio
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

BASE_URL = "https://westwingwiki.com"
NUM_SEASONS = 3
OUTPUT_DIR = Path("scripts")

# Delay between requests to be respectful to the server
REQUEST_DELAY_SECONDS = 3
# Longer delay between seasons to avoid Cloudflare blocks
SEASON_DELAY_SECONDS = 15


async def wait_for_cloudflare(page, timeout_ms=60000):
    """Wait for Cloudflare challenge to resolve."""
    try:
        # Wait until the page title changes from "Just a moment..."
        await page.wait_for_function(
            "document.title !== 'Just a moment...'",
            timeout=timeout_ms,
        )
        # Give extra time for page content to fully load after challenge
        await page.wait_for_load_state("networkidle", timeout=30000)
        # Additional settle time for Cloudflare redirect
        await asyncio.sleep(3)
    except Exception:
        # If the function times out, the page might already be loaded
        pass


async def get_episode_links(page, season: int) -> list[dict]:
    """Navigate to a season page and extract all episode links."""
    season_url = f"{BASE_URL}/the-wiki/scripts/season-{season}/"
    print(f"Navigating to {season_url}")
    await page.goto(season_url, wait_until="domcontentloaded")
    await wait_for_cloudflare(page)

    content = await page.content()
    soup = BeautifulSoup(content, "html.parser")

    # Find all links that point to individual episode pages
    episode_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)
        # Match episode links - they typically contain "episode" in the URL
        # or the link text references an episode
        if re.search(r"season-\d+-episode-\d+", href):
            full_url = urljoin(BASE_URL, href)
            if full_url not in [ep["url"] for ep in episode_links]:
                episode_links.append({"url": full_url, "title": text})
                print(f"  Found: {text} -> {full_url}")

    if not episode_links:
        # Fallback: try to find links in the main content area
        main_content = soup.find("article") or soup.find(
            "div", class_=re.compile(r"entry|content|post")
        )
        if main_content:
            for a_tag in main_content.find_all("a", href=True):
                href = a_tag["href"]
                text = a_tag.get_text(strip=True)
                if href.startswith("http") and "westwingwiki" in href:
                    full_url = urljoin(BASE_URL, href)
                    if full_url not in [ep["url"] for ep in episode_links]:
                        episode_links.append({"url": full_url, "title": text})
                        print(f"  Found (fallback): {text} -> {full_url}")

    if not episode_links:
        debug_path = Path(f"debug_season_{season}.html")
        debug_path.write_text(content)
        print(f"  No episodes found. Saved page HTML to {debug_path} for debugging.")

    print(f"\nFound {len(episode_links)} episode links total.")
    return episode_links


async def scrape_script(page, url: str) -> str | None:
    """Navigate to an episode page and extract the script text."""
    try:
        await page.goto(url, wait_until="domcontentloaded")
        await wait_for_cloudflare(page)

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        # Try to find the main content area containing the script
        # WordPress sites typically use these selectors
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"entry-content|post-content"))
            or soup.find("div", class_="content")
        )

        if article:
            # Remove navigation, comments, share buttons, etc.
            for unwanted in article.find_all(
                ["nav", "footer", "aside", "script", "style", "form"]
            ):
                unwanted.decompose()
            for unwanted in article.find_all(
                class_=re.compile(
                    r"share|social|comment|sidebar|nav|menu|footer|related"
                )
            ):
                unwanted.decompose()

            # Get the text, preserving line breaks
            # Replace <br> and <p> with newlines for proper formatting
            for br in article.find_all("br"):
                br.replace_with("\n")
            for p in article.find_all("p"):
                p.insert_before("\n")
                p.insert_after("\n")

            text = article.get_text()
            # Clean up excessive whitespace while preserving structure
            lines = text.split("\n")
            cleaned_lines = [line.strip() for line in lines]
            text = "\n".join(cleaned_lines)
            # Remove excessive blank lines (more than 2 consecutive)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        # Fallback: get body text
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n").strip()

        return None
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def sanitize_filename(title: str) -> str:
    """Convert episode title to a safe filename."""
    # Remove non-alphanumeric characters except spaces and hyphens
    clean = re.sub(r'[^\w\s\-]', '', title)
    # Replace spaces with underscores
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:100]  # Limit length


async def main():
    async with async_playwright() as p:
        for season in range(1, NUM_SEASONS + 1):
            season_dir = OUTPUT_DIR / f"season_{season}"

            # # Skip seasons that already have scraped content
            # if season_dir.exists() and any(season_dir.glob("*.txt")):
            #     existing = list(season_dir.glob("*.txt"))
            #     print(f"\nSkipping season {season} ({len(existing)} scripts already scraped).")
            #     continue

            print(f"\n{'='*60}")
            print(f"  SEASON {season}")
            print(f"{'='*60}")

            season_dir.mkdir(parents=True, exist_ok=True)

            # Fresh browser context per season to reset Cloudflare state
            # Use headed mode (headless=False) to bypass Cloudflare bot detection
            browser = await p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            # Step 1: Get all episode links from the season page
            episode_links = await get_episode_links(page, season)

            if not episode_links:
                print(f"No episode links found for season {season}. Skipping.")
                await browser.close()
                continue

            # Step 2: Visit each episode page and extract the script
            for i, episode in enumerate(episode_links, 1):
                url = episode["url"]
                title = episode["title"]
                print(f"\n[{i}/{len(episode_links)}] Scraping: {title}")
                print(f"  URL: {url}")

                script_text = await scrape_script(page, url)

                if script_text:
                    filename = sanitize_filename(title) or f"episode_{i}"
                    filepath = season_dir / f"{filename}.txt"
                    filepath.write_text(script_text, encoding="utf-8")
                    print(f"  Saved to {filepath} ({len(script_text)} chars)")
                else:
                    print(f"  WARNING: No script content found for {title}")

                # Be respectful with rate limiting
                if i < len(episode_links):
                    print(f"  Waiting {REQUEST_DELAY_SECONDS}s before next request...")
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)

            await browser.close()

            # Longer delay between seasons to avoid Cloudflare blocks
            if season < NUM_SEASONS:
                print(f"\nWaiting {SEASON_DELAY_SECONDS}s before next season...")
                await asyncio.sleep(SEASON_DELAY_SECONDS)

    print(f"\nDone! Scripts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
