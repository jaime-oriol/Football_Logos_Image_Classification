"""
MEGA FRUIT SCRAPER - Multi-source image downloader
Downloads high-quality fruit images from multiple legal and free sources:
- Unsplash API (high quality, free)
- Pexels API (high quality, free)
- Pixabay API (high quality, free)

All sources are 100% legal for educational/research use.
"""

import os
import requests
import time
from pathlib import Path
import json


class MegaFruitScraper:
    """
    Multi-source fruit image scraper.
    Uses 3 professional photography APIs with free access.
    """

    def __init__(self, output_dir='data_fruits'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # API Keys (you need to register for free at each service)
        # Unsplash: https://unsplash.com/developers
        # Pexels: https://www.pexels.com/api/
        # Pixabay: https://pixabay.com/api/docs/

        self.unsplash_key = None  # Replace with your key
        self.pexels_key = None    # Replace with your key
        self.pixabay_key = None   # Replace with your key

        # Spanish to English fruit mapping for better search results
        self.fruit_mapping = {
            'Albaricoques': 'apricot',
            'Higos': 'fig',
            'Ciruelas': 'plum',
            'Cerezas': 'cherry',
            'Melón': 'melon',
            'Sandía': 'watermelon',
            'Nectarinas': 'nectarine',
            'Paraguayos': 'flat peach',
            'Melocotón': 'peach',
            'Nísperos': 'loquat',
            'Pera': 'pear',
            'Plátano': 'banana',
            'Frutos rojos': 'berries',
            'Caqui': 'persimmon',
            'Chirimoya': 'cherimoya',
            'Granada': 'pomegranate',
            'Kiwis': 'kiwi',
            'Mandarinas': 'mandarin',
            'Manzana': 'apple',
            'Naranja': 'orange',
            'Pomelo': 'grapefruit'
        }

        self.session = requests.Session()
        self.download_stats = {fruit: 0 for fruit in self.fruit_mapping.keys()}

    def setup_apis(self, unsplash_key=None, pexels_key=None, pixabay_key=None):
        """Configure API keys for the scraper."""
        self.unsplash_key = unsplash_key
        self.pexels_key = pexels_key
        self.pixabay_key = pixabay_key

    def download_image(self, url, filepath):
        """Download a single image from URL."""
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"  ✗ Error downloading {filepath.name}: {e}")
            return False

    def scrape_unsplash(self, fruit_en, fruit_es, max_images=100):
        """
        Scrape images from Unsplash API.
        Unsplash has the highest quality photos.
        """
        if not self.unsplash_key:
            print(f"  ⊘ Unsplash API key not configured for {fruit_es}")
            return 0

        print(f"  → Scraping Unsplash for {fruit_es} ({fruit_en})...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        page = 1
        per_page = 30

        while downloaded < max_images:
            try:
                url = f"https://api.unsplash.com/search/photos"
                params = {
                    'query': f'{fruit_en} fruit',
                    'page': page,
                    'per_page': per_page,
                    'client_id': self.unsplash_key,
                    'orientation': 'squarish'  # Better for ML
                }

                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get('results'):
                    break

                for idx, photo in enumerate(data['results']):
                    if downloaded >= max_images:
                        break

                    # Download regular quality (good balance)
                    img_url = photo['urls']['regular']
                    photo_id = photo['id']

                    filepath = fruit_dir / f"unsplash_{photo_id}.jpg"

                    if filepath.exists():
                        continue

                    if self.download_image(img_url, filepath):
                        downloaded += 1
                        print(f"    ✓ {fruit_es}: {downloaded}/{max_images}")

                    time.sleep(0.5)  # Rate limiting

                page += 1

            except Exception as e:
                print(f"  ✗ Unsplash error for {fruit_es}: {e}")
                break

        return downloaded

    def scrape_pexels(self, fruit_en, fruit_es, max_images=100):
        """
        Scrape images from Pexels API.
        Pexels has great variety and quality.
        """
        if not self.pexels_key:
            print(f"  ⊘ Pexels API key not configured for {fruit_es}")
            return 0

        print(f"  → Scraping Pexels for {fruit_es} ({fruit_en})...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        page = 1
        per_page = 80  # Max for Pexels

        while downloaded < max_images:
            try:
                url = f"https://api.pexels.com/v1/search"
                headers = {'Authorization': self.pexels_key}
                params = {
                    'query': f'{fruit_en} fruit',
                    'page': page,
                    'per_page': per_page,
                    'orientation': 'square'
                }

                response = self.session.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get('photos'):
                    break

                for photo in data['photos']:
                    if downloaded >= max_images:
                        break

                    # Download medium quality
                    img_url = photo['src']['medium']
                    photo_id = photo['id']

                    filepath = fruit_dir / f"pexels_{photo_id}.jpg"

                    if filepath.exists():
                        continue

                    if self.download_image(img_url, filepath):
                        downloaded += 1
                        print(f"    ✓ {fruit_es}: {downloaded}/{max_images}")

                    time.sleep(0.3)

                page += 1

            except Exception as e:
                print(f"  ✗ Pexels error for {fruit_es}: {e}")
                break

        return downloaded

    def scrape_pixabay(self, fruit_en, fruit_es, max_images=100):
        """
        Scrape images from Pixabay API.
        Pixabay has huge variety but requires API key.
        """
        if not self.pixabay_key:
            print(f"  ⊘ Pixabay API key not configured for {fruit_es}")
            return 0

        print(f"  → Scraping Pixabay for {fruit_es} ({fruit_en})...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        page = 1
        per_page = 200  # Max for Pixabay

        while downloaded < max_images:
            try:
                url = f"https://pixabay.com/api/"
                params = {
                    'key': self.pixabay_key,
                    'q': f'{fruit_en} fruit',
                    'page': page,
                    'per_page': per_page,
                    'image_type': 'photo',
                    'orientation': 'horizontal'
                }

                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get('hits'):
                    break

                for hit in data['hits']:
                    if downloaded >= max_images:
                        break

                    # Download webformat (medium quality)
                    img_url = hit['webformatURL']
                    photo_id = hit['id']

                    filepath = fruit_dir / f"pixabay_{photo_id}.jpg"

                    if filepath.exists():
                        continue

                    if self.download_image(img_url, filepath):
                        downloaded += 1
                        print(f"    ✓ {fruit_es}: {downloaded}/{max_images}")

                    time.sleep(0.2)

                page += 1

            except Exception as e:
                print(f"  ✗ Pixabay error for {fruit_es}: {e}")
                break

        return downloaded

    def scrape_all(self, images_per_fruit=300):
        """
        Scrape all fruits from all sources.
        Distributes downloads across sources: 40% Unsplash, 30% Pexels, 30% Pixabay.
        """
        print("=" * 70)
        print("MEGA FRUIT SCRAPER - Starting download")
        print("=" * 70)
        print(f"Target: {images_per_fruit} images per fruit")
        print(f"Total fruits: {len(self.fruit_mapping)}")
        print(f"Total images target: {images_per_fruit * len(self.fruit_mapping)}")
        print("=" * 70)

        unsplash_per_fruit = int(images_per_fruit * 0.4)  # 40%
        pexels_per_fruit = int(images_per_fruit * 0.3)    # 30%
        pixabay_per_fruit = int(images_per_fruit * 0.3)   # 30%

        for fruit_es, fruit_en in self.fruit_mapping.items():
            print(f"\n{'=' * 70}")
            print(f"Downloading: {fruit_es} ({fruit_en})")
            print(f"{'=' * 70}")

            total_downloaded = 0

            # Source 1: Unsplash (highest quality)
            if self.unsplash_key:
                count = self.scrape_unsplash(fruit_en, fruit_es, unsplash_per_fruit)
                total_downloaded += count

            # Source 2: Pexels
            if self.pexels_key:
                count = self.scrape_pexels(fruit_en, fruit_es, pexels_per_fruit)
                total_downloaded += count

            # Source 3: Pixabay
            if self.pixabay_key:
                count = self.scrape_pixabay(fruit_en, fruit_es, pixabay_per_fruit)
                total_downloaded += count

            self.download_stats[fruit_es] = total_downloaded

            print(f"\n✓ {fruit_es} completed: {total_downloaded} images")

            time.sleep(1)  # Cool down between fruits

        self.print_summary()

    def print_summary(self):
        """Print download statistics."""
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)

        for fruit, count in sorted(self.download_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fruit:20s}: {count:4d} images")

        total = sum(self.download_stats.values())
        print("=" * 70)
        print(f"TOTAL DOWNLOADED: {total} images")
        print(f"Average per fruit: {total / len(self.download_stats):.1f} images")
        print("=" * 70)


def main():
    """Main execution function."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                    MEGA FRUIT SCRAPER v1.0                         ║
║                   Multi-source Image Downloader                    ║
╚════════════════════════════════════════════════════════════════════╝

This scraper uses 3 professional photography APIs:
  1. Unsplash - https://unsplash.com/developers
  2. Pexels   - https://www.pexels.com/api/
  3. Pixabay  - https://pixabay.com/api/docs/

SETUP INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Register for FREE API keys at each service (takes 2 min each)
2. Copy your API keys below
3. Run the scraper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET: 300 images per fruit × 22 fruits = 6,600 total images
""")

    # Get API keys from user
    print("Please enter your API keys (or press Enter to skip that source):\n")

    unsplash_key = input("Unsplash API Key (Access Key): ").strip() or None
    pexels_key = input("Pexels API Key: ").strip() or None
    pixabay_key = input("Pixabay API Key: ").strip() or None

    if not any([unsplash_key, pexels_key, pixabay_key]):
        print("\n⚠️  No API keys provided. Please register for at least one service.")
        print("\nQuick registration links:")
        print("  Unsplash: https://unsplash.com/oauth/applications")
        print("  Pexels:   https://www.pexels.com/api/new/")
        print("  Pixabay:  https://pixabay.com/api/docs/#api_register")
        return

    # Create scraper
    scraper = MegaFruitScraper(output_dir='data_fruits')
    scraper.setup_apis(
        unsplash_key=unsplash_key,
        pexels_key=pexels_key,
        pixabay_key=pixabay_key
    )

    # Ask for images per fruit
    try:
        images_per_fruit = int(input("\nImages per fruit [default: 300]: ").strip() or "300")
    except ValueError:
        images_per_fruit = 300

    print(f"\n🚀 Starting download: {images_per_fruit} images × {len(scraper.fruit_mapping)} fruits")
    print("This may take 30-60 minutes depending on your internet connection...")
    input("\nPress ENTER to start...")

    # Start scraping
    scraper.scrape_all(images_per_fruit=images_per_fruit)

    print("\n✅ Scraping completed!")
    print(f"📁 Images saved to: {scraper.output_dir.absolute()}")


if __name__ == "__main__":
    main()
