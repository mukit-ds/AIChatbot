import requests
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import time
import re
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# Marrfa API configuration
MARRFA_API_BASE = "https://apiv2.marrfa.com"
MARRFA_PROPERTIES_URL = f"{MARRFA_API_BASE}/properties"
MARRFA_WEBSITE_BASE = "https://www.marrfa.com"

# Create a session for reuse
session = requests.Session()


def search_property_by_name(property_name: str) -> Dict[str, Any]:
    """
    Search for a specific property by name

    Args:
        property_name: Name of the property to search for

    Returns:
        Dictionary with property details or None if not found
    """
    params = {
        "page": 1,
        "per_page": 20,  # Get more results to increase chance of finding the property
        "search_query": property_name
    }

    try:
        response = session.get(MARRFA_PROPERTIES_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract items from response
        items = data.get("items", []) or data.get("data", []) or []

        # Look for the property with the exact or similar name
        for item in items:
            title = item.get("name") or item.get("title", "")
            if property_name.lower() in title.lower():
                return {
                    "id": item.get("id"),
                    "title": title,
                    "location": item.get("area") or item.get("location", "Dubai"),
                    "developer": item.get("developer", "Unknown"),
                    "listing_url": f"{MARRFA_WEBSITE_BASE}/propertylisting/{item.get('id')}" if item.get('id') else None
                }

        return None

    except Exception as e:
        print(f"Error searching for property: {e}")
        return None


def fix_nextjs_image_url(img_url: str) -> str:
    """
    Fix a Next.js image URL to ensure it's valid

    Args:
        img_url: The Next.js image URL to fix

    Returns:
        The fixed image URL
    """
    try:
        # Extract the original URL and parameters
        if "_next/image?url=" in img_url:
            # Extract the encoded URL part
            encoded_url = img_url.split("_next/image?url=")[1].split("&")[0]
            original_url = urllib.parse.unquote(encoded_url)

            # Extract the width parameter
            width = "1536"  # Default width
            if "&w=" in img_url:
                width = img_url.split("&w=")[1].split("&")[0]

            # Extract the quality parameter
            quality = "75"  # Default quality
            if "&q=" in img_url:
                quality = img_url.split("&q=")[1].split("&")[0]

            # Ensure quality is not 0
            if quality == "0":
                quality = "75"

            # Reconstruct the URL with proper parameters
            fixed_url = f"{MARRFA_WEBSITE_BASE}/_next/image?url={urllib.parse.quote(original_url)}&w={width}&q={quality}"
            return fixed_url
        else:
            return img_url
    except Exception as e:
        print(f"Error fixing image URL: {e}")
        return img_url


def is_property_image(img_url: str, img_size: str = "") -> bool:
    """
    Determine if an image URL is likely a property image rather than a logo or other non-property image

    Args:
        img_url: URL of the image
        img_size: Size parameter from the URL (e.g., "w=128&q=75")

    Returns:
        True if it's likely a property image, False otherwise
    """
    # Get the original URL for analysis
    try:
        if "_next/image?url=" in img_url:
            encoded_url = img_url.split("_next/image?url=")[1].split("&")[0]
            original_url = urllib.parse.unquote(encoded_url)
        else:
            original_url = img_url
    except:
        original_url = img_url

    # Convert to lowercase for case-insensitive matching
    url_lower = original_url.lower()

    # Filter out common non-property image patterns
    non_property_patterns = [
        'logo', 'icon', 'favicon', 'avatar', 'nav', 'footer', 'header',
        'brand', 'company', 'developer', 'builder', 'agent', 'broker',
        'contact', 'phone', 'email', 'social', 'facebook', 'twitter',
        'instagram', 'linkedin', 'whatsapp', 'map', 'location', 'pin',
        'mail.e883c10d.png'  # Specific to the mail icon we're seeing
    ]

    # Check if any non-property patterns are in the URL
    for pattern in non_property_patterns:
        if pattern in url_lower:
            return False

    # Check image size - logos are often smaller (w=128)
    if "w=128" in img_size or "q=0" in img_size:
        return False

    # If it's a static media file (like icons), it's probably not a property image
    if "_next/static/media/" in img_url:
        return False

    # If no patterns match, assume it's a property image (default to True)
    return True


def fetch_property_images_and_overview(property_url: str) -> Dict[str, Any]:
    """
    Fetch property images and overview using enhanced Selenium techniques

    Args:
        property_url: URL of the property listing page

    Returns:
        Dictionary with images and overview
    """
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    try:
        # Automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Navigate to the property page
        driver.get(property_url)

        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Additional wait for dynamic content
        time.sleep(5)

        # Get page source for BeautifulSoup parsing
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract project overview using multiple methods
        overview = ""

        # Method 1: Try to get the overview from the page using Selenium
        try:
            overview_elements = driver.find_elements(By.CSS_SELECTOR,
                                                     "div[class*='overview'] p, div[class*='description'] p")
            if overview_elements:
                overview = "\n".join([elem.text for elem in overview_elements])
        except Exception as e:
            print(f"Error extracting overview with Selenium: {e}")

        # Method 2: Try to get overview using BeautifulSoup
        if not overview:
            try:
                # Look for common overview selectors
                overview_selectors = [
                    "div[class*='overview'] p",
                    "div[class*='description'] p",
                    "section[class*='overview'] p",
                    "div[class*='about'] p",
                    "div[class*='project'] p"
                ]

                for selector in overview_selectors:
                    elements = soup.select(selector)
                    if elements:
                        text = "\n".join([elem.get_text(strip=True) for elem in elements])
                        if len(text) > 50:  # Only use if we got substantial text
                            overview = text
                            break
            except Exception as e:
                print(f"Error extracting overview with BeautifulSoup: {e}")

        # Method 3: Look for the largest text block that might be the description
        if not overview:
            try:
                all_text_blocks = []
                for elem in soup.find_all(['p', 'div']):
                    text = elem.get_text(strip=True)
                    if 50 < len(text) < 2000:  # Reasonable length for a description
                        # Filter out common non-description text
                        if not any(
                                skip in text.lower() for skip in ['cookie', 'privacy', 'terms', 'login', 'register']):
                            all_text_blocks.append((text, elem))

                if all_text_blocks:
                    # Sort by length (descending) and take the longest
                    all_text_blocks.sort(key=lambda x: len(x[0]), reverse=True)
                    overview = all_text_blocks[0][0]
            except Exception as e:
                print(f"Error extracting overview from text blocks: {e}")

        # Extract images using multiple methods
        images = []

        # Method 1: Try to find the Next.js data in the page
        try:
            next_data = driver.execute_script("return window.__NEXT_DATA__")
            if next_data:
                # Navigate through the Next.js data structure to find images
                props = next_data.get('props', {})
                page_props = props.get('pageProps', {})

                # Look for images in various possible locations
                if 'property' in page_props:
                    property_data = page_props['property']
                    if 'images' in property_data:
                        images.extend(property_data['images'])
                    if 'gallery' in property_data:
                        if isinstance(property_data['gallery'], list):
                            for item in property_data['gallery']:
                                if isinstance(item, dict) and 'url' in item:
                                    images.append(item['url'])
                                elif isinstance(item, str):
                                    images.append(item)

                # Also check in other possible locations
                for key, value in page_props.items():
                    if isinstance(value, dict) and 'images' in value:
                        if isinstance(value['images'], list):
                            images.extend(value['images'])
        except Exception as e:
            print(f"Error extracting Next.js data: {e}")

        # Method 2: Look for images in the DOM
        if not images:
            try:
                # Look for image elements that might be property images
                img_elements = driver.find_elements(By.TAG_NAME, "img")
                for img in img_elements:
                    src = img.get_attribute('src') or img.get_attribute('data-src')
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        images.append(src)
            except Exception as e:
                print(f"Error extracting images from DOM: {e}")

        # Method 3: Look for images in script tags
        if not images:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string:
                        # Look for image URLs in JavaScript
                        matches = re.findall(r'["\']([^"\']+\.(jpg|jpeg|png|webp))["\']', script.string, re.IGNORECASE)
                        for match in matches:
                            img_url = match[0]
                            if img_url.startswith('http') or img_url.startswith('//'):
                                images.append(img_url)
            except Exception as e:
                print(f"Error extracting images from scripts: {e}")

        # Method 4: Try to scroll and find lazy-loaded images
        try:
            # Scroll down to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Look for newly loaded images
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            for img in img_elements:
                src = img.get_attribute('src') or img.get_attribute('data-src')
                if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    images.append(src)
        except Exception as e:
            print(f"Error extracting lazy-loaded images: {e}")

        # Clean and deduplicate images
        clean_images = []
        seen = set()
        for img in images:
            # Convert relative URLs to absolute
            if img.startswith('//'):
                img = 'https:' + img
            elif img.startswith('/'):
                img = MARRFA_WEBSITE_BASE + img
            elif not img.startswith('http'):
                img = MARRFA_WEBSITE_BASE + '/' + img

            if img not in seen:
                seen.add(img)
                clean_images.append(img)

        # Filter out non-property images and fix URLs
        property_images = []
        for img in clean_images:
            # Extract the size parameter from the URL
            size_param = ""
            if "&w=" in img:
                size_param = "&w=" + img.split("&w=")[1].split("&")[0]

            if is_property_image(img, size_param):
                # Fix the Next.js image URL
                fixed_url = fix_nextjs_image_url(img)
                property_images.append(fixed_url)

        # Limit to 6 images
        limited_images = property_images[:6]

        return {
            "url": property_url,
            "overview": overview,
            "images": limited_images,
            "image_count": len(limited_images)
        }

    except Exception as e:
        return {
            "url": property_url,
            "overview": "",
            "images": [],
            "image_count": 0,
            "error": str(e)
        }
    finally:
        try:
            driver.quit()
        except:
            pass


def main():
    """
    Main function to search for Marbella Villas and fetch its images and overview
    """
    property_name = "Marbella Villas"

    print(f"Searching for property: {property_name}")

    # Search for the property
    property_info = search_property_by_name(property_name)

    if not property_info:
        print(f"Property '{property_name}' not found.")
        return

    print(f"Found property: {property_info['title']}")
    print(f"Location: {property_info['location']}")
    print(f"Developer: {property_info['developer']}")
    print(f"Listing URL: {property_info['listing_url']}")

    # Fetch images and overview
    print("\nFetching property details...")
    property_details = fetch_property_images_and_overview(property_info['listing_url'])

    if property_details.get('error'):
        print(f"Error fetching property details: {property_details['error']}")
        return

    # Display the overview
    overview = property_details.get('overview', '')
    if overview:
        print("\n=== PROJECT OVERVIEW ===")
        print(overview)
    else:
        print("\nNo project overview found.")

    # Display the images
    images = property_details.get('images', [])
    if images:
        print(f"\n=== PROPERTY IMAGES ({len(images)}) ===")
        for i, img in enumerate(images, 1):
            print(f"{i}. {img}")
    else:
        print("\nNo images found.")


if __name__ == "__main__":
    main()