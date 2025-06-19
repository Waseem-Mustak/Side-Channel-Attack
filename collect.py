import time
import json
import os
import signal
import sys
import random
import traceback
import socket
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 600
FINGERPRINTING_URL = "http://localhost:5000"
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    try:
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Create and return the driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Mask automation
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        
        return driver
        
    except Exception as e:
        print(f"Error setting up WebDriver: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Chrome browser is installed")
        print("2. Try these commands:")
        print("   pip uninstall selenium webdriver-manager")
        print("   pip install selenium==4.15.0")
        print("3. If still having issues:")
        print("   a. Check your Chrome version at chrome://version")
        print("   b. Download matching ChromeDriver from: https://chromedriver.chromium.org/downloads")
        print("   c. Extract chromedriver.exe to your project directory")
        raise

# end of setup_webdriver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []


def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()
    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))


def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining = {site: max(0, TRACES_PER_SITE - count)
                 for site, count in current_counts.items()}
    total_remaining = sum(remaining.values())
    print(f"Total traces remaining: {total_remaining}")
    return total_remaining == 0


def collect_single_trace(driver, wait, website_url):
    """Collect a single fingerprint trace for the given website."""
    try:
        print(f"\n▶ Starting trace for {website_url}")
        driver.get(FINGERPRINTING_URL)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        # print("   • Clicking “Collect trace”…", end=" ")
        collect_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Collect trace')]"))
        )
        collect_button.click()
        # print("Done")

        time.sleep(2)  # initialize
        fingerprinting_tab = driver.current_window_handle

        # print("   • Opening target site…", end=" ")
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(website_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        # print("Loaded")

        # print("   • Simulating scrolling…", end=" ")
        height = driver.execute_script("return document.body.scrollHeight;")
        for _ in range(random.randint(3, 6)):
            driver.execute_script(f"window.scrollTo(0, {random.randint(0, height)});")
            time.sleep(random.uniform(0.5, 1.0))
        # print("Done")

        driver.close()
        driver.switch_to.window(fingerprinting_tab)
        # print("   • Back to collector tab")

        # print("   • Waiting for trace to finish…", end=" ")
        time.sleep(3)
        # print("Ready")

        trace_data = driver.execute_script("""
            return fetch('/download_traces')
                .then(response => response.json())
                .then(data => (data && data.length) ? data.pop().trace_data : null)
                .catch(() => null);
        """)

        if not trace_data:
            print("   ✗ No trace data received")
            return False

        site_idx = WEBSITES.index(website_url)
        print(f"   • Saving trace (site #{site_idx})…", end=" ")
        if database.db.save_trace(website_url, site_idx, trace_data):
            print("Success ✔")
            return True
        else:
            print("Failed ✗")
            return False

    except Exception as e:
        print(f"   ! Error on {website_url}: {e}")
        traceback.print_exc()
        return False


def collect_fingerprints(driver, target_counts=None):
    """Collect fingerprints until each site has TRACES_PER_SITE traces."""
    total_new = 0
    current = database.db.get_traces_collected()
    for site in WEBSITES:
        remaining = TRACES_PER_SITE - current.get(site, 0)
        print(f"\n─── {site} ({remaining} traces to go) ───")
        while remaining > 0:
            ok = collect_single_trace(driver, WebDriverWait(driver, 20), site)
            if ok:
                total_new += 1
                remaining -= 1
                print(f"   ✔ Collected {TRACES_PER_SITE - remaining}/{TRACES_PER_SITE}")
            else:
                print("   ⚠ Retry collecting…")
            database.db.export_to_json(OUTPUT_PATH)
    return total_new


def main():
    """Main entrypoint for automated trace collection."""
    print("\n=== Fingerprint Trace Collector ===")
    if not is_server_running():
        print("→ Flask backend not running; starting it now…")
        subprocess.Popen([sys.executable, "app.py"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        print("→ Backend should be up")

    print("→ Initializing database…", end=" ")
    database.db.init_database()
    print("Done")

    driver = setup_webdriver()
    total_new = 0
    try:
        while not is_collection_complete():
            added = collect_fingerprints(driver)
            total_new += added
            print(f"\n*** Round complete; +{added} new traces (total so far: {total_new}) ***")
        print(f"\n=== Collection complete! Total new traces collected: {total_new} ===")
        database.db.export_to_json(OUTPUT_PATH)
    except Exception as e:
        print(f"\n!!! Fatal error: {e}")
        traceback.print_exc()
    finally:
        driver.quit()
        print("→ WebDriver closed. Exiting.")

if __name__ == "__main__":
    main()
