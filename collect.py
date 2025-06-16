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

TRACES_PER_SITE = 3
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
        # Open fingerprinting page
        driver.get(FINGERPRINTING_URL)
        # Ensure page loaded
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        # Find and click the "Collect Trace" button
        collect_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Collect trace')]")))
        collect_button.click()
        print("  - Started trace collection monitoring...")
        time.sleep(2)  # Give time for monitoring to initialize
        
        # Store the fingerprinting tab handle
        fingerprinting_tab = driver.current_window_handle
        
        # Open target site in a new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        print(f"  - Opening target website: {website_url}")
        driver.get(website_url)
        
        # Wait for the target website to load
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2)  # Additional wait for dynamic content
        
        # Simulate user interaction on target site
        print("  - Simulating user interaction...")
        height = driver.execute_script("return document.body.scrollHeight;")
        for _ in range(random.randint(3, 6)):
            scroll = random.randint(0, height)
            driver.execute_script(f"window.scrollTo(0, {scroll});")
            time.sleep(random.uniform(0.5, 1.0))
        
        # Close target tab and return to fingerprinting tab
        driver.close()
        driver.switch_to.window(fingerprinting_tab)
        print("  - Returned to fingerprinting tab")
        
        # Wait for trace collection to complete
        print("  - Waiting for trace collection to finish...")
        time.sleep(3)  # Give time for trace collection
        
        # Get the trace data
        trace_data = driver.execute_script("""
            return fetch('/download_traces')
                .then(response => response.json())
                .then(data => {
                    if (data && data.length > 0) {
                        return data[data.length - 1].trace_data;
                    }
                    return null;
                })
                .catch(() => null);
        """)
        
        if trace_data:
            # Save to database
            site_idx = WEBSITES.index(website_url)
            print(f"  - Saving trace to database for {website_url} at index {site_idx}")
            if database.db.save_trace(website_url, site_idx, trace_data):
                print("  - Successfully saved trace to database")
                return True
            else:
                print("  - Failed to save trace to database")
                return False
        else:
            print("  - No trace data found")
            return False
            
    except Exception as e:
        print(f"Error collecting trace for {website_url}: {e}")
        traceback.print_exc()
        return False


def collect_fingerprints(driver, target_counts=None):
    """Collect fingerprints until each site has TRACES_PER_SITE traces."""
    new_count = 0
    current = database.db.get_traces_collected()
    targets = {site: TRACES_PER_SITE - current.get(site, 0) for site in WEBSITES}
    for site, remaining in targets.items():
        print(f"Collecting {remaining} traces for {site}")
        while remaining > 0:
            success = collect_single_trace(driver, WebDriverWait(driver, 20), site)
            if success:
                new_count += 1
                remaining -= 1
                print(f"  - Collected trace #{TRACES_PER_SITE - remaining} for {site}")
            else:
                print(f"  - Failed to collect trace for {site}, retrying...")
            # periodic save
            database.db.export_to_json(OUTPUT_PATH)
    return new_count


def main():
    """Main entrypoint for automated trace collection."""
    # Ensure Flask backend is running
    if not is_server_running():
        print("Starting Flask server...")
        subprocess.Popen([sys.executable, "app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    
    # Initialize database
    print("Initializing database...")
    database.db.init_database()
    
    # Setup WebDriver
    driver = setup_webdriver()
    try:
        total_new = 0
        while not is_collection_complete():
            total_new += collect_fingerprints(driver)
        print(f"Collection complete. Total new traces: {total_new}")
        database.db.export_to_json(OUTPUT_PATH)
    except Exception as e:
        print(f"Fatal error during collection: {e}")
        traceback.print_exc()
    finally:
        driver.quit()
        print("WebDriver closed.")

if __name__ == "__main__":
    main()
