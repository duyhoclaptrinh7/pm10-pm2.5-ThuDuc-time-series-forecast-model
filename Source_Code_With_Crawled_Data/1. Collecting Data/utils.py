import time, re, pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np

def create_url(station_code, month, year):
    return f"https://www.timeanddate.com/weather/@{station_code}/historic?month={month}&year={year}"

def to_YYYYMMDD(date_text):
    import datetime
    dt = datetime.datetime.strptime(date_text, "%d %B %Y")
    return dt.strftime("%Y-%m-%d")


def get_month_data(url):
    full_hours = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # bật nếu muốn chạy ngầm
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    wait = WebDriverWait(driver, 20)
    driver.get(url)

    def safe_texts(by, value, retries=10, delay=1):
        """Lấy text từ các element, retry nếu StaleElementReferenceException"""
        for _ in range(retries):
            try:
                elements = driver.find_elements(by, value)
                return [el.text.strip() for el in elements]
            except:
                time.sleep(delay)
        return []

    wait.until(EC.presence_of_element_located((By.ID, "wt-his-select")))
    df_month = pd.DataFrame()

    def safe_temperature(by="td.wt-ic", retries=10, delay=1):
        """Lấy nhiệt độ °C từ td kế tiếp td.wt-ic, retry nếu StaleElement"""
        for _ in range(retries):
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, by)
                temps = []
                for td in elements:
                    next_td = td.find_element(By.XPATH, "following-sibling::td[1]")
                    text = next_td.text.strip()
                    match = re.search(r"(\d+)", text)
                    temps.append(int(match.group()) if match else np.nan)
                return temps
            except:
                time.sleep(delay)
        return []
    

    # Lặp qua các ngày
    num_options = len(Select(driver.find_element(By.ID, "wt-his-select")).options)
    for i in range(num_options):
        select_elem = Select(driver.find_element(By.ID, "wt-his-select"))
        option = select_elem.options[i]
        date_text = option.text
        date_value = option.get_attribute("value")
        select_elem.select_by_value(date_value)
        print("Đang chọn ngày:", date_text)

        # Chờ table render xong
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "th")))

        # Time
        th_texts = safe_texts(By.TAG_NAME, "th")
        if i == 8: print(th_texts)
        hours_list = []
        for text in th_texts:
            text = text.split("\n")[0]
            if ":" in text:
                try:
                    h, m = map(int, text.split(":"))
                    if 0 <= h <= 23 and 0 <= m <= 59:
                        hours_list.append(text)
                except:
                    continue
        
        # Temperature
        temperature_list = safe_temperature()


        # Weather
        weather_texts = safe_texts(By.CSS_SELECTOR, "td.small")

        # Wind
        sep_texts = safe_texts(By.CSS_SELECTOR, "td.sep")
        wind = []
        for text in sep_texts:
            if "No wind" in text:
                wind.append(0)
            elif "km/h" in text:
                match = re.search(r"\d+", text)
                if match:
                    wind.append(int(match.group()))

        # Humidity
        td_texts = safe_texts(By.TAG_NAME, "td")
        humidity_list = []
        for text in td_texts:
            if "%" in text:
                match = re.search(r"\d+", text)
                if match:
                    humidity_list.append(int(match.group()))
        humidity_list = humidity_list[3:]  # bỏ 3 giá trị đầu nếu cần

        # Barometer
        pressure_list = []
        for text in sep_texts:
            if "mbar" in text:
                match = re.search(r"\d+", text)
                if match:
                    pressure_list.append(int(match.group()))
        pressure_list = pressure_list[3:]  # bỏ 3 giá trị đầu nếu cần
        print(f"  - Lấy được {len(hours_list)} giờ, {len(weather_texts)} weather, {len(wind)} wind, {len(humidity_list)} RH, {len(pressure_list)} P")
        missing_hours = [h for h in full_hours if h not in hours_list]
        while len(hours_list) < len(pressure_list): hours_list.append(None)
        print("Các giờ bị thiếu:", missing_hours)
        
        # Tạo DataFrame

        max_len = max(len(hours_list), len(temperature_list), len(weather_texts), len(wind), len(humidity_list), len(pressure_list))
        def pad_nan(lst, target_len):
            return lst + [np.nan] * (target_len - len(lst))
        
        hours_list = pad_nan(hours_list, max_len)
        temperature_list = pad_nan(temperature_list, max_len)
        weather_texts = pad_nan(weather_texts, max_len)
        wind = pad_nan(wind, max_len)
        humidity_list = pad_nan(humidity_list, max_len)
        pressure_list = pad_nan(pressure_list, max_len)
        
        df_day = pd.DataFrame({
            "time": hours_list,
            "temp": temperature_list,
            "weather": weather_texts,
            "wind": wind,
            "RH": humidity_list,
            "P": pressure_list
        })
        df_day["time"] = to_YYYYMMDD(date_text) + "T" + df_day["time"]

        df_month = pd.concat([df_month, df_day], ignore_index=True)

    driver.quit()
    return df_month