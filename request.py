#pip install requests beautifulsoup4 selenium pillow
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO

# 定义下载文件夹
download_folder = "东方明珠图片"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# 确保tower文件夹存在
tower_dir = os.path.join(os.path.dirname(__file__), 'tower')
os.makedirs(tower_dir, exist_ok=True)

# 设置 Chrome 驱动路径（需根据你的环境设置）
chrome_driver_path = 'path/to/chromedriver'

# 创建浏览器实例
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 隐藏浏览器窗口
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)

# 打开 Google 图片搜索页面
search_query = "上海 东方明珠"
driver.get(f"https://www.google.com/search?hl=en&tbm=isch&q={search_query}")

# 获取页面上的图片
image_count = 0
scroll_pause_time = 2  # 每次滚动等待的时间（秒）
scroll_height = driver.execute_script("return document.body.scrollHeight")  # 获取页面的高度

while True:
    # 获取页面上的所有图片元素
    images = driver.find_elements(By.XPATH, '//*[@class="Q4LuWd"]')
    for image in images[image_count:]:
        try:
            # 点击图片查看原图
            image.click()
            time.sleep(1)

            # 获取图片的 URL
            img_url = driver.find_element(By.XPATH, '//*[@class="n3VNCb"]').get_attribute("src")
            if img_url.startswith('http'):
                # 下载图片并保存到本地
                img_data = requests.get(img_url).content
                img_name = os.path.join(download_folder, f"{search_query}_{image_count}.jpg")
                with open(img_name, 'wb') as f:"backspace"