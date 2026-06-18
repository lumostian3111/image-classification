"""截图工具：使用 Playwright 对 Web 应用各页面截图"""
import os, sys
from playwright.sync_api import sync_playwright

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(output_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1024, "height": 768})

    # 1. 预测页
    print("截图: 预测页...")
    page.goto("http://localhost:5000/", wait_until="networkidle")
    page.wait_for_timeout(500)
    page.screenshot(path=os.path.join(output_dir, "01_predict.png"), full_page=True)

    # 获取所有 tab 按钮，按索引点击
    tabs = page.query_selector_all(".tab-btn")
    print(f"  找到 {len(tabs)} 个标签按钮")

    # 2. 训练页 (index 1)
    print("截图: 训练页...")
    tabs[1].click()
    page.wait_for_timeout(400)
    page.screenshot(path=os.path.join(output_dir, "02_train.png"), full_page=True)

    # 3. 评估页 (index 2)
    print("截图: 评估页...")
    tabs[2].click()
    page.wait_for_timeout(400)
    page.screenshot(path=os.path.join(output_dir, "03_evaluate.png"), full_page=True)

    # 4. 数据管理页 (index 3)
    print("截图: 数据管理页...")
    if len(tabs) >= 4:
        tabs[3].click()
        page.wait_for_timeout(600)
        page.screenshot(path=os.path.join(output_dir, "04_data.png"), full_page=True)
    else:
        print("  ⚠ 只有 3 个标签页，跳过数据管理截图")

    browser.close()
    print(f"\n全部截图完成! 保存在: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")
