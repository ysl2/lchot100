import json

import selenium
import webdriver_manager.chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

service = selenium.webdriver.ChromeService(
    webdriver_manager.chrome.ChromeDriverManager().install()
)
options = selenium.webdriver.ChromeOptions()
options.add_argument('--headless')
driver = selenium.webdriver.Chrome(service=service, options=options)

driver.get('https://leetcode.cn/studyplan/top-100-liked')
res = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="__NEXT_DATA__"]'))
)
res = res.get_attribute('innerHTML')
res = json.loads(res)

with open('main.md', 'w') as f:
    for planSubGroup in res['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']['studyPlanV2Detail']['planSubGroups']:
        f.write(f'## {planSubGroup["name"]}\n')
        for question in planSubGroup['questions']:
            f.write(f'### [{question["questionFrontendId"]}. {question["translatedTitle"]} ({question["title"]})](https://leetcode.cn/problems/{question["titleSlug"]}/description)\n')
