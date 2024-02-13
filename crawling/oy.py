import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_info(page):
    Company = [] ##회사명
    Product = [] ## 제품명
    Image = []   ## 이미지
    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get('https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100070007&fltDispCatNo=&prdSort=01&pageIdx='+str(page)+'&rowsPerPage=24&searchTypeSort=btn_thumb&plusButtonFlag=N&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0&trackingCd=Cat100000100070007_Small')
    time.sleep(0.5)

    company = driver.find_element(By.CSS_SELECTOR,'#Contents>ul:nth-child(7)>li.flag>div>div>a>span').text ## 1행1열에 제품
    print(company)
    product = driver.find_element(By.CSS_SELECTOR,'#Contents>ul:nth-child(7)>li.flag>div>div>a>p').text
    #image = driver.find_elements(By.CLASS_NAME,'prd_info').get_attribute('src')
    Company.append(company)
    Product.append(product)
    #Image.append(image)
    for i in range(2,5): ## 두번째 품목부터 append
        for j in range(7,13):
            company = driver.find_element(By.CSS_SELECTOR,'#Contents>ul:nth-child('+str(j)+')>li:nth-child('+str(i)+')>div>div>a>span').text
            product = driver.find_element(By.CSS_SELECTOR,'#Contents>ul:nth-child(7)>li:nth-child('+str(j)+')>div>div>a>p').text
            Company.append(company)
            Product.append(product)
   
    df = pd.DataFrame({
        '회사명': Company,
        '제품명': Product,
        #'이미지': Image
        })

    df.to_csv('olive'+str(page)+'.csv', encoding='euc-kr')

get_info(1)# 1페이지에 있는 정보 추출 for 문으로 끝까지 반복
