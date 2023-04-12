import re
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import time
from selenium import webdriver


# 반드시 확인!! 스팀 Kaggle 데이터 읽기  경로만 본인 환경에 맞게 재설정
a_dataframe = pd.read_csv('../dataset/games.csv') 
# AppID 속성 추출한 데이터프레임 총 70210 개
game_no = a_dataframe['AppID']

def saveGameData(gameIdx, gameCount):
  savef = open("read.txt",'w')
  print(str(gameIdx))
  savef.write(str(gameIdx)+','+str(gameCount)+'\n')
  savef.close()

# first_idx - 탐색 시작할 인덱스 번호  result - 결과값 담을 배열
def start_crawling(first_idx, fileSave):
  driver = webdriver.Chrome('../WebDriver/chromedriver.exe') # chromedriver 연결

  for no in range(first_idx, len(game_no)+1):
    result = []
    ### URL 가져오기 시작 ###
    url = 'https://store.steampowered.com/app/%d' %(int(game_no[no]))
    print(url)
    driver.get(url)
    time.sleep(2)
    html = urllib.request.urlopen(url)
    driver.execute_script("window.scrollTo(0, 5000);") # 스크롤 내린 상태로 Parsing 진행
    time.sleep(10)
    html = driver.page_source
    SoupUrl = BeautifulSoup(html, 'html.parser')
    ### URL 가져오기 끝 ###

    tag_name_overall_reviews = SoupUrl.find('div', attrs={'id':'review_histogram_rollup_section'}) # Overall Reviews 정보
    tag_name_recent_reviews = SoupUrl.find('div', attrs={'id':'review_histogram_recent_section'}) # Recent Reviews 정보

    ### Overall Review 및 recent_review 데이터 추출 시작 ###
    try:
      overall_reviews = tag_name_overall_reviews.find('span', attrs={'class':'game_review_summary'})
      overall_reviews = overall_reviews.get_text()

      recent_reviews = tag_name_recent_reviews.find('span', attrs={'class':'game_review_summary'})
      recent_reviews = recent_reviews.get_text()
    except AttributeError as e:
      print("Overall Review 또는 Recent Review 데이터가 없습니다.")
    ### Overall Review 및 recent_review 데이터 추출 끝 ###
    ### metacritic URL 주소 추출 시작 ###
    try:
      metacritic_url = SoupUrl.find('div', attrs={'id':'game_area_metalink'})
      metacritic_url = metacritic_url.find('a')['href']
    except:
      print("Metacritic 링크가 존재하지 않습니다.")
    ### metacritic URL 주소 추출 끝 ###
    ### 리뷰 부분 시작 ###
    reviews_total = []
    try:
      # 리뷰있는 부분 찾기
      tag_name_most_helpful_reviews = SoupUrl.find('div', attrs={'id':'Reviews_summary', 'class':'user_reviews_container'})    
      most_helpful_reviews = tag_name_most_helpful_reviews.select("div>div.leftcol>div.review_box")

      for c_review in most_helpful_reviews:
        rec = c_review.find("div", attrs={'class':'title ellipsis'}).get_text() # 추천 or 비추천
        content = c_review.find("div", attrs={'class':'content'}).get_text() # 리뷰 내용

        customer_review = rec + ", "  + content
        reviews_total.append(customer_review)
    except AttributeError as e:
      print("리뷰가 없습니다.")
    ### 리뷰 부분 종료 ###
    ### About This Game - Game Description 추출 시작 ###
    try:
      tag_about_this_game = SoupUrl.find('div', attrs={'id':'aboutThisGame'})
      game_description = tag_about_this_game.find('div', attrs={'id':'game_area_description'}).get_text()
      
      pattern = r'(?<=About This Game).*'
      g_des = re.findall(pattern, game_description, re.S)
      game_description = g_des[0].strip()
    except:
      print("Game Description이 존재하지 않습니다.")
    ### About This Game - Game Description 추출 끝 ###
    ### dataset 저장 시작 ###
    steam_value = [overall_reviews] + [recent_reviews] + [metacritic_url] + [reviews_total] + [game_description]
    print(steam_value)
    result.append(steam_value)
    Wcrawling_tbl = pd.DataFrame(result, columns=('Overall Reviews', 'Recent Reviews', 'Most Helpful Reviews', 'Currently popular', 'Game Description'))
    try:
      Wcrawling_tbl.to_csv('../dataset/steamGameNo%d.csv' %(int(no)), encoding='UTF-8', mode='w', index=True)
    except UnicodeEncodeError as e:
      print("지원하지 않는 Unicode가 포함되어 있습니다.", e)

    saveGameData(no, fileSave)
    fileSave += 1
    ### dataset 저장 종료 ###

def main():
  #초기 실행 이후 주석 해제
  f = open("read.txt",'r')
  data = f.read()
  f.close()
  saveData = data.split(',')
  start_idx = int(saveData[0])
  fileSave = int(saveData[1])
  #
  
  #초기 실행 이후 주석 처리
  # start_idx = 0
  # fileSave = 0
  #
  start_crawling(start_idx, fileSave)

if __name__ == '__main__':
  main()