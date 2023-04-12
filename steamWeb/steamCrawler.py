#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from http.cookiejar import CookieJar
from urllib.request import build_opener, HTTPCookieProcessor

a_dataframe = pd.read_csv('C:/Users/yttn0/Desktop/git/GameChanger/dataset/games.csv')
a_dataframe

# AppID 속성 추출한 데이터프레임
game_no = a_dataframe['AppID']

# 뽑아내야 할 데이터들 태그 정리 및 리스트
#  metacritic URL 주소 링크 (div id=game_area_metalink  a href)

#  metacritic 페이지에 안에서 가져와야 할 것  
# summary (li class=summary_detail product_summary  span class=data)
# Genre(s) (ul class=summary_details  li class=summary_detail product_genre  span class=data)
# critic reviews (ol class=reviews critic_reviews  li class=review critic_review first_review & li class=review critic_review & li class=review critic_review last_review)
# user reviews (ol class=reviews user_reviews  li class=review user_review first_review & li class=review user_review & li class=review user_review last_review)

# 스팀 페이지에서 가져와야 할 것 (완료)
# CUSTOMER REVIEWS (div class=summary_section  span class=game_review_summary)
# Overall Review
# Recent Reviews 
# MOST HELPFUL REVIEWS (div id=user_reviews_container) 
# Currently popular(지금 유행중인 게임인지 여부) (div class=block responsive_apppage_details_right recommendation_reasons  p class=reason for) 

# first_idx - 탐색 시작할 인덱스 번호  result - 결과값 담을 배열
def start_crawling(first_idx, result):
    driver = webdriver.Chrome('C:/Users/yttn0/My_Python/WebDriver/chromedriver.exe') # chromedriver 연결

    for no in range(first_idx, len(game_no)+1):
        ### URL 가져오기 시작 ###
        url = 'https://store.steampowered.com/app/%d' %(int(game_no[no]))
        print(url)
        driver.get(url)
        time.sleep(3)
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
    except:
        print("Overall Review 또는 Recent Review 데이터가 없습니다.")
    ### Overall Review 및 recent_review 데이터 추출 끝 ###

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
    except:
        print("리뷰가 없습니다.")
    ### 리뷰 부분 종료 ###
    
    ############ Metacritic ##############
    ### metacritic URL 주소 추출 시작 ###
    try:
        metacritic_url = SoupUrl.find('div', attrs={'id':'game_area_metalink'})
        metacritic_url = metacritic_url.find('a')['href']
        print(metacritic_url)
        
#         driver = webdriver.Chrome('C:/Users/yttn0/My_Python/WebDriver/chromedriver.exe') # chromedriver 연결
#         url = "https://www.metacritic.com/game/pc/stardew-valley"
#         print(url)

        try:
            driver.get(metacritic_url)
            time.sleep(3)
            html = driver.page_source
            # 리디랙션 루프 방지
            cj = CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            SoupUrl = BeautifulSoup(html, 'html.parser')

            ### metacritic - summary #### (완)
            metacritic_summary = SoupUrl.find('li', {'class':'summary_detail product_summary'})
            try:
                meta_summary = metacritic_summary.find('span', attrs={'class':'blurb blurb_expanded'}).get_text() 
            except:
                meta_summary = metacritic_summary.find("span", {'class':'data'}).get_text() # 스크립트수가 많은 경우

            print('summary: ' + meta_summary)
            print('---------------------------------')
            ### summary 끝 ####

            ### metacritic - genre #### (완)
            metacritic_genre = SoupUrl.find('li', attrs={'class':'summary_detail product_genre'})
            try:
                find_genre = metacritic_genre.find_all('span', attrs = {'class': 'data'})# 모든 장르
                meta_genre  = ''
                for i in find_genre:
                    meta_genre = meta_genre + i.get_text() + ' '
        #             print(i.get_text())
            except:
                print('genre 데이터가 없습니다')

            print('genre: ' + meta_genre)
            print('--------------------------------')
            ### genre 끝 ####

            ### metacritic - critic review ####

            meta_creview = ''
            try:
                driver.find_element(By.CSS_SELECTOR,"li.nav.nav_critic_reviews > span > span > a").click() # critic review로 이동
        #         driver.find_element(By.CSS_SELECTOR,".see_all > a").click() # critic review로 이동
                metacritic_creview = SoupUrl.find('ol', attrs = {'class':'reviews critic_reviews'})

                # 모든 리뷰(first, last 제외)


                ###### 모든 리뷰가 뽑히지 않음!!!!#####
                find_creview = metacritic_creview.find_all('li', attrs = {'class': 'review critic_review'}) ### error#####
                print(len(find_creview))

                for i in find_creview:
                    meta_creview = meta_creview + i.find('div', attrs = {'class': 'review_body'}).get_text()
        #             print(i.find('div', attrs = {'class': 'review_body'}).get_text())

                # fist, last 리뷰(완)
                # first
                find_f_creview = metacritic_creview.find('li', attrs = {'class': 'review critic_review first_review'})
                meta_creview = meta_creview + find_f_creview.find('div', attrs = {'class': 'review_body'}).get_text()
                #last
                find_l_creview = metacritic_creview.find('li', attrs = {'class': 'review critic_review last_review'})
                meta_creview = meta_creview + find_l_creview.find('div', attrs = {'class': 'review_body'}).get_text()

                print(meta_creview)
                print('----------------------------------------------')
            except:
                print('critic review가 없습니다')
            ### \critic review 끝 ####

            ### metacritic - user review ####
            meta_ureview = ''

        #     pagenum = SoupUrl.find('li', attrs = {'class':'page last_page'}).find('a').get_text()
        #     print(pagenum)


            try:
                driver.find_element(By.CSS_SELECTOR,"li.nav.nav_user_reviews > span > span > a").click() # critic review로 이동
                metacritic_ureview = SoupUrl.find('ol', attrs = {'class':'reviews user_reviews'})

                # 모든 리뷰(first, last 제외)

                ###### 모든 리뷰가 뽑히지 않음!!!!#####
                find_ureview = metacritic_ureview.find_all('li', attrs = {'class': 'review user_review'}) ### error#####
                print(len(find_ureview))

                for i in find_ureview:
                    try: # expand
                        meta_ureview = meta_ureview + i.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text()
                        print(i.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text())
                    except: # expand 없는 경우
                        meta_ureview = meta_ureview + i.find('div', attrs = {'class': 'review_body'}).find('span').get_text()
                        print(i.find('div', attrs = {'class': 'review_body'}).find('span').get_text())
                    print('-------------------------------------------------------')

        #         fist, last 리뷰(완)
                #first
                find_f_ureview = metacritic_ureview.find('li', attrs = {'class': 'review user_review first_review'})
                try: # expand
                    meta_ureview = meta_ureview + find_f_ureview.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text()
                    print('first: '+ find_f_ureview.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text())
                except: # expand 없는 경우
                    meta_ureview = meta_ureview + find_f_ureview.find('div', attrs = {'class': 'review_body'}).find('span').get_text()
                    print('first: '+find_f_ureview.find('div', attrs = {'class': 'review_body'}).find('span').get_text())

                #last
                find_l_ureview = metacritic_ureview.find('li', attrs = {'class': 'review user_review last_review'})
                meta_ureview = meta_ureview + find_l_ureview.find('div', attrs = {'class': 'review_body'}).find('span').get_text()
                try: # expand
                    meta_ureview = meta_ureview + find_l_ureview.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text()
        #             print('last: ' +find_l_ureview.find('div', attrs = {'class': 'review_body'}).find('span', attrs = {'class':'blurb blurb_expanded'}).get_text())
                except: # expand 없는 경우
                    meta_ureview = meta_ureview + find_l_ureview.find('div', attrs = {'class': 'review_body'}).find('span').get_text()
        #             print('last: ' + find_l_ureview.find('div', attrs = {'class': 'review_body'}).find('span').get_text())

        #         print(meta_ureview)
            except:
                print('user review가 없습니다')
            ### \critic review 끝 ####

        except:
            print("url 실패")
    except:
        print("Metacritic 링크가 존재하지 않습니다.")
            ### metacritic URL 주소 추출 끝 ###

    steam_value = [overall_reviews] + [recent_reviews] + [metacritic_url] + [reviews_total]
    print('steam_value: '+ steam_value)
    result.append(steam_value)
    
    meta_value = [metacritic_summary] + [metacritic_genre]
    print('meta_Value: ' + meta_value)
    result.append(meta_value)
        
def main():
    result=[]
    start_crawling(0, result) # 임시로 0번째 인덱스(처음)부터 시작하게끔 설정한 상태
    Wcrawling_tbl = pd.DataFrame(result, columns=('Overall Reviews', 'Recent Reviews', 'Most Helpful Reviews', 'Currently popular'))
    Wcrawling_tbl.to_csv('CUsersravengitgameChangerdatasetsteamGames.csv', encoding='cp949', mode='w', index=True)
    del result[:]
if __name__ == '__main__':
    main()

