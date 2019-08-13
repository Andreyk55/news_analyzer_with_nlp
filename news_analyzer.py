import pandas
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from text_translation import translate_list

from sentiment_analysis import sentiment


def get_url(website_name):
    if website_name == 'Ynet':
        return "https://www.ynet.co.il/home/0,7340,L-8,00.html"

def main():
    # Note - commented code because translations are limited.

    # hebrew_ynet_data = get_clean_data_in_list('Ynet')

    # english_ynet_data = list()

    # write_list_to_file("output.txt", hebrew_ynet_data)

    # translate_list(hebrew_ynet_data, english_ynet_data)

    # write_list_to_file("output_english.txt", english_ynet_data)

    clean_english_ynet_data = list()
    clean_english_ynet_data = open('output_english.txt', 'r', encoding="utf-8").read().splitlines()

    # Sentiment analysis:
    for title in clean_english_ynet_data:
        (title_sentiment, algoritm_confidence) = sentiment(title)
        print(str(title) + '     **********' '  sentiment - ' + str(title_sentiment) +
              ', algoritm confidence -  ' + str(algoritm_confidence))


def ynet_clean_dirty_data(dirty_data_list):
# Here there is no need for cleaning data.
    pass

    return dirty_data_list

def get_clean_data_in_list(website_name):
# Gets html data from website.
    html = urlopen(get_url(website_name))

    page = BeautifulSoup(html, "html.parser")

    if website_name == 'Ynet':
        dirty_data_list = ynet_create_list_from_html_page(page)

    if website_name == 'Ynet':
        clean_data_list = ynet_clean_dirty_data(dirty_data_list)

    return clean_data_list

def clean_dirty_data(in_list):
    pass

def ynet_create_list_from_html_page(page):
    """
    * This function returns all titles from web-page.

    how to get data from website:
    1. Titles:
        <a> + <div> tags && class = title
    2. Sub titles:
        <a> tags + class = mta_title
    classes: "title", "mta_title"

    - Note: 1 + 2 covers all data and a few more unnecessary sentences.
    """
    hypers = list()
    # 1:
    hypers.append(page.find_all('a', class_ = 'title'))
    hypers.append(page.find_all('div', class_ = 'title'))

    # 2:
    hypers.append(page.find_all('a', class_ = 'mta_title'))
    hypers.append(page.find_all('a', class_ = 'sub_title sub_title_no_credit'))
    hypers.append(page.find_all('a', class_ = 'multiimagesnews_main_title'))
    hypers.append(page.find_all('a', class_ = 'rpphp_main_title'))
    hypers.append(page.find_all('a', class_ = 'pphp_main_title'))
    hypers.append(page.find_all('a', class_ = 'MultiImagesLeft_main_title'))

    # TODO: write function for this (there was a type mismatch here)
    hypers_un = hypers[0] + hypers[1] + hypers[2] +hypers[3] + hypers[4] +hypers[5] + hypers[6] + hypers[7]

    res_list = list()
    for i in range(len(hypers_un)):
        res_list.append(hypers_un[i].text)
    # Debbug:
    #write_list_to_file("output.txt", res_list)
    return res_list

def write_list_to_file (file, list):
    output_file = open(file, 'w', encoding="utf-8")
    for i in range(len(list)):
        output_file.write(str(list[i]) + '\n')
    # output_file.close()

def print_title(in_list, word):
    for sentence in in_list:
        if word in sentence:
            print(sentence)


if __name__=='__main__':
    main()


