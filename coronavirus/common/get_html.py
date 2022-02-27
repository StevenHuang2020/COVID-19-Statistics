#python3 unicode
#author:Steven Huang 10/02/20
#function:common get html content by requests

import requests
import urllib.request
import wget
# from user_agent import random_agent


def getUrlByUrllib(url, needDecode=True):
    try:
        with urllib.request.urlopen(url) as response:
            charset = response.info().get_content_charset()
            print('charset = ', charset)
            if needDecode:
                if charset is None:
                    charset = "utf-8"
                return response.read().decode(charset, 'ignore')
            else:
                return response.read()
    except:
        print("Something Wrong by Urllib!")
        return None


def getUrlByRequest(url):
    #headers = random_agent()
    #print(headers)

    try:
        r = requests.get(url, timeout=30)  # headers=headers
        if r.status_code != 200:
            print(r.status_code)

        r.raise_for_status()

        # print('encoding=', r.apparent_encoding)
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print("Something Wrong by requests!")
        return None


def getUrlLocal(url):
    return open(url, "rb").read()


def openUrl(url, save=False, file=r'./a.html'):
    print('start to open url:', url)
    html = getUrlByRequest(url)
    if save:
        saveToFile(html, file)
    return html


def saveToFile(html, file):
    with open(file, "w", encoding='utf-8') as f:
        f.write(html)


def openUrlUrlLib(url, save=False, file=r'./a.html'):
    html = getUrlByUrllib(url)
    if save:
        saveToFile(html, file)
    return html


def download_webfile(url, dst):
    if 0:
        wget.download(url, out=dst)
    else:
        print('Start file download, url=\n', url, '\ndst=', dst)
        r = requests.get(url)
        # Retrieve HTTP meta-data
        status_code = r.status_code
        print('status_code=', status_code)
        # print(r.headers['content-type'])
        # print(r.encoding)
        if status_code == 200:
            with open(dst, 'wb') as f:
                f.write(r.content)
            return True

    return False
