import random
from fake_useragent import UserAgent


agent_list = '''Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50
Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50
Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;)
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)'''


def random_agent():
    headers = agent_list.split('\n')
    length = len(headers)
    return headers[random.randint(0, length - 1)]

def get_random_agent():
    ua = UserAgent(cache=False).random
    #print(ua)
    return ua


def main():
    # agent = get_random_agent()
    agent = random_agent()
    print('agent=', agent)


if __name__ == "__main__":
    main()
