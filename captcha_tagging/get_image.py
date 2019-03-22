import urllib.request

for i in range(1, 1000):
    print(i)
    url = 'http://www.yb.cc/code?t=%s' % str(1552640517290+i)
    urllib.request.urlretrieve(url, '/Users/lianmingjie/Pictures/code%d.png' % i)