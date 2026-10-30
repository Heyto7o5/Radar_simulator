"""Read individual public ZIP members via bounded HTTP ranges, including ZIP64."""
from __future__ import annotations
import io
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=64)
def fetch_range(url,size,start,n):
    end=start+n-1
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=206:raise IOError('server ignored Range; refusing full archive download')
        expected=f'bytes {start}-{end}/{size}'
        if response.headers.get('Content-Range')!=expected:raise IOError('unexpected Content-Range')
        data=response.read(n+1)
    if len(data)!=n:raise IOError('truncated remote range')
    return data


class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url: str,size: int):
        self.url=url;self.size=size;self.position=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0:raise ValueError('negative seek')
        self.position=position;return position

    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0:return b''
        if n>200_000_000:raise ValueError('single remote read exceeds 200 MB safety limit')
        data=fetch_range(self.url,self.size,self.position,n)
        self.position+=n;return data
