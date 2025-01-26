import httpx 
from parsel import Selector
import dateparser
from csv import DictWriter


def letra(url):
    response = httpx.get(url)

    s = Selector(response.text)

    letra = '\n'.join(s.css('[data-lyrics-container]::text').getall())
    return letra


def faixas(url):
    response = httpx.get(url)
    s = Selector(response.text)
    musicas = s.css('div.chart_row-content')
    return [
        (
            musica.css('h3::text').get().strip(),
            musica.css('a').attrib['href'],
        ) for musica in musicas
    ]


def discos(url):
    response = httpx.get(url)
    s = Selector(response.text)
    discos = s.css('.ZvWhZ')

    return [
        (
            disco.css('.kqeBAm').attrib['href'],
            disco.css('.gpuzaZ::text').get(),
            disco.css('.cedmJJ::text').get(),
        ) for disco in discos
    ]


url = 'https://genius.com/artists/Dead-fish/albums'


with open('deadfish.csv', 'w') as file:
    writer = DictWriter(file, ['album', 'data', 'musica', 'letra'])
    writer.writeheader()
    for disco in discos(url):
    
        for faixa in faixas(disco[0]):
            row = {
                'album': disco[1],
                'data': dateparser.parse(disco[2]),
                'musica': faixa[0],
                'letra': letra(faixa[1])
            }

            writer.writerow(row)


