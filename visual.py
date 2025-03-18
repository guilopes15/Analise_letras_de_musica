import matplotlib
import matplotlib.pyplot as plt
import polars as pl


matplotlib.use('Agg')

df = pl.read_csv('deadfish_stats.csv')

# counter = df.group_by('album').len().sort('len')
# fig, ax = plt.subplots()
# ax.barh(counter['album'], counter['len'])


for index, album in enumerate(df.sort('data').partition_by('album')):
    album = album.sort('tokens')

    mean = album['ttr'].mean()
    
    fig, ax = plt.subplots()
     
    ax.barh(album['musica'], album['tokens'])
    ax.barh(album['musica'], album['types'])
    ax.barh(album['musica'], album['ttr'] )
    ax.axvline(x=mean)
    
    ax.set_title(f"{album['album'][0]} - {album['data'][0]}")
    ax.set_xlabel('Estatísticas de texto')
    ax.set_ylabel('Nome da música')
     
    ax.legend(
        (
            'Media de TTR',
            'Tokens (N): Quantidade de palavras', 
            'Types (N): Quantidade de palavras únicas', 
            'TTR (%): Relação entre Tokens e Types',
        ), 
        loc='lower right', 
        fontsize=7, 
        handletextpad=0.3, 
        labelspacing=0.3
    )
    
    plt.savefig(
        f'graficos/{album['album'][0]}.png', dpi=300, bbox_inches='tight'
    )

boxes = {}
target = 'tokens'

for index, album in enumerate(df.sort('data').partition_by('album')):
    boxes |= {
        album['album'][0]: album[target].drop_nulls()
    }

fig, ax = plt.subplots()

median = df[target].median()
ax.axhline(y=median)

ax.boxplot(boxes.values())
ax.set_xticklabels(boxes.keys(), rotation=45, ha="right")

plt.savefig(f'graficos/boxplot.png', dpi=300, bbox_inches='tight')