import spotipy
import sys
import pprint
sp = spotipy.Spotify()

if len(sys.argv) > 1:
    artist_name = ' '.join(sys.argv[1:])
    results = sp.search(q=artist_name, limit=5)
    #pprint.pprint(results)
    for i, t in enumerate(results['tracks']['items']):
        print(' ', i, t['name'], ': ', t['id'])