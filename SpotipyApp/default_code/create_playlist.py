
# Creates a playlist for a user

import pprint
import sys
import os
import subprocess

import spotipy
import spotipy.util as util

SPOTIPY_CLIENT_ID='f81f4a6d1096485b9e177e175513c779'
SPOTIPY_CLIENT_SECRET='246d94081f1d47e7a11d0e72f68fd06b'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'

if len(sys.argv) > 2:
    username = sys.argv[1]
    playlist_name = sys.argv[2]
else:
    print("Usage: %s username playlist-name" % (sys.argv[0],))
    sys.exit()

token = util.prompt_for_user_token(username,scope=None,client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI)

if token:
    sp = spotipy.Spotify(auth=token)
    sp.trace = False
    playlists = sp.user_playlist_create(username, playlist_name)
    pprint.pprint(playlists)
else:
    print("Can't get token for", username)