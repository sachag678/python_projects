import pprint
import requests
import spotipy
import subprocess
import sys
import os
import spotipy.util as util

SPOTIPY_CLIENT_ID='f81f4a6d1096485b9e177e175513c779'
SPOTIPY_CLIENT_SECRET='246d94081f1d47e7a11d0e72f68fd06b'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'

def generate_playlist(songs):
	"""Takes a list of songs as a parameter and returns a url that contains the the playlist containing those songs"""

	username = '22wnnsp2nhoeqfiohtbrrv2ji'
	playlist_name = 'new_playlist'
	scope = 'playlist-modify-public'

	#get authorization
	token = util.prompt_for_user_token(username,scope,client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI)

	if token:
		sp = spotipy.Spotify(auth=token)
		sp.trace = False
		#create new playlist
		playlists = sp.user_playlist_create(username, playlist_name)
		playlist_id = playlists['id']

		#creat list of track ids
		songs = songs[0:10]
		track_ids = [sp.search(q=song, type = 'track')['tracks']['items'][0]['id'] for song in songs]
		
		#add tracks to playlist
		sp.user_playlist_add_tracks(username, playlist_id, track_ids)

		#print url
		playlists = sp.user_playlist(username, playlist_id)
		pprint.pprint(playlists['external_urls']['spotify'])
	else:
		print("can't get token for: " + username)

# songs = ['Enter Sandman',  'Nothing Else Matters', 'The Unforgiven', 'Sad But True', 'Master Of Puppets']
# ids = ['1hKdDCpiI9mqz1jVHRKG0E', '3ZFwuJwUpIl0GeXsvF1ELf','5SnOyuBtyzufoXBAKOdcxD','5pQYjzkALsgYOcFTC8DMmU','6NwbeybX6TDtXlpXvnUOZC']
# generate_playlist(songs)