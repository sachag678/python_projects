import scrape
import spotify

def create_playlist(artist):
	"""Calls the two previous functions, takes an artist and generates a playlist of the artists songs"""
	spotify.generate_playlist(scrape.print_songs(artist))

artist = input("Please enter an artist name: ")
create_playlist(artist)