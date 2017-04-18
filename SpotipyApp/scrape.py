import requests
from bs4 import BeautifulSoup

def print_songs(artist):
	"""takes an artist and queries the url and prints the songs of the artist"""
	url = "http://www.setlist.fm/search?query="
	r = requests.get(url+artist)

	try:
		if(r.status_code==200):
			soup = BeautifulSoup(r.content, "lxml")

			songs = []

			for link in soup.find_all(attrs={"class":"setSummary"}): #tried list comprehension here but didnt work very well
				for el in link.find_all("li"):
				 	if not songs.__contains__(el.text):
				 		songs.append(el.text)
				 		#print(el.text.replace("'",""))

			if(not songs):
				print("No songs by this artist")
			else:	 		
				print(songs)
				return songs
		else:
			print("Server is not ready")
			r.raise_for_status();

	except Exception as e:
		print(e)



# artist = input("Please Enter artist name: ")
# print_songs(artist)






