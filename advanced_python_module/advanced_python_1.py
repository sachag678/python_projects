
import requests

def check_weather(coord):
	"""Takes coordinates and returns temp(C) and humidity(BPA)"""
	#creates the url from the coord
	partial_url = "http://api.openweathermap.org/data/2.5/weather?lat="
	url = partial_url+coord[0]+"&lon="+coord[1]+"&units=metric&appid=95150fc1272f2a4d44d4e0026fa6f470"
	
	try:
		#requests data from url
		r = requests.get(url, timeout = 1)
		#checks status code
		if(r.status_code==200):
			#converts it into json
			json_object = r.json()

			#gathers relevent info
			temp = json_object['main']['temp']
			hum = json_object['main']['humidity']

			#print
			print("the current temp at latitude " + coord[0] + " and longitude " + coord[1] + " is: " + str(temp) + " C.")
			print("The current humidity is: " + str(hum))
		else:
			#prints error and throws status error
			print("Server is not ready")
			r.raise_for_status();
	except Exception as e:
			print(e)


def get_input():
	"""
	gets user input, does error handling and calls check_weather if the
	the data is valid
	"""
	try:
		lat = input("Enter latitude (between -90 & 90): ")
		lon = input("Enter longitude (between -100 & 180): ")

		if (is_not_number(lat) or is_not_number(lon)):
			raise ValueError("There is a non-numeric input.")
		if(float(lat)<-90.0 or float(lat)>90.0 or float(lon)<-180.0 or float(lon)>180.0):
			raise ValueError("The lat should be -90->90 and lon should be -180->180.")

	except ValueError as e:
		print(e)
	else:
		return  (lat,lon)

def is_not_number(s):
	"""Checks if the string is number and returns true if it is"""
	try:
		float(s)
		bool_a = False
	except:
		bool_a = True

	return bool_a		

def run():
	coordinate = get_input()
	check_weather(coordinate)

#run()

