"""get weather forecast."""
import requests


def get_current_weather(city):
    """Get the current weather for the given city."""
    r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=' + city + ',CA&units=metric&appid=95150fc1272f2a4d44d4e0026fa6f470')
    json_object = r.json()
    temp_c = float(json_object['main']['temp'])
    print('The weather in', city, 'is', temp_c, 'celcius.')
