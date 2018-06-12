"""Brain."""
import webbrowser
import sys
from getWeather import get_current_weather


def open_search(query):
    """Default search if not sure what to do."""
    url = 'https://www.google.ca/search?q=' + query
    webbrowser.open_new(url)


if __name__ == '__main__':
    query = sys.argv[1:]
    if query[0] == 'What is the current weather?':
        get_current_weather('Ottawa')
    else:
        open_search(query[0])
