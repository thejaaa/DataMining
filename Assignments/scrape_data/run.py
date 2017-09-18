#importing the libraries.
import requests
from bs4 import BeautifulSoup
import json


#list.
restaurant_details = []

for i in range(9,10):
    letter = chr(i+60)
    #url of the website we want to parse.
    url = "https://www.allmenus.com/custom-results/-/ny/new-york/a/"
    print(url)
    
    # Beautiful soup way
    html = requests.get(url).text
    # parsing html using beautiful soap, and store that in soup varaible.
    soup = BeautifulSoup(html,'html.parser')
    #extracting all the content with findAll().
    newyork = soup.findAll('li',{'class',"restaurant-list-item clearfix"})
    for restaurants in newyork:
        #getting values using a unique class name.
        subrest = restaurants.findNext('div',{'class':'s-rows'})
        # finding the restaurant names in the newyork.
        name = restaurants.findNext('h4').text
        #print(name)
        ad = restaurants.findNext('p')
        #finding the adress of the restaurants.
        adress = ad.findNext('p').text
        
        # dictionary is defined.
        restaurant_dict = {}
        restaurant_dict['RestaurantName'] = name
        restaurant_dict['Adress'] = adress
        #appending all the dictionaries into a list.
        restaurant_details.append(restaurant_dict)
        #time.sleep(random.randint(0,3))
           
# printingthe restuarant_details.
print(restaurant_details)


#writing the output to a json file.
with open('data.json', 'wt') as outfile:
    result=json.dump(restaurant_details, outfile,indent=4,separators=(',', ': '))
   