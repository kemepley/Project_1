import pandas as pd
house = pd.read_csv('https://raw.githubusercontent.com/learn-co-students/dsc-v2-mod1-final-project-dc-ds-career-042219/master/kc_house_data.csv')

import math
from ast import literal_eval
def merc(Coords):
#     Coordinates = literal_eval(Coords)
    lat = Coords[0]
    lon = Coords[1]
    
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale
    return (x, y)

house['location'] = list(zip(house['lat'], house['long']))

house['coords_x'] = house['location'].apply(lambda x: merc(x)[0])
house['coords_y'] = house['location'].apply(lambda x: merc(x)[1])
house['map_dot'] = house['price']/700000

from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models import Circle, BasicTicker, ColorBar
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5
from bokeh.io import output_notebook

output_file("tile.html")

# range bounds supplied in web mercator coordinates
p = figure(x_range=(-13638000, -13504000), y_range=(5967000, 6069000),
           x_axis_type="mercator", y_axis_type="mercator", title="House Sale Values in King County Washington")
p.add_tile(CARTODBPOSITRON)

source = ColumnDataSource(
    data=dict(
        lat=house.coords_x.tolist(),
        lon=house.coords_y.tolist(),
        size=house.map_dot.tolist(),
        color=house.price.tolist(),
        legend=house.price.tolist()))

color_mapper = LinearColorMapper(palette="Viridis256", low=78000, high=7070000)

output_notebook()

circle = Circle(x='lat', y='lon', size='size', fill_color={'field': 'color', 'transform': color_mapper}, fill_alpha = .6, line_color=None)

p.add_glyph(source, circle)

color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

p.add_layout(color_bar, 'right')

show(p)