"""
Notes
- 3.65" x 3.7" (3.712")
"""
import os
from solid import *
from solid.utils import *
d = cube(5) + right(5)(sphere(5)) - cylinder(r=10, h=6)

scad_render_to_file(d, os.path.join(os.path.dirname(__file__), "result.scad"))
