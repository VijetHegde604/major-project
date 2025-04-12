from django.shortcuts import render
from django.http import HttpResponse
from .utils import extract_lat_long

# Create your views here.
def extract_location(request):
    coords = extract_lat_long("/home/vijeth/major-project/backend/PXL_20240921_055604115.jpg")
    return HttpResponse(f"<h1>Latitude = {coords[0]} Logitude = {coords[1]}")