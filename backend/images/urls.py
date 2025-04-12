from django.urls import path
from .views import extract_location

urlpatterns = [
    path("extract/", extract_location)
]