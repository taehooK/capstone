import os
from os import listdir
from os.path import join
from os.path import isfile
import requests

from django.shortcuts import render, redirect
from tensorflow.keras import backend as K
from rest_framework.decorators import api_view

from models.MusicGenerator import MusicGenerator
from django.http import FileResponse

def index(request):
    return render(request, 'index.html')

@api_view(['GET'])
def download(request):
    return FileResponse(open('models/samples/example.midi', 'rb'), content_type='audio/midi')

def endPage(request):
    K.clear_session()
    filename = 'example'
    music_gen = MusicGenerator()
    score = music_gen.Generate()
    music_gen.notes_to_midi('models/', score, filename)
    music_gen.notes_to_png('models/', score, filename)

    return render(request, 'endPage.html')
