from django.shortcuts import render
from rest_framework import generics
from projet.models import Order
from .serializers import OrderSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .modele_ia import transform
import random
from rest_framework.exceptions import ValidationError
# Create your views here.

class OrderList(generics.ListCreateAPIView):
    queryset  = Order.objects.all()#orderobject
    serializer_class = OrderSerializer
    

class OrderDetail(generics.RetrieveDestroyAPIView):
    Queryset = Order.objects.all()
    serializer_class = OrderSerializer
    
    
#Cette bue prend en  entrée le texte du site(bien l'appeler "text") et renvoie les différents subtrade sous la forme de json intriqués: par exemple si le modele trouve deux subtrades la vue renverra:
# {0 : {text : subtrade, isin : isin du subtrade, etc}
# 1 : {text : subtrade, isin : isin du subtrade, etc}}
#la page interprète ensuite ces données pour les afficher dans des tableaux

class ApplyIaView(APIView):
    
    def post(self,request):
        texte = request.data.get("text")
        subtrades = []
        orders = transform(texte)
        for order in orders:

            order.save()
            serializer = OrderSerializer(order)
            subtrades.append(serializer.data)
        response = {}
        for i in range(len(subtrades)) :
            response[i] = subtrades[i]

        



        return Response(response)
        





