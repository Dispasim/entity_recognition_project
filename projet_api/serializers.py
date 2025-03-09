from rest_framework import serializers
from projet.models import Order

class OrderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = ("text","isin","trade_date","settlement_date","primary_brocker","sens","dealer","trader","price","size","price_type","currency")
        
