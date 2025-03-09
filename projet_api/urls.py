from django.urls import path
from .views import OrderList, OrderDetail,ApplyIaView


app_name = 'projet_api'

urlpatterns = [
    #path('<int:pk>/',OrderDetail.as_view(),name='detailcreate'),
    #path('',OrderList.as_view(), name='listcreate'),
    path("transform", ApplyIaView.as_view(), name="transform"),
]