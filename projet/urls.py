from django.urls import path
from django.views.generic import TemplateView


app_name = 'projet'

urlpatterns = [
    
   
    
    

    path('',TemplateView.as_view(template_name="projet/template.html")),
]