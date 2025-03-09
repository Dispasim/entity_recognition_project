from django.contrib import admin
from . import models
# Register your models here.

@admin.register(models.Order)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ("text","isin","trade_date","settlement_date","primary_brocker","sens","dealer","trader","price","size","price_type","currency")


admin.site.register(models.Category)