from django import forms


class CreateSubtradeForm(forms.Form):
    text_subtrade = forms.CharField(label = "",max_length=100,widget=forms.Textarea(attrs={'rows': 20, 'cols': 150,'class': 'boites-container'}))