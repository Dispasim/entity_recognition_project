

from projet.models import Order
from collections import Counter
import random
from datetime import date
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
import joblib
import re
from datetime import datetime, timedelta
import pandas as pd
import spacy

label_encoder_sens = joblib.load("files/label_encoder_sens.pkl")
tokenizer_sens = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_sens = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder_sens.classes_))
model_sens.load_state_dict(torch.load("files/model_sens.pth"))
model_sens.eval()
df_dealer = pd.read_csv('files/ISIN-Dealer-Trader.csv',sep=",")
df_broker = pd.read_csv("files/Brokers.csv",sep=",")
df_dealer = df_dealer.dropna()
traders = []
df_traders = df_dealer["Traders"].drop_duplicates().tolist()

for traders_list in df_traders:
    traders_list = traders_list.split(",")
    traders_list = [trader.strip() for trader in traders_list]
    for trader in traders_list:
        traders.append(trader)

def tokenizer_sens_fct(text):
    inputs = tokenizer_sens(text, padding=True, truncation=True, return_tensors="pt")
    return inputs

def predict_sens(text):
    inputs = tokenizer_sens_fct(text)
    with torch.no_grad():
        outputs = model_sens(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return label_encoder_sens.classes_[predicted_class]
#%% date
default_month = "09"  # Septembre
default_year = "2023"

# Tous les formats de date possibles à prendre en compte
date_formats = [
    '%d/%m/%Y', '%d/%m/%y',  # Jour/Mois/Année à 4 ou 2 chiffres
    '%m/%d/%Y', '%m/%d/%y',  # Mois/Jour/Année à 4 ou 2 chiffres
    '%Y/%m/%d', '%y/%m/%d',  # Année à 4 ou 2 chiffres/Mois/Jour
    '%d-%m-%Y', '%d-%m-%y',  # Jour-Mois-Année à 4 ou 2 chiffres
    '%m-%d-%Y', '%m-%d-%y',  # Mois-Jour-Année à 4 ou 2 chiffres
    '%Y-%m-%d', '%y-%m-%d',  # Année à 4 ou 2 chiffres-Mois-Jour
    '%d %b %Y', '%d %b %y',  # Jour abréviation du Mois Année à 4 ou 2 chiffres
    '%b %d %Y', '%b %d %y',  # Abréviation du Mois Jour Année à 4 ou 2 chiffres
    '%d %B %Y', '%d %B %y',  # Jour Mois en toutes lettres Année à 4 ou 2 chiffres
    '%B %d %Y', '%B %d %y',  # Mois en toutes lettres Jour Année à 4 ou 2 chiffres
    # Note : '%d' seul ne nécessite pas de duplication car il n'inclut pas l'année
    '%d %b %Y', '%d %b %y',   # Jour abréviation du Mois Année à 4 ou 2 chiffres (dupliqué pour cohérence)
    '%d-%m',  # Jour-Mois sans année
    '%d/%m'   # Jour/Mois sans année, si non déjà inclus
    
]

def parse_date_search_near_keywords(text, keyword_patterns, search_window, is_settlement_date=False):
    for keyword_pattern in keyword_patterns:
        for keyword in re.finditer(keyword_pattern, text, re.IGNORECASE):
            end_pos = keyword.end()
            snippet = text[end_pos:end_pos + search_window + 50]  # Élargir la fenêtre de recherche
            date_match = re.search(r'\d{1,2}([\/\-]\d{1,2})([\/\-]\d{2,4})?', snippet)
            if date_match:
                date_str = date_match.group().strip()
                # Gérer les cas sans année
                if date_str.count('/') < 2 and date_str.count('-') < 2:
                    date_str += f"/{default_year}"
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, date_format)
                        return parsed_date, is_settlement_date
                    except ValueError:
                        continue
    return None, None


def find_date_with_fallbacks(text):
    # Tentatives de recherche pour Trade Date et Settlement Date séparément
    trade_date, is_trade_date_settlement = parse_date_search_near_keywords(text, [r'\btd\b', r'\btrade date\b'], 20)
    settlement_date, is_settlement_date = parse_date_search_near_keywords(text, [r'\bvd\b', r'\bsettlement date\b', r'\bbroker settlement date\b'], 20, True)
    
    # Si les deux dates sont trouvées mais marquées comme settlement, ajuster logiquement
    if trade_date and settlement_date and is_trade_date_settlement and is_settlement_date:
        if trade_date > settlement_date:
            # Si la trade_date est postérieure à settlement_date, inverser (improbable, mais pour la logique)
            trade_date, settlement_date = settlement_date, trade_date
    elif not trade_date and settlement_date and is_settlement_date:
        # Si seulement une date de règlement est trouvée, calculer la trade date
        trade_date = subtract_business_days(settlement_date, 2)
    elif trade_date and not settlement_date:
        # Si seulement une date de transaction est trouvée, calculer la settlement date
        settlement_date = add_business_days(trade_date, 2)

    return trade_date, settlement_date

    # Calcul de la date de règlement à partir de la date de transaction si nécessaire, et vice versa
    if trade_date and not settlement_date:
        settlement_date = add_business_days(trade_date, 2)
    elif settlement_date and not trade_date:
        trade_date = subtract_business_days(settlement_date, 2)

    return trade_date, settlement_date

def add_business_days(start_date, days_to_add):
    current_date = start_date
    while days_to_add > 0:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Samedi et dimanche non inclus
            days_to_add -= 1
    return current_date

def subtract_business_days(end_date, days_to_subtract):
    current_date = end_date
    while days_to_subtract > 0:
        current_date -= timedelta(days=1)
        if current_date.weekday() < 5:  # Samedi et dimanche non inclus
            days_to_subtract -= 1
    return current_date

def get_trade_and_settlement_dates_dict(text):
    trade_date, settlement_date = find_date_with_fallbacks(text)
    return {"trade_date":trade_date.strftime('%Y-%m-%d') if trade_date else None,"settlement_date":settlement_date.strftime('%Y-%m-%d') if settlement_date else None}


#%% spacy

chemin = "files/modele_1103"
nlp = spacy.load(chemin)
def find_closest_entityv3(entities, index, target_labels):
    closest_entity = None
    closest_distance = float('inf')
    for ent in entities:
        if ent.label_ in target_labels:
            # Pour les entités avant l'ISIN, utilisez ent.end_char pour calculer la distance
            if ent.end_char < index:
                distance = index - ent.end_char
            # Pour les entités après l'ISIN, utilisez ent.start_char
            else:
                distance = ent.start_char - index
            
            if distance < closest_distance:
                closest_entity = ent
                closest_distance = distance
    
    return closest_entity
def spacy_pred(textes):
    transactions = {}
    processed_isins = set()
    doc = nlp(textes)
    for ent in doc.ents:
        if ent.label_ == "ISIN":
            # Vérifie si l'ISIN a déjà été traité
            if ent.text in processed_isins:
            # Si oui, passe à la prochaine entité
                continue

        # Ajoute l'ISIN actuel à l'ensemble des ISINs traités
            processed_isins.add(ent.text)

        # Initialisation de l'entrée de transaction pour cet ISIN si elle n'existe pas déjà
            if ent.text not in transactions:
                transactions[ent.text] = {"ISIN": ent.text, "First_Entity": None, "Second_Entity": None}
            reference_point = (ent.start_char + ent.end_char) / 2

        # Trouver la première entité (Quantity ou Price) la plus proche de l'ISIN, avant ou après
            first_entity = find_closest_entityv3(doc.ents, reference_point, ["PRICE", "QUANTITY"])
            if first_entity:
                transactions[ent.text]["First_Entity"] = (first_entity.text, first_entity.label_)
            
            # Basé sur la première entité, déterminer quel type chercher ensuite 
                next_label = "PRICE" if first_entity.label_ == "QUANTITY" else "POURCENTAGE"
                second_entity = find_closest_entityv3(doc.ents, ent.end_char, [next_label])
                if second_entity:
                    transactions[ent.text]["Second_Entity"] = (second_entity.text, second_entity.label_)


    #%% identification

def transform(text):

    transformations = []
    mots = text.split()
    mots_frequent = Counter(mots).most_common(7)
    sens_ = predict_sens(text)
    dates = get_trade_and_settlement_dates_dict(text)
    transactions = {}
    processed_isins = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ISIN":
            # Vérifie si l'ISIN a déjà été traité
            if ent.text in processed_isins:
            # Si oui, passe à la prochaine entité
                continue

        # Ajoute l'ISIN actuel à l'ensemble des ISINs traités
            processed_isins.add(ent.text)

        # Initialisation de l'entrée de transaction pour cet ISIN si elle n'existe pas déjà
            if ent.text not in transactions:
                transactions[ent.text] = {"ISIN": ent.text, "First_Entity": None, "Second_Entity": None}
            reference_point = (ent.start_char + ent.end_char) / 2

        # Trouver la première entité (Quantity ou Price) la plus proche de l'ISIN, avant ou après
            first_entity = find_closest_entityv3(doc.ents, reference_point, ["PRICE", "QUANTITY"])
            if first_entity:
                transactions[ent.text]["First_Entity"] = (first_entity.text, first_entity.label_)
            
            # Basé sur la première entité, déterminer quel type chercher ensuite 
                next_label = "PRICE" if first_entity.label_ == "QUANTITY" else "POURCENTAGE"
                second_entity = find_closest_entityv3(doc.ents, ent.end_char, [next_label])
                if second_entity:
                    transactions[ent.text]["Second_Entity"] = (second_entity.text, second_entity.label_)
    isin = "not found"
    price = "not found"
    price_type = "not found"
    quantity = "not found"
    currency = "not found"
    dealer="not found"
    broker = "not found"
    trader_ = ""
    #for transaction,info in transactions.items():
    #    if "USD" in info["Second_Entity"][1]:
    #        currency = "USD"
    #    if "EUR" in info["Second_Entity"][1]:
    #        currency = "EUR"
    #    if "USD" in info["Second_Entity"][1]:
    #        currency = "USD"
    #    if "EUR" in info["Second_Entity"][1]:
    #        currency = "EUR" 
    if "USD" in text:
        currency = "USD"
    if "EUR" in text:
        currency = "EUR"       

    for transaction,info in transactions.items():
        isin = transaction
        if info["First_Entity"][1]=="PRICE":
            price = info["First_Entity"][0]
        if info["First_Entity"][1]=="QUANTITY":
            quantity=  info["First_Entity"][0]
            price_type = "quantity"
        if info["First_Entity"][1]=="POURCENTAGE":
            quantity=  info["First_Entity"][0]   
            price_type = "percentage" 
        if info["Second_Entity"][1]=="PRICE":
            price = info["Second_Entity"][0]
        if info["Second_Entity"][1]=="QUANTITY":
            quantity=  info["Second_Entity"][0]
            price_type = "quantity"
        if info["Second_Entity"][1]=="POURCENTAGE":
            quantity=  info["Second_Entity"][0]   
            price_type = "percentage"    
        try:      
            dealer = df_dealer[df_dealer["Unnamed: 0"] == isin]["Dealer"].values[0] 
        except Exception as e:
            dealer = "not found"   
        try:     
            broker = df_broker[df_broker["Contract ISIN"] == isin]["Primary broker"].values[0]
        except:
            broker = "not found"    
        #traders = df_dealer[df_dealer["Unnamed: 0"] == "CH81O34918F6"]["Traders"].values[0].split(",")   
        #traders = [trader.strip() for trader in traders]
        trader_ = "not found"
        for trader in traders:
            if trader in text:
                trader_ = trader 
        transformations.append(Order(text=text,isin=isin,trade_date=dates["trade_date"],settlement_date=dates["settlement_date"],primary_brocker=broker,sens=sens_,dealer = dealer,trader = trader_,price = price,size = quantity,price_type=price_type, currency = currency))





    #transformation = Order(text= text, isin = mots_frequent[0],trade_date=dates["trade_date"],settlement_date=dates["settlement_date"],primary_brocker=mots_frequent[1], sens = sens_, trader = mots_frequent[3], price = random.randint(1, 100), size = random.randint(1, 100), price_type = mots_frequent[4], currency = mots_frequent[5] )
    return transformations


#def transform(text):
#    return Order(text= text, isin = "US0378331005",trade_date=datetime.strptime("2024/01/28","%Y/%m/%d").date(),settlement_date=datetime.strptime("2024/03/18","%Y/%m/%d").date(),primary_brocker="JP Morgan", sens = "buys", trader = "Jhon Doe", price = 1000000000, size = 20000000, price_type = "%", currency = "EUR" )




