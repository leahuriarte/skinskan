from flask import Flask, render_template, request, jsonify
import numpy as np
from fastai.vision import *
import pickle 
import io
from fastai.text import *
import os
from fastai import learner
from fastai.vision.all import *
import PIL
import torchvision.transforms as T
import requests

cwd = os.getcwd()
path= cwd

app = Flask(__name__)

model = load_learner("all.pkl", cpu=True, pickle_module=pickle)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == 'POST':
        file = request.files['image'].read()
        open('facebook.jpg', 'wb').write(file)
        #else:
            #return render_template('index.html')

        # Resizing img to 224 X 224 , This is the size on which model was trained
        #img_fastai = PILImage.create(cancer_input)
        # Prediction using model
        prediction = model.predict("/Users/Leah/wound/facebook.jpg")[0]
        rec = ""
        rec2 = ""
        rec3 = ""
        rec4 = ""
        rec5 = ""
        sources = "static/img/"

        if prediction == "actinic keratosis":
            rec = """Actinic keratosis is a precancerous skin condition that is caused by long-term sun exposure. 
            It is characterized by rough, scaly patches or bumps that develop on sun-exposed areas of the skin, 
            such as the face, scalp, ears, neck, arms, and hands. Actinic keratosis is also known as solar keratosis, 
            and while it is not cancerous, it can increase the risk of developing skin cancer if left untreated."""
            rec2 = """It is important to manage actinic keratosis and protect your skin from further damage to reduce the risk of 
            developing skin cancer. If you have been diagnosed with actinic keratosis, it is important to take action to protect your 
            skin and prevent further damage. Actinic keratosis is a precancerous condition that is often caused by long-term sun 
            exposure, and it can increase your risk of developing skin cancer. To manage actinic keratosis, it is recommended that you 
            protect your skin from the sun by wearing protective clothing, using sunscreen with an SPF of at least 30, and avoiding 
            peak sun hours. You should also check your skin regularly for changes or new growths, and schedule regular check-ups with 
            your doctor or dermatologist. In some cases, treatment may be necessary to remove the affected areas of skin. This can 
            include cryotherapy, topical medications, or surgical removal. Your doctor can advise you on the best course of action 
            based on the severity of your condition."""
            rec3 = """Remember, prevention is key when it comes to actinic keratosis and skin cancer. Take steps to protect your skin, 
            and seek medical attention if you notice any changes or growths on your skin."""
            sources += "actinic-keratosis.png"
        elif prediction == "basal cell carcinoma":
            rec = """Basal cell carcinoma is a type of skin cancer that develops in the basal cells, which are located in the bottom 
            layer of the skin. This type of skin cancer is usually caused by long-term sun exposure, and it can appear as a shiny or 
            pearly bump, a red or pink patch, or a scaly or crusty lesion on the skin."""
            rec2 = """If you have been diagnosed with basal cell carcinoma, it is important to seek medical attention and follow your 
            doctor's recommendations for treatment. Treatment for basal cell carcinoma may involve surgical removal of the affected 
            area, radiation therapy, or topical medications."""
            rec3 = """To prevent basal cell carcinoma and other types of skin cancer, it is important to protect your skin from the 
            sun by wearing protective clothing, using sunscreen with an SPF of at least 30, and avoiding peak sun hours. You should 
            also check your skin regularly for changes or new growths, and schedule regular check-ups with your doctor or 
            dermatologist."""
            rec4 = """Early detection and treatment of basal cell carcinoma can increase the chances of a successful outcome, so if 
            you notice any unusual changes or growths on your skin, it is important to seek medical attention right away."""
            sources += "basal-cell-carcinoma.png"
        elif prediction == "dermatofibroma":
            rec = """Dermatofibroma is a common benign skin growth that often appears as a small, firm bump on the skin. It is 
            typically painless and may range in color from pink to brown or black. Dermatofibromas often develop on the legs, but they 
            can occur anywhere on the body."""
            rec2 = """If you have a dermatofibroma, it is usually not a cause for concern. However, if you notice any changes in size, 
            color, or texture of the bump, or if it becomes painful or starts to bleed, you should consult a dermatologist for further 
            evaluation."""
            rec3 = """Treatment for dermatofibroma is generally not necessary unless it is causing discomfort or is in a location 
            where it may be easily irritated or prone to injury. If treatment is desired, options may include surgical removal, 
            cryotherapy, or laser therapy."""
            rec4 = """While there is no known way to prevent dermatofibromas from developing, protecting your skin from excessive sun 
            exposure and avoiding injury to the skin may help reduce the risk of developing these growths. Regular skin checks with a 
            dermatologist can also help detect any changes in the skin and allow for early intervention if necessary."""
            sources += "dermatofibroma.png"
        elif prediction == "melanoma":
            rec = """Melanoma is a serious form of skin cancer that can develop from the pigment-producing cells in the skin. It is 
            often characterized by the appearance of a new mole or a change in an existing mole, and it may also cause itching, 
            bleeding, or ulceration."""
            rec2 = """If you have been diagnosed with melanoma, it is important to seek medical attention and follow your doctor's 
            recommendations for treatment. Treatment for melanoma may involve surgical removal of the affected area, radiation therapy, 
            or chemotherapy. In some cases, targeted therapy or immunotherapy may also be recommended."""
            rec3 = """To prevent melanoma and other types of skin cancer, it is important to protect your skin from the sun by wearing 
            protective clothing, using sunscreen with an SPF of at least 30, and avoiding peak sun hours. You should also check your 
            skin regularly for changes or new growths, and schedule regular check-ups with your doctor or dermatologist."""
            rec4 = """Early detection and treatment of melanoma is crucial for a successful outcome, so if you notice any unusual 
            changes or growths on your skin, it is important to seek medical attention right away. Additionally, individuals with a 
            family history of melanoma or a personal history of skin cancer may be at an increased risk and should be particularly 
            vigilant in monitoring their skin and seeking prompt medical attention if any changes are noted."""
            sources += "melanoma.png"
        elif prediction == "nevus":
            rec = """A nevus, also known as a mole, is a common benign growth on the skin. It is usually a result of clusters of 
            pigmented cells that appear as a dark spot on the skin. Most nevi are harmless, but in rare cases, they can become 
            cancerous."""
            rec2 = """If you have a nevus, it is usually not a cause for concern. However, if you notice any changes in size, color, 
            or texture of the mole, or if it becomes painful or starts to bleed, you should consult a dermatologist for further 
            evaluation."""
            rec3 = """If a nevus needs to be removed, the procedure is usually done under local anesthesia and is relatively simple. 
            Removal may be recommended if the nevus is bothersome or if it appears to be changing or growing."""
            rec4 = """To prevent nevi from developing, it is important to protect your skin from excessive sun exposure and to avoid 
            using tanning beds. Regular skin checks with a dermatologist can also help detect any changes in the skin and allow for 
            early intervention if necessary."""
            rec5 = """If you have a large number of nevi, particularly atypical or dysplastic nevi, you may be at an increased risk of
             developing melanoma. In this case, your dermatologist may recommend more frequent skin checks and possibly even a biopsy 
             of suspicious-looking nevi."""
            sources += "nevus.png"
        elif prediction == "pigmented benign keratosis":
            rec = """Pigmented benign keratosis, also known as seborrheic keratosis, is a common benign skin growth that typically 
            appears as a waxy, raised bump or patch on the skin. These growths are often brown or black and may appear on any part of 
            the body."""
            rec2 = """If you have a pigmented benign keratosis, it is usually not a cause for concern. However, if you notice any 
            changes in size, color, or texture of the growth, or if it becomes painful or starts to bleed, you should consult a 
            dermatologist for further evaluation."""
            rec3 = """Treatment for pigmented benign keratosis is generally not necessary unless the growth is causing discomfort or 
            is in a location where it may be easily irritated or prone to injury. If treatment is desired, options may include 
            surgical removal, cryotherapy, or laser therapy."""
            rec4 = """While there is no known way to prevent pigmented benign keratosis from developing, protecting your skin from 
            excessive sun exposure and avoiding injury to the skin may help reduce the risk of developing these growths. Regular skin
            checks with a dermatologist can also help detect any changes in the skin and allow for early intervention if necessary."""
            sources += "pigmented-benign-keratosis.png"
        elif prediction == "squamous cell carcinoma":
            rec = """Squamous cell carcinoma is a common type of skin cancer that often appears as a scaly, red or pink bump or patch 
            on the skin. It can also appear as an open sore that does not heal or a wart-like growth. Squamous cell carcinoma can be 
            locally invasive, meaning it can spread to nearby tissues, and in rare cases, it can metastasize or spread to other parts
             of the body."""
            rec2 = """If you have been diagnosed with squamous cell carcinoma, it is important to seek medical attention and follow 
            your doctor's recommendations for treatment. Treatment for squamous cell carcinoma may involve surgical removal of the 
            affected area, radiation therapy, or chemotherapy. In some cases, targeted therapy or immunotherapy may also be 
            recommended."""
            rec3 = """To prevent squamous cell carcinoma and other types of skin cancer, it is important to protect your skin from the
             sun by wearing protective clothing, using sunscreen with an SPF of at least 30, and avoiding peak sun hours. You should 
             also check your skin regularly for changes or new growths, and schedule regular check-ups with your doctor or 
             dermatologist."""
            rec4 = """Early detection and treatment of squamous cell carcinoma is crucial for a successful outcome, so if you notice 
            any unusual changes or growths on your skin, it is important to seek medical attention right away. Additionally, 
            individuals with a history of sun exposure or a weakened immune system may be at an increased risk and should be 
            particularly vigilant in monitoring their skin and seeking prompt medical attention if any changes are noted."""
            sources += "squamous-cell-carcinoma.png"
        elif prediction == "vascular lesion":
            rec = """Vascular lesions are abnormal growths of blood vessels in the skin, which can cause a range of cosmetic and 
            functional issues. There are several types of vascular lesions, including port wine stains, hemangiomas, and cherry 
            angiomas, among others."""
            rec2 = """The treatment for vascular lesions depends on the specific type and severity of the lesion. For smaller lesions, 
            laser therapy may be effective in removing or reducing the appearance of the lesion. In some cases, surgical removal or 
            other forms of medical therapy may be necessary."""
            rec3 = """It is important to seek medical attention if you have a vascular lesion that is causing discomfort or if you are 
            concerned about the appearance of the lesion. A dermatologist or other medical professional can provide an accurate 
            diagnosis and recommend the appropriate treatment for your specific situation."""
            rec4 = """While the exact cause of vascular lesions is not always clear, there are some known risk factors, including
            genetics, sun exposure, and certain medical conditions. Protecting your skin from sun exposure and maintaining a healthy
            lifestyle may help reduce the risk of developing vascular lesions. Regular skin checks with a dermatologist can also help 
            detect any changes in the skin and allow for early intervention if necessary."""
            sources += "vascular-lesion.png"
        elif prediction == "first":
            prediction = "a first degree burn"
            rec = "A first degree burn is a minor burn that affects only the outermost layer of the skin. These burns are often caused by brief exposure to heat, such as touching a hot surface, or exposure to the sun."
            rec2 = "The symptoms of a first degree burn include redness, pain, and mild swelling. The affected area may also be tender to the touch. In most cases, first degree burns can be treated at home with simple first aid measures."
            rec3 = "If you have a first degree burn, the first step is to cool the affected area with cool water or a cool compress. You can also apply aloe vera or a moisturizing lotion to the burn to help soothe the skin. Over-the-counter pain relievers, such as ibuprofen or acetaminophen, can help relieve pain and reduce swelling."
            rec4 = "It is important to keep the burned area clean and dry to prevent infection. Avoid using any products that may irritate the skin, such as perfumes or harsh soaps. If the burn is on a particularly sensitive area, such as the face or genitals, or if the burn covers a large area of the body, seek medical attention."
            rec5 = "To prevent first degree burns, be cautious when working with hot objects or around fire, and protect your skin from the sun with clothing and sunscreen."
            sources += "first.png"
        elif prediction == "second":
            prediction = "a second degree burn"
            rec = "A second degree burn is a burn that affects both the outermost layer of skin (the epidermis) and the layer beneath it (the dermis). These burns are often caused by exposure to flames, hot liquids, or prolonged exposure to the sun."
            rec2 = "The symptoms of a second degree burn include redness, blistering, and severe pain. The affected area may also be swollen and appear wet or shiny. Second degree burns can be more serious than first degree burns and may require medical attention."
            rec3 = "If you have a second degree burn, the first step is to cool the affected area with cool water or a cool compress for at least 10-15 minutes. Do not apply ice or butter, as this can worsen the burn. You can also apply a sterile gauze bandage to the burn to protect it and keep it clean."
            rec4 = "Over-the-counter pain relievers, such as ibuprofen or acetaminophen, can help relieve pain and reduce swelling. If the burn is on a particularly sensitive area, such as the face, hands, feet, or genitals, or if the burn covers a large area of the body, seek medical attention."
            rec5 = "Medical treatment for second degree burns may include cleaning the burn, applying a topical antibiotic ointment or cream, and possibly taking oral antibiotics to prevent infection. In some cases, a tetanus shot may also be recommended. More severe burns may require hospitalization and additional treatment, such as wound care, skin grafting, or surgery."
            sources += "second.png"
        elif prediction == "third":
            prediction = "a third degree burn"
            rec = "A third degree burn is a severe burn that affects all layers of the skin, as well as the underlying tissue. These burns are often caused by exposure to flames, chemicals, electricity, or prolonged exposure to hot liquids or objects."
            rec2 = "The symptoms of a third degree burn include white or blackened skin, a dry and leathery texture, and severe pain or numbness. Third degree burns are a medical emergency and require IMMEDIATE medical attention."
            rec3 = "If you or someone else has a third degree burn, call for emergency medical services immediately. Do NOT attempt to cool the affected area or apply any ointments or creams. Cover the burn with a clean, dry cloth or sterile bandage to prevent infection."
            rec4 = "Medical treatment for third degree burns may include fluid resuscitation to replace fluids lost due to the burn, pain management, and wound care. Depending on the severity of the burn, skin grafting or other surgical procedures may be necessary to promote healing and prevent complications."
            rec5 = "To prevent third degree burns, take precautions when working with hot objects, chemicals, or electricity. Wear protective clothing and follow safety guidelines when working with these materials. In case of a fire, evacuate the area immediately and call for emergency services."
            sources += "third.png"
        return render_template('results.html', prediction = prediction, sources = sources, rec = rec, rec2 = rec2, rec3 = rec3, rec4 = rec4, rec5 = rec5)

