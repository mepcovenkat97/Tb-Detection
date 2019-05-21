from django.shortcuts import render
from django.shortcuts import render_to_response
from django.shortcuts import redirect
import requests
from django.template import loader  
# Create your views here.  
from django.http import HttpResponse  
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.core.files.storage import FileSystemStorage  
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image, ImageFile

# Create your views here.
def index(requests):
    print("Inside index \n");
    template = loader.get_template('index.html') # getting our template  
    return HttpResponse(template.render())  
   #return render_to_response('index.html')

@csrf_exempt
def report(request):
   print("Inside Report \n")
   my_file=""
   result=""
   prediction=""
   nname=""
   nage=""
   template = loader.get_template('report.html')
   if request.method == 'POST':
       handle_uploaded_file(request.FILES['myfile'])
       my_file = request.FILES['myfile']
       result,prediction=predictTB(my_file)
       nname = request.POST.get('name');
       nage = request.POST.get('age');
       createReport(nname,nage,my_file,result,prediction)
   #return HttpResponse(template.render(name))
   print("Report Done")
   print ("Downloaded")
   fs1 = FileSystemStorage()
   report_flag = fs1.url("/static/report/"+nname+".pdf")
   return render(request,'report.html',{'name':request.POST.get('name'),'age':request.POST.get('age'),'filename':my_file,'result':result,'prediction':prediction,'report_flag':report_flag})

def handle_uploaded_file(f):
   with open('tb/static/upload/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)
@csrf_exempt
def predictTB(f):
   from keras import backend as P
   P.clear_session()
   classifier = load_model('tb/Model.h5')
   test_image = image.load_img(f, target_size = (64, 64))
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis = 0)
   result = classifier.predict(test_image)
   dic = {'No':0, 'Yes':1};
   #training_set.class_indices
   res = result[0][0] * 100;
   if (res >= 60 and res <= 99):
       prediction = 'Normal'
   else:
       prediction = 'Abnormal'
   return round(res,2),prediction
   #return 1,2

def createReport(nname,nage,ima,res,pre):
    print("Inside createReport \n")
    from fpdf import FPDF
    from datetime import date,time
    pdf = FPDF(format="A4")
    pdf.add_page()
    pdf.set_font("Arial", size=32)
    pdf.cell(200, 10, txt="Tuberculosis Detection Report", ln=1, align="C")
    pdf.set_font("Arial", size=20)
    #pdf.cell(200, 10, txt="A Project done by D.R.V", ln=1, align="C")
    pdf.cell(200, 10, txt="\n", ln=1, align="C")
    dt = date.today();
    pdf.cell(180, 10, txt="Date :"+ str(dt.strftime("%d %B %Y")), ln=1, align="R")
    pdf.set_font("Arial", size=15)
    pdf.cell(200, 10, txt="\n", ln=1, align="C")
    pdf.cell(200, 12, txt="Name   : "+nname, ln=1, align="L")
    pdf.cell(200, 12, txt="Age    : "+nage, ln=1, align="L")
    pdf.cell(200, 12, txt="Result : "+str(pre)+" with "+str(res)+"%", ln=1, align="L")
    pdf.ln(85)
    pdf.image('tb/static/upload/'+ima.name, x=15, y=90, w=180,h=150)
    pdf.set_font("Arial", size=12)
    pdf.image('tb/static/icons/sign.png', x=120, y=255,w=100)
    pdf.ln(94)
    pdf.cell(180, 10, txt="Senior Consultant", ln=1, align="R")
    pdf.output("tb/static/report/"+nname+".pdf")

@csrf_exempt
def downloadReport(requests):
    print ("Downloaded")
    fs1 = FileSystemStorage()
    report_flag = fs1.url("/tb/static/report/drv.pdf")
    """file_name = 'drv.pdf'
    file_path = '/tb/static/report/drv.pdf'
    from wsgiref.util import FileWrapper
    import mimetypes
    from django.utils.encoding import smart_str
    import os
    file_wrapper = FileWrapper(open(file_path,'rb'))
    file_mimetype = mimetypes.guess_type(file_path)
    response = HttpResponse(file_wrapper, content_type=file_mimetype )
    response['X-Sendfile'] = file_path
    response['Content-Length'] = os.stat(file_path).st_size
    response['Content-Disposition'] = 'attachment; filename=%s' % smart_str(file_name)"""
    #return 1
    