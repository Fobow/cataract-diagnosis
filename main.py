# coding:utf-8
import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
from PIL import Image
from utils import run, initialize_model

UPLOAD_FOLDER = r'static\uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = '123456'

label_set = ['眼球健康: ','皮质型白内障: ']
shape_set = ['形态规则, 大小正常,','轴距增大,','白内障术后,大小正常,']
thickness_set = ['边缘未见增厚,','边缘稍增厚,','边缘明显增厚,']
echo_set = ['内未见异常回声。','内见点絮状回声漂浮。']

basedir = os.path.abspath(os.path.dirname(__file__))
filedir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])

model = initialize_model()
print('model initialized!')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
@app.route('/ajax')
def rt_result():
    print('in ajax')
    print(result_txt[0])
    return jsonify(result = result_txt)
'''

@app.route('/')
def show_index():
    return render_template('index.html')

@app.route('/project_list')
def show_project():
    return render_template('project-list.html')

@app.route('/bnz')
def show_bnz():
    return render_template('bnz.html')

@app.route('/team')
def show_team():
    return render_template('team.html')

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #print files
        #print(request.files)
        #print inputs
        #print(request.form)
        # check if the post request has the file part
        if 'file0' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file0']
        size = request.form
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img = Image.open(file.stream)
            img = crop(img,size)
            imgpath = os.path.join(filedir, filename)
            img.save(imgpath,'jpeg')
            result = run(model, imgpath)
            result_str = getDescription(result)
            return jsonify(result = result_str)
            #return redirect(url_for('uploaded_file',filename=filename))
    return jsonify(result = '')
        
    

def getDescription(result):
    description = ""
    label = result['label'][0]
    shape = result['shape'][0]
    thickness = result['thickness'][0]
    echo = result['echo'][0]

    description = description + label_set[label] + '眼球' + shape_set[shape] + '轮廓清楚, 前房清亮, 晶状体附着位置正常,' +\
        thickness_set[thickness] + '玻璃体内透声尚可,' + echo_set[echo]
    
    return description

def crop(img,size_dict):
    imgw,imgh = img.size
    size = size_dict.to_dict()
    w = 465
    h = 465
    if(imgw > w or imgh > h):
        #图片尺寸过大, 对img进行剪裁操作（剪裁为方形），得到img1
        if(imgw / imgh > w / h):
            #偏宽的图片
            #print('too wide')
            left = (imgw - imgh)/2
            right = left + imgh
            top = 0
            bottom = imgh
            img = img.crop((left, top, right, bottom))
        elif(imgw / imgh < w / h):
            #偏高的图片
            #print('too high')
            top = (imgh - imgw)/2
            bottom = top +imgw
            left = 0
            right = imgw
            img = img.crop((left, top, right, bottom))
    #与图片框成比例的图片不需要剪裁，直接进行缩放

    #把img1按比例缩放
    newsize = (465,465)
    img = img.resize(newsize)

    #按jcrop剪裁框坐标剪裁图片，得到结果img2
    if(size['x'] != '' and size['y'] != '' and size['x2'] != '' and size['y2'] != ''):
        x = int(size['x'])
        y = int(size['y'])
        x2 = int(size['x2'])
        y2 = int(size['y2'])
        box = (x,y,x2,y2)
        img = img.crop(box)
    return img


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    img = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return img

if __name__ == '__main__':
    app.run()