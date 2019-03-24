#encoding:utf-8
# WEB后端服务程序
# 基于Python Flask 生成 RestfulAPI

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, Response
import time
import os
import uuid
import random
import string
import base64
import json
import numpy as np
import tensorflow as tf
from scipy import misc
from face_db import face_db
from configparser import ConfigParser
import facenet
import align.detect_face
import error

app = Flask(__name__)

# 加载配置文件
cfg = ConfigParser()
cfg.read('./_config.ini')

app.config['UPLOAD_FOLDER'] = cfg.get('restful', 'upload_dir')
app.config['CROP_IMG_FOLDER'] = cfg.get('restful', 'crop_face_dir')
basedir = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(os.path.join(basedir, app.config['UPLOAD_FOLDER'])):
    os.makedirs(os.path.join(basedir, app.config['UPLOAD_FOLDER']))
if not os.path.exists(os.path.join(basedir, app.config['CROP_IMG_FOLDER'])):
    os.makedirs(os.path.join(basedir, app.config['CROP_IMG_FOLDER']))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG'])

# 使用ASCII随机生成face_token，默认长度为24
def gen_face_token(len=24):
    salt = ''.join(random.sample(string.ascii_letters + string.digits, len))
    return salt

# 允许上传文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# JSONEncoder 重写
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# 检测人脸位置
def DetectFaceLocation(ImagePath):
    minsize = 20 # 最小人脸尺寸
    threshold = [ 0.6, 0.7, 0.7 ]  # 阈值
    factor = 0.709

    ImgArray = misc.imread(os.path.expanduser(ImagePath), mode='RGB')
    bounding_boxes, _ = align.detect_face.detect_face(ImgArray, minsize, pnet, rnet, onet, threshold, factor)
    det_result = bounding_boxes.astype(int)
    if len(det_result) < 1:
        return False, None
    return True, det_result

# 生成人脸向量模型
def EmbeddingFace(ImagePath):
    images = []
    img = misc.imread(ImagePath, mode='RGB')
    image = facenet.prewhiten(img)
    images.append(image)
    img_stack = np.stack(images)
    feed_dict = { images_placeholder: img_stack, phase_train_placeholder:False }
    img_emb = sess.run(embeddings, feed_dict=feed_dict)
    return img_emb

# 人脸剪切对齐
# return face json
def ImgFaceCrop(image_path, new_image_size, face_location):
    # 读取图片
    img = misc.imread(image_path, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    det = face_location
    det[:,0] = np.maximum(det[:,0], 0)
    det[:,1] = np.maximum(det[:,1], 0)
    det[:,2] = np.minimum(det[:,2], img_size[1])
    det[:,3] = np.minimum(det[:,3], img_size[0])
    
    # face剪切对齐后保存文件夹
    crop_dir = os.path.join(basedir, app.config['CROP_IMG_FOLDER'])
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    # 原图的人脸名称及位置
    faces = []
    for i in range(len(face_location)):
        temp_crop = img[det[i,1]:det[i,3], det[i,0]:det[i,2], :]
        aligned = misc.imresize(temp_crop, (new_image_size, new_image_size), interp='bilinear')
        crop_name_pre = str(uuid.uuid1())
        crop_name = crop_name_pre + '.jpg'
        new_img_path = os.path.join(crop_dir, crop_name)
        misc.imsave(new_img_path, aligned)
        faces.append({
            'face_name': crop_name,
            'top_left': (face_location[i,0],face_location[i,1]),
            'bottom_right': (face_location[i,2], face_location[i,3])
        })
    return faces

# 加载上传文件页面
@app.route('/upload')
def upload():
    return render_template(cfg.get('templates', 'upload_html'))
 
# Face detect api
@app.route('/detect', methods=['POST'], strict_slashes=False)
def detect():
    if request.method != "POST":
        pass
    upload_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    crop_face_dir = os.path.join(basedir, app.config['CROP_IMG_FOLDER'])
    f = request.files['image_file']
    if f and allowed_file(f.filename):
        # 使用 secure_filename 防止 恶意 url 攻击
        try:
            fname = secure_filename(f.filename)
            ext = fname.split('.', 1)[1]
            new_filename = str(uuid.uuid1()) + '.' + ext
            new_file_path = os.path.join(upload_dir, new_filename)
            # 存储image文件到本地
            f.save(new_file_path)
            print("Save " + new_filename + " sucess!")
        except Exception as ex:
            return error.get_error("1004", str(ex))
        # 检测人脸位置
        status, face_location = DetectFaceLocation(new_file_path)
        if status:
            face_num = len(face_location)
            faces = ImgFaceCrop(new_file_path, 160, face_location)
            for face in faces:
                face_path = os.path.join(crop_face_dir, face['face_name'])
                face_dist = EmbeddingFace(face_path)
                face_token = gen_face_token()
                face['face_token'] = face_token
                face_info.append({
                    "face_token": face_token, 
                    "face_dist": face_dist, 
                    "face_path": face_path
                })
        else:
            face_num = 0
            faces = None
        return jsonify({"sucess": True, "face_num": face_num, "faces": faces})
    else:
        return error.get_error("1005", error_msg=None)

# 查找相似人脸信息
@app.route('/search', methods={'POST', 'GET'})
def search():
    if request.method == "GET":
        face_token = request.args.get('face_token')
        threshold = request.args.get('threshold')
        if face_token is None or threshold is None:
            # 输入参数不完整
            return error.get_error("1002", error_msg=None)
        else:
            face_set = FaceDB.get_face_info_all()
            not_found = True
            for face in face_info:
                if(face['face_token'] == face_token):
                    face_dist_cmp = face['face_dist']
                    not_found = False
            if not_found:
                # face_token 在 face_info 中没有找到
                return error.get_error("1006", error_msg=None)
            re_info = []
            for face in face_set:
                face_set_token = face['face_token']
                face_set_dist = face['face_dist']
                face_set_name = face['face_name']
                dist = np.sqrt(np.sum(np.square(np.subtract(face_dist_cmp, face_set_dist))))
                if face_token != face_set_token and dist <= float(threshold):
                    re_face = {}
                    re_face['face_token'] = face_set_token
                    re_face['distant'] = dist
                    re_face['face_name'] = face_set_name
                    re_info.append(re_face)
            return jsonify({
                "sucess": True, 
                "face_token": face_token, 
                "threshlod": threshold, 
                "cmp_result": re_info
            })
    elif request.method == 'POST':
        try:
            # 获取上传参数
            f = request.files['image_file']
            threshold = request.form['threshold']
        except:
            # 上传参数获取出错
            return error.get_error("1005", error_msg=None)
        if f is None or threshold == '':
            # 输入参数不完整
            return error.get_error("1002", error_msg=None)
        else:
            upload_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
            crop_face_dir = os.path.join(basedir, app.config['CROP_IMG_FOLDER'])
            if allowed_file(f.filename):
                try:
                    fname = secure_filename(f.filename)
                    ext = fname.split('.', 1)[1]
                    new_filename = str(uuid.uuid1()) + '.' + ext
                    new_file_path = os.path.join(upload_dir, new_filename)
                    # 存储image文件到本地
                    f.save(new_file_path)
                    print("Save " + new_filename + " sucess!")
                except Exception as ex:
                    return error.get_error("1004", str(ex))
                # 检测人脸位置
                status, face_location = DetectFaceLocation(new_file_path)
                if status:
                    # 获取最大的人脸图像
                    face_op = ImgFaceCrop(new_file_path, 160, face_location)[0]
                    face_path_op = os.path.join(crop_face_dir, face_op['face_name'])
                    # 生成向量
                    face_dist_op = EmbeddingFace(face_path_op)
                    # 生成face_token
                    face_token_op = gen_face_token()
                    face_op['face_token'] = face_token_op
                    # 向内存的face_info中添加该人脸信息
                    face_info.append({
                        "face_token": face_token_op, 
                        "face_dist": face_dist_op, 
                        "face_path": face_path_op
                    })
                    # 从数据库FaceDB中获得face_set
                    face_set = FaceDB.get_face_info_all()
                    # 返回数据
                    re_info = []
                    for face in face_set:
                        face_set_token = face['face_token']
                        face_set_dist = face['face_dist']
                        face_set_name = face['face_name']
                        # 计算欧式空间距离
                        dist = np.sqrt(np.sum(np.square(np.subtract(face_dist_op, face_set_dist))))
                        if face_token_op != face_set_token and dist <= float(threshold):
                            re_face = {}
                            re_face['face_token'] = face_set_token
                            re_face['distant'] = dist
                            re_face['face_name'] = face_set_name
                            re_info.append(re_face)
                    return jsonify({
                        "sucess": True, 
                        "face_token": face_token_op,
                        "threshlod": threshold, 
                        "cmp_result": re_info
                    })
                else:
                    # 未检测到人脸信息
                    return error.get_error("1007", error_msg=None)
            else:
                # 文件错误
                return error.get_error("1004", error_msg=None)
    else:
        pass

# 向 face_set 中添加人脸信息
@app.route('/addface', methods=['GET'])
def addface():
    if request.method != 'GET':
        pass
    face_token = request.args.get('face_token')
    face_name = request.args.get('face_name')
    if face_token is None:
        # 输入参数不完整
        return error.get_error("1002", error_msg=None)
    if face_name is None:
        face_name = 'unkonwn'
    not_found = True
    for face in face_info:
        if(face['face_token'] == face_token):
            # try:
            not_found = False
            add_face_token = face_token
            add_face_dist = face['face_dist']
            add_face_name = face_name
            add_face_path = face['face_path']
            flag = FaceDB.insert_face(add_face_token, add_face_dist, add_face_name, add_face_path)
            if flag:
                return jsonify({
                    "sucess": True,
                    "face_token": add_face_token,
                    "face_neme": add_face_name
                })
            else:
                return error.get_error("1003", error_msg=None)
            # except Exception:
            #     return error.get_error("1003", error_msg=None)
            break
    if not_found:
        return error.get_error("1006", error_msg=None)

# 从face_set中删除人脸信息
@app.route('/removeface', methods=['GET'])
def removeface():
    if request.method != 'GET':
        pass
    face_token_op = request.args.get('face_token')
    if face_token_op is None:
        # 输入参数不完整
        return error.get_error("1002", error_msg=None)
    flag = FaceDB.delete_face(face_token_op)
    if flag:
        return jsonify({
            "sucess": True,
            "face_token": face_token_op,
        })
    else:
        return error.get_error("1003", error_msg=None)
    
# 下载图片
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass

# 显示图片
@app.route('/show_photo/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass
 
if __name__ == '__main__':
    # 创建load_and_align_data网络
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            # 加载模型
            model=cfg.get('model', 'model_dir')
            facenet.load_model(model)

            # 获取输入输出tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # 加载人脸数据库
    FaceDB = face_db()
    # 连接数据库
    FaceDB.conn_face_db()
    # 从数据库中加载人脸信息
    # 使用列表在内存中保存的人脸信息，[{"face_tohen":"", "face_dist": ""}]
    face_info = FaceDB.get_face_info_all()
    # 开启Resultful服务
    app.json_encoder = MyEncoder
    # 加载ERROR类
    error = error.ERROR()
    app.run(debug=True)
